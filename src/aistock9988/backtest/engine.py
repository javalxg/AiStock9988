"""Small, deterministic event-driven backtest engine.

The engine consumes frozen selection signals and a raw execution-price panel.
It deliberately keeps research returns and accounting prices separate: only
``raw_open``/``raw_close`` are used for orders and NAV.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..execution.prices import validate_execution_panel
from ..execution.corporate_actions import CorporateAction, apply_action
from ..execution.intraday import find_stop_execution
from ..execution.risk import evaluate_close_stop_loss
from ..data.minute_source import normalize_minute_panel


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 1_000_000.0
    max_positions: int = 2
    hold_sessions: int = 5
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    stop_loss_mode: str = "close_next_session_open"
    order_id_prefix: str = "bt"
    buy_commission: float = 0.0003
    sell_commission: float = 0.0003
    stamp_duty: float = 0.0005
    lot_size: int = 1
    buy_slippage: float = 0.0
    sell_slippage: float = 0.0


def run_backtest(signals: pd.DataFrame, prices: pd.DataFrame, *, config: BacktestConfig = BacktestConfig(),
                 corporate_actions: pd.DataFrame | None = None,
                 minute_prices: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Run a causal long-only backtest and return trades, positions and NAV.

    ``signals`` must contain ``asof``, ``ts_code`` and ``candidate_rank``.
    ``prices`` must contain one raw OHLC row per security/session with
    ``trade_date``, ``raw_open`` and ``raw_close``.
    """
    _validate_config(config)
    required_signal = {"asof", "ts_code", "candidate_rank"}
    required_price = {"trade_date", "ts_code", "raw_open", "raw_close"}
    if missing := required_signal - set(signals.columns):
        raise ValueError(f"signals missing columns: {sorted(missing)}")
    if missing := required_price - set(prices.columns):
        raise ValueError(f"prices missing columns: {sorted(missing)}")
    sig = signals.copy()
    px = prices.copy()
    required_economic = {"economic_open", "economic_high", "economic_low", "economic_close", "raw_high", "raw_low", "adj_factor"}
    if not required_economic <= set(px.columns):
        raise ValueError("prices must provide the explicit raw/economic execution contract")
    px = validate_execution_panel(px)
    minute = normalize_minute_panel(minute_prices) if minute_prices is not None else None
    actions = _prepare_actions(corporate_actions)
    sig["asof"] = _dates(sig["asof"])
    px["trade_date"] = _dates(px["trade_date"])
    for col in ("raw_open", "raw_close"):
        px[col] = pd.to_numeric(px[col], errors="raise")
        if (px[col] <= 0).any():
            raise ValueError(f"{col} must be positive")
    if px.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("prices contain duplicate security/session keys")
    sessions = sorted(px["trade_date"].unique())
    by_key = px.set_index(["trade_date", "ts_code"])
    cash = float(config.initial_cash)
    positions: dict[str, dict] = {}
    trades: list[dict] = []
    orders: list[dict] = []
    nav_rows: list[dict] = []
    action_rows: list[dict] = []
    signal_map = {d: g.sort_values(["candidate_rank", "ts_code"], kind="mergesort")
                  for d, g in sig.groupby("asof", sort=True)}

    order_sequence = 0

    def new_order(day, code, side, shares, reason, *, decision_session=None,
                  trigger_price=None, trigger_return=None, trigger_type=None,
                  reference_raw_close=None):
        nonlocal order_sequence
        order_sequence += 1
        order_id = f"{config.order_id_prefix}-{order_sequence:08d}"
        order = {
            "order_id": order_id, "ts_code": code, "side": side,
            "decision_session": decision_session if decision_session is not None else day,
            "trigger_session": decision_session if trigger_type else None,
            "trigger_type": trigger_type, "trigger_price": trigger_price,
            "trigger_return": trigger_return, "requested_shares": shares,
            "reference_raw_close": reference_raw_close,
            "status": "PENDING", "execution_session": None,
            "execution_price": None, "execution_economic_price": None,
            "gap_return": None, "filled_shares": 0,
            "final_reason": reason, "last_attempt_reason": None,
        }
        orders.append(order)
        return order

    def execute(day, code, side, price, shares, reason, order, *, economic_price,
                entry_economic_price=None):
        nonlocal cash
        gross = price * shares
        commission = gross * (config.buy_commission if side == "BUY" else config.sell_commission)
        duty = gross * config.stamp_duty if side == "SELL" else 0.0
        cash_change = -(gross + commission) if side == "BUY" else gross - commission - duty
        cash += cash_change
        order["status"] = "FILLED"
        order["execution_session"] = day
        order["execution_price"] = price
        order["execution_economic_price"] = economic_price
        if order.get("reference_raw_close") is not None:
            order["gap_return"] = price / order["reference_raw_close"] - 1.0
        order["filled_shares"] = shares
        order["final_reason"] = reason
        trades.append({"order_id": order["order_id"], "trade_date": day, "ts_code": code, "side": side, "price": price,
                       "shares": shares, "gross_value": gross, "commission": commission,
                       "stamp_duty": duty, "cash_after": cash, "reason": reason,
                       "trigger_type": order.get("trigger_type"),
                       "trigger_session": order.get("trigger_session"),
                       "trigger_price": order.get("trigger_price"),
                       "trigger_return": order.get("trigger_return"),
                       "execution_economic_price": economic_price,
                       "economic_return": (economic_price / entry_economic_price - 1.0
                                           if entry_economic_price is not None else None),
                       "gap_return": order.get("gap_return"),
                       "gap_flag": bool(order.get("gap_return") is not None and abs(order["gap_return"]) > 1e-12)})
        return gross + commission if side == "BUY" else gross - commission - duty

    for i, day in enumerate(sessions):
        # Company actions are applied before the day's mark and execution.
        for action in actions.get(day, []):
            pos = positions.get(action.ts_code)
            if pos is None:
                continue
            dividend = apply_action(pos, action)
            cash += dividend
            action_rows.append({"trade_date": day, "ts_code": action.ts_code,
                                "split_ratio": action.split_ratio, "cash_dividend": dividend,
                                "cash_after": cash})
        # Decisions made at yesterday's close become executable at today's open.
        for code, pos in list(positions.items()):
            stop = evaluate_close_stop_loss(
                entry_economic_price=pos["entry_economic_price"],
                mark_economic_price=pos["last_economic_close"],
                stop_loss_pct=None if minute is not None else config.stop_loss_pct,
                trigger_session=previous_session(sessions, i),
            )
            take_profit = config.take_profit_pct is not None and (
                pos["last_economic_close"] / pos["entry_economic_price"] - 1.0
                >= config.take_profit_pct
            )
            should_exit = pos.get("exit_order") is not None or (pos["exit_due"] == day) or stop.triggered or take_profit
            if should_exit and pos.get("exit_order") is None:
                trigger_type = "STOP_LOSS" if stop.triggered else "TAKE_PROFIT" if take_profit else "SCHEDULED_EXIT"
                trigger_price = stop.trigger_price if stop.triggered else pos["last_economic_close"]
                trigger_return = stop.trigger_return if stop.triggered else pos["last_economic_close"] / pos["entry_economic_price"] - 1.0
                pos["exit_order"] = new_order(
                    day, code, "SELL", pos["shares"], "awaiting_next_tradable_session",
                    decision_session=previous_session(sessions, i),
                    trigger_price=trigger_price, trigger_return=trigger_return,
                    trigger_type=trigger_type, reference_raw_close=pos["last_close"],
                )
            order = pos.get("exit_order")
            if should_exit:
                row = by_key.loc[(day, code)] if (day, code) in by_key.index else None
                if row is not None and not bool(row.is_suspended) and not bool(row.is_limit_down):
                    execute(day, code, "SELL", float(row.raw_open) * (1.0 - config.sell_slippage), pos["shares"],
                            "stop_loss_exit" if order.get("trigger_type") == "STOP_LOSS" else "scheduled_or_rule_exit", order,
                            economic_price=float(row.economic_open), entry_economic_price=pos["entry_economic_price"])
                    del positions[code]
                elif row is not None:
                    order["last_attempt_reason"] = "suspended" if bool(row.is_suspended) else "limit_down"
                    order["final_reason"] = "pending_non_tradable"
                else:
                    order["last_attempt_reason"] = "missing_execution_price"
                    order["final_reason"] = "pending_missing_price"

        # The signal observed on asof is filled on the next available session.
        previous = sessions[i - 1] if i else None
        if previous in signal_map:
            available = signal_map[previous]
            slots = config.max_positions - len(positions)
            for code in available.head(max(0, slots))["ts_code"]:
                code = str(code)
                if code in positions or (day, code) not in by_key.index:
                    continue
                row = by_key.loc[(day, code)]
                if bool(row.is_suspended) or bool(row.is_limit_up):
                    order = new_order(previous, code, "BUY", 0, "not_tradable_at_open", decision_session=previous)
                    order["status"] = "REJECTED"
                    order["last_attempt_reason"] = "suspended" if bool(row.is_suspended) else "limit_up"
                    continue
                price = float(row.raw_open) * (1.0 + config.buy_slippage)
                shares = int((cash / max(1, config.max_positions) / price) // config.lot_size) * config.lot_size
                if shares <= 0:
                    continue
                order = new_order(previous, code, "BUY", shares, "signal_entry", decision_session=previous)
                execute(day, code, "BUY", price, shares, "signal_entry", order,
                        economic_price=float(row.economic_open), entry_economic_price=None)
                exit_idx = i + config.hold_sessions
                positions[code] = {"shares": shares, "entry_price": price, "entry_economic_price": float(row.economic_open), "entry_date": day,
                                   "exit_due": sessions[exit_idx] if exit_idx < len(sessions) else None,
                                   "last_close": float(row.raw_close), "last_economic_close": float(row.economic_close)}
                positions[code]["exit_order"] = None

        for code, pos in positions.items():
            pos["prior_close_for_intraday"] = pos["last_close"]
            if (day, code) in by_key.index:
                pos["last_close"] = float(by_key.loc[(day, code)].raw_close)
                pos["last_economic_close"] = float(by_key.loc[(day, code)].economic_close)
        # With minute data, stop-loss decisions are made inside the current
        # session. A position cannot be sold on its entry session (A-share T+1).
        if minute is not None and config.stop_loss_pct is not None:
            for code, pos in list(positions.items()):
                if pos["entry_date"] >= day or pos.get("exit_order") is not None:
                    continue
                bars = minute[(minute["trade_date"] == day) & (minute["ts_code"].astype(str) == code)]
                if bars.empty:
                    continue
                stop_result = find_stop_execution(
                    bars, entry_economic_price=pos["entry_economic_price"],
                    stop_loss_pct=config.stop_loss_pct, start_time=day,
                )
                if stop_result.status == "NOT_TRIGGERED":
                    continue
                pos["exit_order"] = new_order(
                    day, code, "SELL", pos["shares"], stop_result.reason,
                    decision_session=day, trigger_price=stop_result.trigger_economic_price,
                    trigger_return=config.stop_loss_pct, trigger_type="STOP_LOSS",
                    reference_raw_close=pos["prior_close_for_intraday"],
                )
                if stop_result.status == "FILLED":
                    fill_row = bars[bars["trade_time"] == stop_result.execution_time].iloc[0]
                    execute(day, code, "SELL", float(stop_result.execution_raw_price), pos["shares"],
                            "intraday_stop_loss", pos["exit_order"],
                            economic_price=float(fill_row.economic_open),
                            entry_economic_price=pos["entry_economic_price"])
                    del positions[code]
                else:
                    pos["exit_order"]["last_attempt_reason"] = stop_result.reason
                    pos["exit_order"]["final_reason"] = "pending_intraday_non_tradable"
        mark = 0.0
        for code, pos in positions.items():
            if (day, code) in by_key.index:
                mark += pos["shares"] * float(by_key.loc[(day, code)].raw_close)
        nav_rows.append({"trade_date": day, "cash": cash, "market_value": mark, "nav": cash + mark,
                         "open_positions": len(positions)})

    # Final liquidation is explicitly marked and uses the last available close.
    final_day = sessions[-1]
    residual = []
    for code, pos in list(positions.items()):
        if (final_day, code) in by_key.index:
            row = by_key.loc[(final_day, code)]
            if not bool(row.is_suspended) and not bool(row.is_limit_down):
                order = pos.get("exit_order") or new_order(final_day, code, "SELL", pos["shares"],
                                                            "end_of_test_liquidation", decision_session=final_day,
                                                            trigger_type="END_OF_TEST", reference_raw_close=pos["last_close"])
                execute(final_day, code, "SELL", float(row.raw_close) * (1.0 - config.sell_slippage), pos["shares"], "end_of_test_liquidation", order,
                        economic_price=float(row.economic_close), entry_economic_price=pos["entry_economic_price"])
                del positions[code]
                continue
        if pos.get("exit_order") is not None:
            pos["exit_order"]["status"] = "EXPIRED"
            pos["exit_order"]["final_reason"] = "unclosed_non_tradable"
        residual.append({"trade_date": final_day, "ts_code": code, "shares": pos["shares"],
                         "raw_mark_price": float(by_key.loc[(final_day, code)].raw_close) if (final_day, code) in by_key.index else None,
                         "reason": "unclosed_non_tradable"})
    if nav_rows and trades and not positions:
        nav_rows[-1]["cash"] = cash
        nav_rows[-1]["market_value"] = 0.0
        nav_rows[-1]["nav"] = cash
        nav_rows[-1]["open_positions"] = 0
    if nav_rows and positions:
        mark = sum(pos["shares"] * float(by_key.loc[(final_day, code)].raw_close)
                   for code, pos in positions.items() if (final_day, code) in by_key.index)
        nav_rows[-1]["cash"] = cash
        nav_rows[-1]["market_value"] = mark
        nav_rows[-1]["nav"] = cash + mark
        nav_rows[-1]["open_positions"] = len(positions)
    return {"trades": pd.DataFrame(trades), "orders": pd.DataFrame(orders), "nav": pd.DataFrame(nav_rows),
            "positions": pd.DataFrame(residual), "corporate_actions": pd.DataFrame(action_rows)}


def _dates(values: pd.Series) -> list[pd.Timestamp]:
    out = pd.to_datetime(values, utc=True).dt.normalize()
    return out.tolist()


def _validate_config(config: BacktestConfig) -> None:
    if config.initial_cash <= 0 or config.max_positions <= 0 or config.hold_sessions <= 0 or config.lot_size <= 0:
        raise ValueError("cash, positions, hold_sessions and lot_size must be positive")
    if config.buy_slippage < 0 or config.sell_slippage < 0:
        raise ValueError("slippage must be non-negative")
    if config.stop_loss_pct is not None and (config.stop_loss_pct >= 0 or config.stop_loss_pct <= -1):
        raise ValueError("stop_loss_pct must be negative, greater than -1, and expressed as a ratio")
    if config.take_profit_pct is not None and config.take_profit_pct <= 0:
        raise ValueError("take_profit_pct must be positive and expressed as a ratio")
    if config.stop_loss_mode != "close_next_session_open":
        raise ValueError("only close_next_session_open is supported with daily data")
    if not config.order_id_prefix or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in config.order_id_prefix):
        raise ValueError("order_id_prefix must contain only safe ASCII characters")


def previous_session(sessions: list[pd.Timestamp], index: int) -> pd.Timestamp | None:
    return sessions[index - 1] if index > 0 else None


def _prepare_actions(frame: pd.DataFrame | None) -> dict[pd.Timestamp, list[CorporateAction]]:
    if frame is None or frame.empty:
        return {}
    required = {"ts_code", "ex_date", "split_ratio", "cash_dividend"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"corporate actions missing columns: {sorted(missing)}")
    out: dict[pd.Timestamp, list[CorporateAction]] = {}
    for row in frame.to_dict("records"):
        action = CorporateAction(str(row["ts_code"]), str(pd.Timestamp(row["ex_date"]).date()),
                                 float(row["split_ratio"]), float(row["cash_dividend"]),
                                 str(row["available_time"]) if row.get("available_time") is not None else None)
        day = pd.Timestamp(action.ex_date, tz="UTC")
        if action.available_time is not None and pd.Timestamp(action.available_time, tz="UTC") > day + pd.Timedelta(hours=16):
            raise ValueError("corporate action is not PIT-visible on its ex-date")
        out.setdefault(day, []).append(action)
    return out
