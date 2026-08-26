"""Small, deterministic event-driven backtest engine.

The engine consumes frozen selection signals and a raw execution-price panel.
It deliberately keeps research returns and accounting prices separate: only
``raw_open``/``raw_close`` are used for orders and NAV.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..execution.prices import validate_execution_panel


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 1_000_000.0
    max_positions: int = 2
    hold_sessions: int = 5
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    buy_commission: float = 0.0003
    sell_commission: float = 0.0003
    stamp_duty: float = 0.0005
    lot_size: int = 1
    buy_slippage: float = 0.0
    sell_slippage: float = 0.0


def run_backtest(signals: pd.DataFrame, prices: pd.DataFrame, *, config: BacktestConfig = BacktestConfig()) -> dict[str, pd.DataFrame]:
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
    nav_rows: list[dict] = []
    signal_map = {d: g.sort_values(["candidate_rank", "ts_code"], kind="mergesort")
                  for d, g in sig.groupby("asof", sort=True)}

    def execute(day, code, side, price, shares, reason):
        nonlocal cash
        gross = price * shares
        commission = gross * (config.buy_commission if side == "BUY" else config.sell_commission)
        duty = gross * config.stamp_duty if side == "SELL" else 0.0
        cash_change = -(gross + commission) if side == "BUY" else gross - commission - duty
        cash += cash_change
        trades.append({"trade_date": day, "ts_code": code, "side": side, "price": price,
                       "shares": shares, "gross_value": gross, "commission": commission,
                       "stamp_duty": duty, "cash_after": cash, "reason": reason})
        return gross + commission if side == "BUY" else gross - commission - duty

    for i, day in enumerate(sessions):
        # Decisions made at yesterday's close become executable at today's open.
        for code, pos in list(positions.items()):
            prior_close = pos["last_close"]
            economic_ret = pos["last_economic_close"] / pos["entry_economic_price"] - 1.0
            should_exit = pos.get("exit_pending", False) or (pos["exit_due"] == day) or (config.stop_loss_pct is not None and economic_ret <= config.stop_loss_pct) or (config.take_profit_pct is not None and economic_ret >= config.take_profit_pct)
            if should_exit:
                row = by_key.loc[(day, code)] if (day, code) in by_key.index else None
                if row is not None and not bool(row.is_suspended) and not bool(row.is_limit_down):
                    execute(day, code, "SELL", float(row.raw_open) * (1.0 - config.sell_slippage), pos["shares"], "scheduled_or_rule_exit")
                    del positions[code]
                elif row is not None:
                    pos["exit_pending"] = True

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
                    continue
                price = float(row.raw_open) * (1.0 + config.buy_slippage)
                shares = int((cash / max(1, config.max_positions) / price) // config.lot_size) * config.lot_size
                if shares <= 0:
                    continue
                execute(day, code, "BUY", price, shares, "signal_entry")
                exit_idx = i + config.hold_sessions
                positions[code] = {"shares": shares, "entry_price": price, "entry_economic_price": float(row.economic_open), "entry_date": day,
                                   "exit_due": sessions[exit_idx] if exit_idx < len(sessions) else None,
                                   "last_close": float(row.raw_close), "last_economic_close": float(row.economic_close)}

        for code, pos in positions.items():
            if (day, code) in by_key.index:
                pos["last_close"] = float(by_key.loc[(day, code)].raw_close)
                pos["last_economic_close"] = float(by_key.loc[(day, code)].economic_close)
        mark = 0.0
        for code, pos in positions.items():
            if (day, code) in by_key.index:
                mark += pos["shares"] * float(by_key.loc[(day, code)].raw_close)
        nav_rows.append({"trade_date": day, "cash": cash, "market_value": mark, "nav": cash + mark,
                         "open_positions": len(positions)})

    # Final liquidation is explicitly marked and uses the last available close.
    final_day = sessions[-1]
    for code, pos in list(positions.items()):
        if (final_day, code) in by_key.index:
            row = by_key.loc[(final_day, code)]
            if not bool(row.is_suspended):
                execute(final_day, code, "SELL", float(row.raw_close) * (1.0 - config.sell_slippage), pos["shares"], "end_of_test_liquidation")
        del positions[code]
    if nav_rows and trades:
        nav_rows[-1]["cash"] = cash
        nav_rows[-1]["market_value"] = 0.0
        nav_rows[-1]["nav"] = cash
        nav_rows[-1]["open_positions"] = 0
    return {"trades": pd.DataFrame(trades), "nav": pd.DataFrame(nav_rows),
            "positions": pd.DataFrame(columns=["trade_date", "ts_code", "shares"])}


def _dates(values: pd.Series) -> list[pd.Timestamp]:
    out = pd.to_datetime(values, utc=True).dt.normalize()
    return out.tolist()


def _validate_config(config: BacktestConfig) -> None:
    if config.initial_cash <= 0 or config.max_positions <= 0 or config.hold_sessions <= 0 or config.lot_size <= 0:
        raise ValueError("cash, positions, hold_sessions and lot_size must be positive")
    if config.buy_slippage < 0 or config.sell_slippage < 0:
        raise ValueError("slippage must be non-negative")
    if config.stop_loss_pct is not None and config.stop_loss_pct >= 0:
        raise ValueError("stop_loss_pct must be negative")
    if config.take_profit_pct is not None and config.take_profit_pct <= 0:
        raise ValueError("take_profit_pct must be positive")
