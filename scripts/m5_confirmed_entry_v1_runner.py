"""Rolling backtest for the preregistered M5 confirmed-entry challenger.

The selection ledger is frozen at T close.  A candidate is only executable on
T+1 after six observable 5-minute bars confirm the move; the 10:05 bar open is
the fill.  This runner is deliberately separate from the daily-open V3 engine
so the intraday timing cannot be silently approximated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from aistock9988.configuration import StrategyConfig
from aistock9988.data.corporate_actions_source import load_corporate_actions
from aistock9988.data.execution_source import load_execution_panel
from aistock9988.data.minute_source import load_minute_execution_panel
from aistock9988.execution.intraday import find_stop_execution
from aistock9988.reporting.v3_metrics import summarize_v3
from aistock9988.time.session import session_open

ROOT = Path(__file__).resolve().parents[1]
SHANGHAI = ZoneInfo("Asia/Shanghai")
CONFIRM_TIMES = ("09:35", "09:40", "09:45", "09:50", "09:55", "10:00")
FILL_TIME = "10:05"


def _day(value: object) -> pd.Timestamp:
    return pd.Timestamp(value).tz_convert("UTC").normalize() if pd.Timestamp(value).tzinfo else pd.Timestamp(value, tz="UTC").normalize()


def _local_hhmm(value: object) -> str:
    return pd.Timestamp(value).tz_convert(SHANGHAI).strftime("%H:%M")


def _scenario(strategy: StrategyConfig, name: str) -> dict[str, float]:
    row = strategy.execution["cost_scenarios"][name]
    return {key: float(row[key]) for key in ("slippage_each_side", "buy_commission", "sell_commission", "stamp_duty")}


def _load_candidates(path: Path, start: pd.Timestamp, signal_end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = pd.read_parquet(path)
    candidates["asof"] = pd.to_datetime(candidates["asof"], utc=True).dt.normalize()
    candidates = candidates[candidates["candidate_status"].eq("IN_VIEW")]
    candidates = candidates[candidates["asof"].between(start, signal_end)].copy()
    if candidates.empty:
        raise ValueError("candidate ledger has no IN_VIEW rows in requested range")
    selection_path = path.with_name("selection_ledger.parquet")
    selection = pd.read_parquet(selection_path)
    selection["asof"] = pd.to_datetime(selection["asof"], utc=True).dt.normalize()
    selection = selection[selection["asof"].between(start, signal_end)].copy()
    return candidates.sort_values(["asof", "candidate_rank", "ts_code"], kind="mergesort"), selection


def _confirm_bars(minute_groups: dict[tuple[pd.Timestamp, str], pd.DataFrame], day: pd.Timestamp,
                  code: str, signal_close: float, session_open_price: float) -> tuple[bool, str, pd.Series | None]:
    bars = minute_groups.get((day, code), pd.DataFrame()).copy()
    if bars.empty:
        return False, "missing_minute_bars", None
    bars["hhmm"] = bars["trade_time"].map(_local_hhmm)
    by_time = {row.hhmm: row for row in bars.itertuples(index=False)}
    missing = [item for item in CONFIRM_TIMES if item not in by_time]
    if missing:
        return False, "missing_confirmation_bar:" + ",".join(missing), None
    cutoff = pd.Timestamp(day).tz_convert(SHANGHAI).replace(hour=10, minute=0).tz_convert("UTC")
    if any(pd.Timestamp(by_time[item].available_time) > cutoff for item in CONFIRM_TIMES):
        return False, "confirmation_bar_not_available_by_1000", None
    close_bar = by_time["10:00"]
    if not (float(close_bar.economic_close) > float(session_open_price)):
        return False, "confirmation_close_not_above_open", None
    if float(close_bar.economic_close) < float(signal_close):
        return False, "confirmation_close_below_signal_close", None
    fill = by_time.get(FILL_TIME)
    if fill is None:
        return False, "missing_1005_fill_bar", None
    if bool(fill.is_locked_limit_up) or bool(fill.is_limit_up):
        return False, "fill_bar_limit_up", None
    return True, "confirmed", pd.Series(fill._asdict())


def _run_scenario(candidates: pd.DataFrame, selection: pd.DataFrame, daily: pd.DataFrame,
                  minutes: pd.DataFrame, actions: pd.DataFrame, strategy: StrategyConfig,
                  scenario_name: str) -> dict[str, pd.DataFrame]:
    costs = _scenario(strategy, scenario_name)
    stop_loss_pct = float(strategy.execution["stop"]["threshold_pct"])
    sessions = pd.DatetimeIndex(sorted(daily["trade_date"].drop_duplicates()))
    by_key = daily.set_index(["trade_date", "ts_code"])
    candidate_map = {day: group for day, group in candidates.groupby("asof", sort=True)}
    selection_map = {row.asof: row for row in selection.itertuples(index=False)}
    action_map = {day: group.to_dict("records") for day, group in actions.groupby("ex_date", sort=True)} if not actions.empty else {}
    minute_groups = {
        (day, str(code)): group
        for (day, code), group in minutes.groupby(["trade_date", "ts_code"], sort=False)
    }
    cash = float(strategy.execution["initial_cash"])
    initial_cash = cash
    positions: dict[str, dict[str, Any]] = {}
    orders: list[dict[str, Any]] = []
    fills: list[dict[str, Any]] = []
    nav_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    order_no = 0

    def lookup(day: pd.Timestamp, code: str) -> pd.Series | None:
        try:
            row = by_key.loc[(day, code)]
        except KeyError:
            return None
        return row if isinstance(row, pd.Series) else None

    def sell(code: str, day: pd.Timestamp, raw_price: float, economic_price: float, reason: str, trigger: str) -> None:
        nonlocal cash
        pos = positions[code]
        shares = float(pos["shares"])
        gross = raw_price * shares
        commission = gross * costs["sell_commission"]
        duty = gross * costs["stamp_duty"]
        cash += gross - commission - duty
        pnl = gross - commission - duty + pos["dividends"] - pos["cost"]
        fills.append({"order_id": pos["order_id"], "decision_id": pos["decision_id"], "trade_date": day,
                      "ts_code": code, "side": "SELL", "price": raw_price, "shares": shares,
                      "gross_value": gross, "commission": commission, "stamp_duty": duty,
                      "cash_after": cash, "reason": reason, "trigger_type": trigger,
                      "economic_price": economic_price, "economic_return": economic_price / pos["entry_economic"] - 1.0,
                      "realized_pnl": pnl, "gap_return": economic_price / pos["last_economic_close"] - 1.0,
                      "gap_flag": True})
        orders.append({"order_id": pos["order_id"], "decision_id": pos["decision_id"], "decision_session": pos["signal_session"],
                       "ts_code": code, "side": "SELL", "status": "FILLED", "execution_session": day,
                       "requested_shares": shares, "filled_shares": shares, "reason": reason})
        del positions[code]

    for i, day in enumerate(sessions):
        # Apply only actions that are explicitly PIT-approved by the loader.
        for action in action_map.get(day, []):
            code = str(action["ts_code"])
            if code in positions:
                pos = positions[code]
                dividend = float(pos["shares"]) * float(action["cash_dividend"])
                cash += dividend
                pos["dividends"] += dividend
                pos["shares"] *= float(action["split_ratio"])

        # Scheduled exits happen at the next tradable daily open.
        for code, pos in list(positions.items()):
            if pos.get("exit_due") != day and not pos.get("pending_exit"):
                continue
            row = lookup(day, code)
            if row is None or bool(row.is_suspended) or bool(row.is_limit_down):
                pos["pending_exit"] = True
                continue
            raw = float(row.raw_open) * (1.0 - costs["slippage_each_side"])
            eco = float(row.economic_open) * (1.0 - costs["slippage_each_side"])
            sell(code, day, raw, eco, pos["exit_reason"], pos["exit_reason"])

        # T+1 confirmation and 10:05 entry.  Missing bars are a skipped trade,
        # never a look-ahead fallback to a later session.
        previous = sessions[i - 1] if i else None
        if previous in candidate_map and previous in selection_map:
            decision = selection_map[previous]
            slots = max(0, int(strategy.portfolio["max_open_positions"]) - len(positions))
            desired = min(int(strategy.portfolio["entries_per_decision"]), slots)
            decision_nav = float(nav_rows[-1]["nav"]) if nav_rows else initial_cash
            target_weight = float(strategy.portfolio["sizing"]["value"])
            exposure = sum(float(p["shares"]) * float(p["last_raw_close"]) for p in positions.values())
            budget_left = max(0.0, float(strategy.portfolio["target_gross_exposure_cap"]) * decision_nav - exposure)
            for candidate in candidate_map[previous].itertuples(index=False):
                if desired <= 0:
                    break
                code = str(candidate.ts_code)
                if code in positions:
                    continue
                row = lookup(day, code)
                if row is None or bool(row.is_suspended):
                    decisions.append({"signal_session": previous, "execution_session": day, "ts_code": code,
                                      "candidate_rank": int(candidate.candidate_rank), "status": "SKIP_MISSING_DAILY"})
                    continue
                previous_row = lookup(previous, code)
                if previous_row is None:
                    decisions.append({"signal_session": previous, "execution_session": day, "ts_code": code,
                                      "candidate_rank": int(candidate.candidate_rank), "status": "SKIP_MISSING_SIGNAL_CLOSE"})
                    continue
                ok, reason, fill_bar = _confirm_bars(
                    minute_groups, day, code, float(previous_row.economic_close), float(row.economic_open)
                )
                if not ok:
                    decisions.append({"signal_session": previous, "execution_session": day, "ts_code": code,
                                      "candidate_rank": int(candidate.candidate_rank), "status": reason})
                    continue
                raw = float(fill_bar["open"]) * (1.0 + costs["slippage_each_side"])
                eco = float(fill_bar["economic_open"]) * (1.0 + costs["slippage_each_side"])
                unit = raw * (1.0 + costs["buy_commission"])
                budget = min(cash, budget_left, decision_nav * target_weight)
                shares = int((budget / unit) // int(strategy.execution["lot_size"])) * int(strategy.execution["lot_size"])
                adv_row = lookup(previous, code)
                adv20 = float(adv_row.adv20_amount) if adv_row is not None and pd.notna(adv_row.adv20_amount) else np.nan
                if np.isfinite(adv20):
                    cap_value = adv20 * float(strategy.execution["amount_unit_multiplier"]) * float(strategy.execution["adv20_max_participation"])
                    shares = min(shares, int((cap_value / unit) // int(strategy.execution["lot_size"])) * int(strategy.execution["lot_size"]))
                if shares <= 0:
                    decisions.append({"signal_session": previous, "execution_session": day, "ts_code": code,
                                      "candidate_rank": int(candidate.candidate_rank), "status": "SKIP_BUDGET"})
                    continue
                order_no += 1
                order_id = f"m5-{scenario_name}-{order_no:08d}"
                gross = raw * shares
                commission = gross * costs["buy_commission"]
                cash -= gross + commission
                fills.append({"order_id": order_id, "decision_id": str(decision.decision_id), "trade_date": day,
                              "ts_code": code, "side": "BUY", "price": raw, "shares": shares,
                              "gross_value": gross, "commission": commission, "stamp_duty": 0.0,
                              "cash_after": cash, "reason": "M5_CONFIRMED_ENTRY", "trigger_type": "M5_CONFIRM",
                              "economic_price": eco, "economic_return": np.nan, "realized_pnl": np.nan,
                              "gap_return": np.nan, "gap_flag": False})
                orders.append({"order_id": order_id, "decision_id": str(decision.decision_id), "decision_session": previous,
                               "ts_code": code, "side": "BUY", "status": "FILLED", "execution_session": day,
                               "requested_shares": shares, "filled_shares": shares, "reason": "M5_CONFIRMED_ENTRY"})
                positions[code] = {"shares": shares, "entry_date": day, "entry_index": i,
                                   "entry_price": raw, "entry_economic": eco, "cost": gross + commission,
                                   "dividends": 0.0, "decision_id": str(decision.decision_id), "order_id": order_id,
                                   "signal_session": previous,
                                   "last_raw_close": float(row.raw_close), "last_economic_close": float(row.economic_close),
                                   "exit_due": None, "exit_reason": None, "pending_exit": False}
                budget_left -= gross + commission
                desired -= 1
                decisions.append({"signal_session": previous, "execution_session": day, "ts_code": code,
                                  "candidate_rank": int(candidate.candidate_rank), "status": "FILLED_M5_1005"})

        # Intraday stop-loss is available only from the session after entry.
        for code, pos in list(positions.items()):
            if pos["entry_date"] >= day or code not in positions:
                continue
            bars = minute_groups.get((day, code), pd.DataFrame())
            result = find_stop_execution(bars, entry_economic_price=float(pos["entry_economic"]),
                                         stop_loss_pct=stop_loss_pct, start_time=session_open(day)) if not bars.empty else None
            if result is not None and result.status == "FILLED":
                fill = bars[bars["trade_time"] == result.execution_time].iloc[0]
                sell(code, day, float(result.execution_raw_price) * (1.0 - costs["slippage_each_side"]),
                     float(result.execution_raw_price) * float(fill.adj_factor) * (1.0 - costs["slippage_each_side"]),
                     "INTRADAY_STOP_LOSS", "STOP_LOSS")
            elif result is not None and result.status == "PENDING":
                decisions.append({"signal_session": pos["signal_session"], "execution_session": day,
                                  "ts_code": code, "candidate_rank": np.nan,
                                  "status": "STOP_PENDING_" + result.reason})

        # Update marks and set the H5 path-dependent exit.
        for code, pos in list(positions.items()):
            row = lookup(day, code)
            if row is not None:
                pos["last_raw_close"] = float(row.raw_close)
                pos["last_economic_close"] = float(row.economic_close)
            age = i - int(pos["entry_index"])
            if age == 5 and pos.get("exit_due") is None:
                if pos["last_economic_close"] / pos["entry_economic"] - 1.0 <= 0:
                    pos["exit_due"] = sessions[i + 1] if i + 1 < len(sessions) else None
                    pos["exit_reason"] = "H5_NON_POSITIVE"
                else:
                    pos["exit_due"] = sessions[int(pos["entry_index"]) + 10] if int(pos["entry_index"]) + 10 < len(sessions) else None
                    pos["exit_reason"] = "H10"
        market_value = sum(float(p["shares"]) * float(p["last_raw_close"]) for p in positions.values())
        nav_rows.append({"trade_date": day, "cash": cash, "market_value": market_value,
                         "nav": cash + market_value, "open_positions": len(positions),
                         "gross_exposure": market_value / (cash + market_value) if cash + market_value > 0 else np.nan})

    return {"fills": pd.DataFrame(fills), "orders": pd.DataFrame(orders), "nav": pd.DataFrame(nav_rows),
            "positions": pd.DataFrame([{"ts_code": code, **pos} for code, pos in positions.items()]),
            "execution_decisions": pd.DataFrame(decisions)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    strategy = StrategyConfig.from_yaml(args.strategy)
    start, signal_end, execution_end = map(lambda x: pd.Timestamp(x, tz="UTC").normalize(),
                                            (args.start, args.signal_end, args.execution_end))
    candidates, selection = _load_candidates(args.candidate_ledger, start, signal_end)
    codes = sorted(candidates["ts_code"].astype(str).unique())
    daily = load_execution_panel(str(start.date()), str(execution_end.date()), ts_codes=codes)
    daily["trade_date"] = pd.to_datetime(daily["trade_date"], utc=True).dt.normalize()
    daily = daily.sort_values(["trade_date", "ts_code"], kind="mergesort")
    daily["amount"] = pd.to_numeric(daily["amount"], errors="coerce")
    daily["adv20_amount"] = daily.groupby("ts_code", sort=False)["amount"].transform(
        lambda values: values.shift(1).rolling(20, min_periods=20).median()
    )
    minutes = load_minute_execution_panel(str(start.date()), str(execution_end.date()), freq="5min", ts_codes=codes)
    actions = load_corporate_actions(str(start.date()), str(execution_end.date()), ts_codes=codes)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"immutable output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(output / "candidate_view.parquet", index=False)
    selection.to_parquet(output / "selection_ledger.parquet", index=False)
    daily.to_parquet(output / "execution_daily.parquet", index=False)
    minutes.to_parquet(output / "execution_5min.parquet", index=False)
    actions.to_parquet(output / "corporate_actions.parquet", index=False)
    portfolios = {}
    for name in ("base", "stress"):
        result = _run_scenario(candidates, selection, daily, minutes, actions, strategy, name)
        target = output / name
        target.mkdir()
        for key, frame in result.items():
            frame.to_parquet(target / f"{key}.parquet", index=False)
        metrics = summarize_v3(result["nav"], result["fills"], initial_cash=float(strategy.execution["initial_cash"]))
        decision_rows = result["execution_decisions"]
        metrics.update({"scenario": name, "signal_rows": int(len(candidates)), "codes": len(codes),
                        "start": str(start.date()), "signal_end": str(signal_end.date()),
                        "execution_end": str(execution_end.date()), "entry_contract": "M5_09:35-10:00_then_10:05_open",
                        "h5_contract": "non_positive_next_open_else_h10", "stop_loss_pct": float(strategy.execution["stop"]["threshold_pct"]),
                        "entry_decision_status_counts": ({str(k): int(v) for k, v in decision_rows["status"].value_counts().items()}
                                                          if not decision_rows.empty else {}),
                        "acceptance": {
                            "pf_ge_2": bool(metrics["portfolio_profit_factor"] is not None and metrics["portfolio_profit_factor"] >= 2.0),
                            "maxdd_le_15pct": bool(metrics["max_drawdown"] >= -0.15),
                            "excluding_best_week_positive": bool(metrics["return_excluding_best_week"] > 0),
                        }})
        (target / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8")
        portfolios[name] = metrics
    summary = {"strategy": strategy.strategy_id, "portfolios": portfolios,
               "minute_trade_date_max": str(minutes["trade_date"].max().date()),
               "daily_trade_date_max": str(daily["trade_date"].max().date())}
    (output / "SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-ledger", type=Path, required=True)
    parser.add_argument("--strategy", type=Path, default=ROOT / "configs/strategy/m5_confirmed_entry_v1.yaml")
    parser.add_argument("--start", required=True)
    parser.add_argument("--signal-end", required=True)
    parser.add_argument("--execution-end", required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
