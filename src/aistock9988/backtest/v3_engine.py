"""Deterministic V3 long-only event engine for frozen rule selections."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..configuration import StrategyConfig


class BacktestDataError(RuntimeError):
    pass


@dataclass(frozen=True)
class CostScenario:
    name: str
    slippage_each_side: float
    buy_commission: float
    sell_commission: float
    stamp_duty: float


def run_v3_backtest(
    *,
    candidate_ledger: pd.DataFrame,
    selection_ledger: pd.DataFrame,
    execution_panel: pd.DataFrame,
    corporate_actions: pd.DataFrame,
    strategy: StrategyConfig,
    execution_sessions: tuple[str, ...],
    scenario_name: str,
) -> dict[str, pd.DataFrame]:
    scenario = _scenario(strategy, scenario_name)
    sessions = pd.DatetimeIndex(pd.to_datetime(execution_sessions, utc=True)).normalize()
    execution_source = execution_panel.copy()
    rex_exit_mode = str(strategy.execution.get("exit_mode", "fixed_hold"))
    rex_enabled = rex_exit_mode == "rex_conditional_extension_v1"
    rex_confirmation_sessions = 10
    rex_max_sessions = int(strategy.execution["hold_sessions_from_fill"])
    if rex_enabled:
        extension = strategy.execution.get("extension", {})
        rex_confirmation_sessions = int(extension.get("confirmation_sessions", 10))
        rex_max_sessions = int(extension.get("max_hold_sessions", 20))
        if rex_confirmation_sessions <= 0 or rex_max_sessions <= rex_confirmation_sessions:
            raise ValueError(
                "REX extension requires 0 < confirmation_sessions < max_hold_sessions"
            )
        # Calculate MA5 before clipping to execution sessions so the H10
        # predicate has the same warm-up history as the feature engine.
        execution_source = _prepare_rex_execution(execution_source, strategy)
    execution = execution_source[execution_source["trade_date"].isin(sessions)].copy()
    if execution.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("execution panel contains duplicate security/session keys")
    by_key = execution.set_index(["trade_date", "ts_code"])
    candidate_view = candidate_ledger[candidate_ledger["candidate_status"].eq("IN_VIEW")].copy()
    candidate_map = {
        day: group.sort_values(["candidate_rank", "ts_code"], kind="mergesort")
        for day, group in candidate_view.groupby("asof", sort=True)
    }
    selection_map = {row.asof: row for row in selection_ledger.itertuples(index=False)}
    actions = _action_map(corporate_actions)

    initial_cash = float(strategy.execution["initial_cash"])
    cash = initial_cash
    positions: dict[str, dict[str, Any]] = {}
    orders: list[dict[str, Any]] = []
    fills: list[dict[str, Any]] = []
    position_events: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    nav_rows: list[dict[str, Any]] = []
    execution_decisions: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    order_sequence = 0
    event_sequence = 0

    def event(day: pd.Timestamp, code: str, event_type: str, state_before: str, state_after: str, reason: str) -> None:
        nonlocal event_sequence
        event_sequence += 1
        position_events.append({
            "event_id": f"evt-{scenario.name}-{event_sequence:08d}", "trade_date": day,
            "ts_code": code, "event_type": event_type, "state_before": state_before,
            "state_after": state_after, "reason": reason,
        })

    def new_order(day: pd.Timestamp, decision_id: str, code: str, side: str, requested: float, reason: str) -> dict[str, Any]:
        nonlocal order_sequence
        order_sequence += 1
        row = {
            "order_id": f"v3-{scenario.name}-{order_sequence:08d}", "decision_id": decision_id,
            "decision_session": day, "ts_code": code, "side": side,
            "requested_shares": float(requested), "status": "CREATED", "reason": reason,
            "execution_session": pd.NaT, "execution_price": np.nan, "filled_shares": 0.0,
        }
        orders.append(row)
        return row

    def reject(order: dict[str, Any], day: pd.Timestamp, reason: str) -> None:
        order.update(status="REJECTED", execution_session=day, reason=reason)

    def fill_buy(order: dict[str, Any], day: pd.Timestamp, code: str, shares: float, row: pd.Series) -> None:
        nonlocal cash
        raw_price = float(row.raw_open) * (1.0 + scenario.slippage_each_side)
        economic_price = float(row.economic_open) * (1.0 + scenario.slippage_each_side)
        gross = raw_price * shares
        commission = gross * scenario.buy_commission
        total_cost = gross + commission
        if total_cost > cash + 1e-8:
            raise AssertionError("buy fill exceeds available cash")
        cash -= total_cost
        order.update(status="FILLED", execution_session=day, execution_price=raw_price, filled_shares=shares)
        fills.append({
            "order_id": order["order_id"], "decision_id": order["decision_id"], "trade_date": day,
            "ts_code": code, "side": "BUY", "price": raw_price, "shares": shares,
            "entry_date": day,
            "gross_value": gross, "commission": commission, "stamp_duty": 0.0,
            "cash_after": cash, "reason": "SIGNAL_ENTRY", "trigger_type": None,
            "economic_price": economic_price, "economic_return": np.nan,
            "realized_pnl": np.nan, "gap_return": np.nan, "gap_flag": False,
        })
        entry_index = sessions.get_indexer([day])[0]
        positions[code] = {
            "state": "ACTIVE", "shares": shares, "entry_date": day, "entry_price": raw_price,
            "entry_economic_price": economic_price, "total_cost": total_cost,
            # REX observes H10/H20 at the close.  Its normal scheduled exit is
            # therefore disabled until the close-triggered state machine sets
            # an EXIT_PENDING order; the fixed-hold path is unchanged.
            "dividends_received": 0.0,
            "scheduled_exit_index": (
                entry_index + rex_max_sessions + 1
                if rex_enabled
                else entry_index + int(strategy.execution["hold_sessions_from_fill"])
            ),
            "exit_reason": None, "last_raw_close": float(row.raw_close),
            "last_economic_close": float(row.economic_close), "decision_id": order["decision_id"],
            "rex_enabled": rex_enabled,
            # H10/H20 refer to the 10th/20th held bar (entry itself is bar 1).
            # Close-triggered exits execute on the following session's open.
            "rex_h10_index": entry_index + rex_confirmation_sessions - 1,
            "rex_max_index": entry_index + rex_max_sessions - 1,
            "rex_extended": False,
            "rex_h10_economic_close": np.nan,
            "rex_post_h10_peak": np.nan,
            "rex_exit_triggers": [],
        }
        event(day, code, "ENTRY_FILL", "ENTRY_ATTEMPTED", "ACTIVE", "SIGNAL_ENTRY")

    def fill_sell(order: dict[str, Any], day: pd.Timestamp, code: str, row: pd.Series) -> None:
        nonlocal cash
        pos = positions[code]
        raw_price = float(row.raw_open) * (1.0 - scenario.slippage_each_side)
        economic_price = float(row.economic_open) * (1.0 - scenario.slippage_each_side)
        shares = float(pos["shares"])
        gross = raw_price * shares
        commission = gross * scenario.sell_commission
        duty = gross * scenario.stamp_duty
        proceeds = gross - commission - duty
        cash += proceeds
        realized = proceeds + float(pos["dividends_received"]) - float(pos["total_cost"])
        order.update(status="FILLED", execution_session=day, execution_price=raw_price, filled_shares=shares)
        fills.append({
            "order_id": order["order_id"], "decision_id": order["decision_id"], "trade_date": day,
            "ts_code": code, "side": "SELL", "price": raw_price, "shares": shares,
            "entry_date": pos["entry_date"],
            "gross_value": gross, "commission": commission, "stamp_duty": duty,
            "cash_after": cash, "reason": str(pos["exit_reason"]), "trigger_type": str(pos["exit_reason"]),
            "economic_price": economic_price,
            "economic_return": economic_price / float(pos["entry_economic_price"]) - 1.0,
            "realized_pnl": realized,
            "gap_return": economic_price / float(pos["last_economic_close"]) - 1.0,
            "gap_flag": True,
            "exit_triggers": list(pos.get("rex_exit_triggers", [])),
        })
        event(day, code, "EXIT_FILL", "EXIT_PENDING", "CLOSED", str(pos["exit_reason"]))
        del positions[code]

    for session_index, day in enumerate(sessions):
        for action in actions.get(day, []):
            code = str(action["ts_code"])
            if code not in positions:
                continue
            pos = positions[code]
            shares_before = float(pos["shares"])
            dividend = shares_before * float(action["cash_dividend"])
            cash += dividend
            pos["dividends_received"] += dividend
            pos["shares"] = shares_before * float(action["split_ratio"])
            pos["entry_price"] = float(pos["entry_price"]) / float(action["split_ratio"])
            action_rows.append({
                "trade_date": day, "ts_code": code, "shares_before": shares_before,
                "shares_after": pos["shares"], "split_ratio": action["split_ratio"],
                "cash_dividend": dividend, "cash_after": cash,
                "decision_id": pos["decision_id"], "entry_date": pos["entry_date"],
            })
            event(day, code, "CORPORATE_ACTION", pos["state"], pos["state"], str(action.get("action_type", "ACTION")))

        for code in list(positions):
            pos = positions[code]
            if (
                pos["state"] == "ACTIVE"
                and not rex_enabled
                and session_index >= int(pos["scheduled_exit_index"])
            ):
                pos["state"] = "EXIT_PENDING"
                pos["exit_reason"] = "TIME_EXIT"
                event(day, code, "EXIT_TRIGGER", "ACTIVE", "EXIT_PENDING", "TIME_EXIT")
            if pos["state"] != "EXIT_PENDING":
                continue
            decision_id = str(pos["decision_id"])
            order = new_order(day, decision_id, code, "SELL", float(pos["shares"]), str(pos["exit_reason"]))
            row = _execution_row(by_key, day, code)
            status = str(row.execution_status)
            if status == "MISSING_REQUIRED_DATA":
                reject(order, day, "SELL_MISSING_REQUIRED_DATA")
                event(day, code, "EXIT_RETRY", "EXIT_PENDING", "EXIT_PENDING", status)
                continue
            if status in {"SUSPENDED", "LIMIT_DOWN", "ZERO_VOLUME", "OUT_OF_UNIVERSE"}:
                reject(order, day, f"SELL_{status}")
                event(day, code, "EXIT_RETRY", "EXIT_PENDING", "EXIT_PENDING", status)
                continue
            fill_sell(order, day, code, row)

        previous = sessions[session_index - 1] if session_index else None
        if previous in selection_map and previous in candidate_map:
            decision = selection_map[previous]
            candidates = candidate_map[previous]
            slots = max(0, int(strategy.portfolio["max_open_positions"]) - len(positions))
            desired = min(int(decision.desired_entries), slots)
            filled = 0
            current_open_value = _open_market_value(positions, by_key, day)
            decision_nav = float(nav_rows[-1]["nav"]) if nav_rows else initial_cash
            exposure_budget = max(0.0, float(strategy.portfolio["target_gross_exposure_cap"]) * decision_nav - current_open_value)
            attempted: set[str] = set()
            for attempt_no, candidate in enumerate(candidates.itertuples(index=False), start=1):
                if filled >= desired:
                    break
                code = str(candidate.ts_code)
                if code in attempted:
                    continue
                attempted.add(code)
                row = _execution_row(by_key, day, code)
                status = str(row.execution_status)
                chosen = False
                reason = ""
                if code in positions:
                    reason = "DUPLICATE_POSITION"
                elif status == "MISSING_REQUIRED_DATA":
                    reason = "BUY_MISSING_REQUIRED_DATA"
                elif status in {"SUSPENDED", "LIMIT_UP", "ZERO_VOLUME", "OUT_OF_UNIVERSE"}:
                    reason = f"BUY_{status}"
                else:
                    price = float(row.raw_open) * (1.0 + scenario.slippage_each_side)
                    target_budget = float(decision.target_weight_each) * decision_nav
                    budget = min(target_budget, exposure_budget, cash)
                    unit_cost = price * (1.0 + scenario.buy_commission)
                    lot = int(strategy.execution["lot_size"])
                    shares = int((budget / unit_cost) // lot) * lot
                    adv_row = _execution_row(by_key, previous, code)
                    adv_value = float(adv_row.adv20_amount) * float(strategy.execution["amount_unit_multiplier"])
                    if np.isfinite(adv_value):
                        cap_shares = int(
                            (adv_value * float(strategy.execution["adv20_max_participation"]) / unit_cost) // lot
                        ) * lot
                        shares = min(shares, max(0, cap_shares))
                    if shares <= 0:
                        reason = "INSUFFICIENT_BUDGET_OR_CAPACITY"
                    else:
                        order = new_order(previous, str(decision.decision_id), code, "BUY", shares, "SIGNAL_ENTRY")
                        fill_buy(order, day, code, shares, row)
                        spent = float(fills[-1]["gross_value"] + fills[-1]["commission"])
                        exposure_budget = max(0.0, exposure_budget - spent)
                        filled += 1
                        chosen = True
                        reason = "FILLED"
                execution_decisions.append({
                    "decision_id": str(decision.decision_id), "signal_session": previous,
                    "execution_session": day, "attempt_no": attempt_no, "candidate_rank": int(candidate.candidate_rank),
                    "ts_code": code, "execution_status": status, "chosen": chosen,
                    "reject_reason": reason, "candidate_snapshot_id": str(candidate.candidate_snapshot_id),
                })

        for code, pos in list(positions.items()):
            row = _execution_row(by_key, day, code)
            status = str(row.execution_status)
            data_eligible = bool(row.execution_data_eligible)
            if not data_eligible:
                event(day, code, "HELD_DATA_GAP", pos["state"], pos["state"], str(row.missing_required_execution))
                # H10 is a close-time decision.  If the close/MA5 inputs are
                # unavailable on that decision session, reject extension
                # explicitly and submit the normal H10 exit on the next open;
                # never carry the position silently to H20.
                if (
                    rex_enabled
                    and pos["state"] == "ACTIVE"
                    and session_index == int(pos["rex_h10_index"])
                ):
                    pos["state"] = "EXIT_PENDING"
                    pos["exit_reason"] = "REX_H10_EXTENSION_DATA_UNAVAILABLE"
                    pos["rex_exit_triggers"] = [pos["exit_reason"]]
                    event(
                        day,
                        code,
                        "EXTENSION_DATA_GAP",
                        "ACTIVE",
                        "EXIT_PENDING",
                        "REX_H10_MA5_UNAVAILABLE",
                    )
                    event(
                        day,
                        code,
                        "EXIT_TRIGGER",
                        "ACTIVE",
                        "EXIT_PENDING",
                        str(pos["exit_reason"]),
                    )
                # A hard H20 boundary is still enforceable when the close row
                # is unavailable: keep the order pending and retry at the
                # next tradable open rather than silently extending the hold.
                if (
                    rex_enabled
                    and pos["state"] == "ACTIVE"
                    and session_index >= int(pos["rex_max_index"])
                ):
                    pos["state"] = "EXIT_PENDING"
                    pos["exit_reason"] = "REX_MAX_H20"
                    pos["rex_exit_triggers"] = ["REX_MAX_H20"]
                    event(day, code, "EXIT_TRIGGER", "ACTIVE", "EXIT_PENDING", "REX_MAX_H20")
                continue
            if pd.notna(row.raw_close):
                pos["last_raw_close"] = float(row.raw_close)
            if pd.notna(row.economic_close):
                pos["last_economic_close"] = float(row.economic_close)
            if pos["state"] == "ACTIVE":
                stop_pct = float(strategy.execution["stop"]["threshold_pct"])
                if float(pos["last_economic_close"]) / float(pos["entry_economic_price"]) - 1.0 <= stop_pct:
                    pos["state"] = "EXIT_PENDING"
                    pos["exit_reason"] = "STOP_LOSS"
                    pos["rex_exit_triggers"] = ["STOP_LOSS"]
                    event(day, code, "EXIT_TRIGGER", "ACTIVE", "EXIT_PENDING", "STOP_LOSS")

            if rex_enabled and pos["state"] == "ACTIVE":
                ma5 = float(row.ma5) if "ma5" in row.index and pd.notna(row.ma5) else np.nan
                economic_close = float(pos["last_economic_close"])
                if session_index == int(pos["rex_h10_index"]):
                    pos["rex_h10_economic_close"] = economic_close
                    pos["rex_post_h10_peak"] = economic_close
                    extension_ok = (
                        economic_close > float(pos["entry_economic_price"])
                        and np.isfinite(ma5)
                        and economic_close > ma5
                    )
                    if extension_ok:
                        pos["rex_extended"] = True
                        event(
                            day,
                            code,
                            "EXTENSION_APPROVED",
                            "ACTIVE",
                            "ACTIVE",
                            "REX_H10_EXTENSION_APPROVED",
                        )
                    else:
                        unavailable = not np.isfinite(ma5)
                        if unavailable:
                            event(
                                day,
                                code,
                                "EXTENSION_DATA_GAP",
                                "ACTIVE",
                                "ACTIVE",
                                "REX_H10_MA5_UNAVAILABLE",
                            )
                        pos["state"] = "EXIT_PENDING"
                        pos["exit_reason"] = (
                            "REX_H10_EXTENSION_DATA_UNAVAILABLE"
                            if unavailable
                            else "REX_H10_EXTENSION_REJECTED"
                        )
                        pos["rex_exit_triggers"] = [pos["exit_reason"]]
                        event(
                            day,
                            code,
                            "EXIT_TRIGGER",
                            "ACTIVE",
                            "EXIT_PENDING",
                            str(pos["exit_reason"]),
                        )
                elif session_index > int(pos["rex_h10_index"]) and pos["rex_extended"]:
                    if not np.isfinite(ma5):
                        # The position cannot be evaluated against the
                        # registered trailing rule without MA5.  Record the
                        # gap and fail closed rather than silently disabling
                        # that exit predicate.
                        pos["state"] = "EXIT_PENDING"
                        pos["exit_reason"] = "REX_POST_H10_MA5_UNAVAILABLE"
                        pos["rex_exit_triggers"] = [pos["exit_reason"]]
                        event(
                            day,
                            code,
                            "EXTENSION_DATA_GAP",
                            "ACTIVE",
                            "EXIT_PENDING",
                            "REX_POST_H10_MA5_UNAVAILABLE",
                        )
                        continue
                    pos["rex_post_h10_peak"] = max(
                        float(pos["rex_post_h10_peak"]), economic_close
                    )
                    peak_drawdown = economic_close / float(pos["rex_post_h10_peak"]) - 1.0
                    extension = strategy.execution.get("extension", {})
                    drawdown_limit = float(extension.get("peak_drawdown_pct", -0.08))
                    reasons: list[str] = []
                    if np.isfinite(ma5) and economic_close < ma5:
                        reasons.append("REX_BELOW_MA5")
                    if economic_close < float(pos["rex_h10_economic_close"]):
                        reasons.append("REX_BELOW_H10_CLOSE")
                    if peak_drawdown <= drawdown_limit:
                        reasons.append("REX_PEAK_DRAWDOWN_8")
                    if reasons:
                        pos["state"] = "EXIT_PENDING"
                        # Keep a stable primary reason while preserving all
                        # same-close predicates in a structured field.
                        pos["exit_reason"] = reasons[0]
                        pos["rex_exit_triggers"] = reasons
                        event(day, code, "EXIT_TRIGGER", "ACTIVE", "EXIT_PENDING", pos["exit_reason"])

                if (
                    pos["state"] == "ACTIVE"
                    and session_index >= int(pos["rex_max_index"])
                ):
                    pos["state"] = "EXIT_PENDING"
                    pos["exit_reason"] = "REX_MAX_H20"
                    pos["rex_exit_triggers"] = ["REX_MAX_H20"]
                    event(day, code, "EXIT_TRIGGER", "ACTIVE", "EXIT_PENDING", "REX_MAX_H20")

        market_value = sum(float(pos["shares"]) * float(pos["last_raw_close"]) for pos in positions.values())
        nav = cash + market_value
        if cash < -1e-8 or abs(nav - cash - market_value) > 1e-8:
            raise AssertionError("NAV accounting identity failed")
        nav_rows.append({
            "trade_date": day, "cash": cash, "market_value": market_value, "nav": nav,
            "open_positions": len(positions), "gross_exposure": market_value / nav if nav > 0 else np.nan,
        })
        for code, pos in positions.items():
            position_rows.append({
                "trade_date": day, "ts_code": code, "state": pos["state"], "shares": pos["shares"],
                "entry_date": pos["entry_date"], "entry_price": pos["entry_price"],
                "decision_id": pos["decision_id"],
                "last_raw_close": pos["last_raw_close"], "market_value": float(pos["shares"]) * float(pos["last_raw_close"]),
                "unrealized_return": float(pos["last_economic_close"]) / float(pos["entry_economic_price"]) - 1.0,
                "exit_reason": pos["exit_reason"],
                "rex_extended": bool(pos.get("rex_extended", False)),
                "rex_h10_economic_close": pos.get("rex_h10_economic_close", np.nan),
                "rex_post_h10_peak": pos.get("rex_post_h10_peak", np.nan),
                "rex_exit_triggers": pos.get("rex_exit_triggers", []),
            })

    for code, pos in positions.items():
        event(sessions[-1], code, "END_MARK", pos["state"], "OPEN_MARK", "EXECUTION_END")

    return {
        "orders": pd.DataFrame(orders),
        "fills": pd.DataFrame(fills),
        "position_events": pd.DataFrame(position_events),
        "positions": pd.DataFrame(position_rows),
        "nav": pd.DataFrame(nav_rows),
        "execution_decisions": pd.DataFrame(execution_decisions),
        "corporate_actions": pd.DataFrame(action_rows),
        "open_positions": pd.DataFrame([
            {"ts_code": code, **{key: value for key, value in pos.items() if key != "decision_id"}}
            for code, pos in positions.items()
        ]),
    }


def _scenario(strategy: StrategyConfig, name: str) -> CostScenario:
    raw = strategy.execution["cost_scenarios"].get(name)
    if raw is None:
        raise ValueError(f"unknown cost scenario: {name}")
    return CostScenario(
        name=name,
        slippage_each_side=float(raw["slippage_each_side"]),
        buy_commission=float(raw["buy_commission"]),
        sell_commission=float(raw["sell_commission"]),
        stamp_duty=float(raw["stamp_duty"]),
    )


def _execution_row(by_key: pd.DataFrame, day: pd.Timestamp, code: str) -> pd.Series:
    try:
        row = by_key.loc[(day, code)]
    except KeyError as exc:
        raise BacktestDataError(f"execution key absent from coverage grid: {day.date()} {code}") from exc
    if isinstance(row, pd.DataFrame):
        raise ValueError(f"duplicate execution key: {day.date()} {code}")
    return row


def _prepare_rex_execution(execution: pd.DataFrame, strategy: StrategyConfig) -> pd.DataFrame:
    """Attach the PIT-safe MA5 used by the H10/extension close predicates.

    The execution panel contains only prices and tradability fields by
    contract.  Recomputing MA5 from economic closes keeps the REX exit rule
    independent of the selection ledger and prevents using any future row.
    """
    out = execution.copy()
    if "trade_date" not in out or "ts_code" not in out:
        raise ValueError("REX execution panel requires trade_date and ts_code")
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="raise", utc=True).dt.normalize()
    extension = strategy.execution.get("extension", {})
    configured = strategy.features.get("ma5", {})
    window = int(extension.get("ma_window_sessions", configured.get("window_sessions", 5)))
    if window <= 0:
        raise ValueError("REX MA5 window must be positive")
    if "ma5" not in out.columns:
        out = out.sort_values(["ts_code", "trade_date"], kind="mergesort")
        out["ma5"] = out.groupby("ts_code", sort=False)["economic_close"].transform(
            lambda values: values.rolling(window, min_periods=window).mean()
        )
    else:
        out["ma5"] = pd.to_numeric(out["ma5"], errors="coerce")
    return out


def _open_market_value(positions: dict[str, dict[str, Any]], by_key: pd.DataFrame, day: pd.Timestamp) -> float:
    value = 0.0
    for code, pos in positions.items():
        row = _execution_row(by_key, day, code)
        price = float(row.raw_open) if pd.notna(row.raw_open) else float(pos["last_raw_close"])
        value += float(pos["shares"]) * price
    return value


def _action_map(frame: pd.DataFrame) -> dict[pd.Timestamp, list[dict[str, Any]]]:
    if frame.empty:
        return {}
    out = frame.copy()
    out["ex_date"] = pd.to_datetime(out["ex_date"], errors="raise", utc=True).dt.normalize()
    return {day: group.to_dict("records") for day, group in out.groupby("ex_date", sort=True)}


__all__ = ["BacktestDataError", "CostScenario", "run_v3_backtest"]
