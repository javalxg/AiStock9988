"""Canonical deterministic event-driven engine for formal backtests.

All experiments use :func:`run_backtest` with the frozen candidate, selection,
and execution contracts. This package exposes one engine only.
"""
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


def run_backtest(
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
    execution = execution_panel[execution_panel["trade_date"].isin(sessions)].copy()
    if execution.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("execution panel contains duplicate security/session keys")
    by_key = execution.set_index(["trade_date", "ts_code"])
    candidate_view = candidate_ledger[candidate_ledger["candidate_status"].eq("IN_VIEW")].copy()
    candidate_map = {
        day: group.sort_values(["candidate_rank", "ts_code"], kind="mergesort")
        for day, group in candidate_view.groupby("asof", sort=True)
    }
    # Rank holding uses the complete model-ranked candidate snapshot, not only
    # the gated entry view. This keeps Top5 hold semantics separate from the
    # Top2 entry pool while remaining optional for other strategies.
    rank_cfg = strategy.execution.get("rank_holding", {})
    if bool(rank_cfg.get("enabled", False)) and "hold_rank" not in candidate_ledger.columns:
        raise ValueError("rank_holding is enabled but candidate ledger has no hold_rank column")
    hold_map = {
        day: group.sort_values(["hold_rank", "ts_code"], kind="mergesort")
        for day, group in candidate_ledger.groupby("asof", sort=True)
        if "hold_rank" in group.columns
    }
    rank_holding_enabled = bool(rank_cfg.get("enabled", False))
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
            "dividends_received": 0.0,
            "entry_session_index": entry_index,
            "scheduled_exit_index": entry_index + int(strategy.execution["hold_sessions_from_fill"]),
            "exit_reason": None, "last_raw_open": float(row.raw_open), "last_raw_close": float(row.raw_close),
            "last_economic_close": float(row.economic_close), "decision_id": order["decision_id"],
            "trailing_reference_economic_close": float(row.economic_close),
            "exit_triggers": [],
        }
        event(day, code, "ENTRY_FILL", "ENTRY_ATTEMPTED", "ACTIVE", "SIGNAL_ENTRY")

    def fill_sell(order: dict[str, Any], day: pd.Timestamp, code: str, row: pd.Series) -> None:
        nonlocal cash
        pos = positions[code]
        exit_price_mode = str(strategy.execution.get("exit_price", "next_tradable_raw_open"))
        if exit_price_mode == "same_session_raw_close":
            raw_value = row.raw_close
            economic_value = row.economic_close
        else:
            raw_value = row.raw_open
            economic_value = row.economic_open
        if pd.isna(raw_value) or pd.isna(economic_value):
            raise BacktestDataError(f"exit price unavailable: {day.date()} {code} {exit_price_mode}")
        raw_price = float(raw_value) * (1.0 - scenario.slippage_each_side)
        economic_price = float(economic_value) * (1.0 - scenario.slippage_each_side)
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
            "exit_triggers": list(pos.get("exit_triggers", [])),
        })
        event(day, code, "EXIT_FILL", "EXIT_PENDING", "CLOSED", str(pos["exit_reason"]))
        del positions[code]

    for session_index, day in enumerate(sessions):
        previous = sessions[session_index - 1] if session_index else None
        hold_pool: set[str] | None = None
        if rank_holding_enabled and previous in selection_map:
            # Every prior signal must have a sealed hold snapshot.  An empty
            # snapshot is a valid abstention and therefore means no symbol is
            # retained; a missing snapshot is a broken ledger contract.
            if previous not in hold_map:
                raise ValueError(f"rank_holding snapshot missing for signal {previous.date()}")
            hold_n = max(1, int(rank_cfg.get("hold_buffer_n", 5)))
            hold_rows = hold_map[previous]
            hold_pool = set(
                hold_rows.loc[hold_rows["hold_rank"].le(hold_n), "ts_code"].astype(str)
            )
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
                and not rank_holding_enabled
                and session_index >= int(pos["scheduled_exit_index"])
            ):
                pos["state"] = "EXIT_PENDING"
                pos["exit_reason"] = "TIME_EXIT"
                event(day, code, "EXIT_TRIGGER", "ACTIVE", "EXIT_PENDING", "TIME_EXIT")
            if (
                pos["state"] == "ACTIVE"
                and hold_pool is not None
                and code not in hold_pool
            ):
                pos["state"] = "EXIT_PENDING"
                pos["exit_reason"] = "RANK_EXIT"
                pos["exit_triggers"] = ["RANK_EXIT"]
                event(day, code, "EXIT_TRIGGER", "ACTIVE", "EXIT_PENDING", "RANK_EXIT")
            if pos["state"] != "EXIT_PENDING":
                continue
            # Same-session-close contracts settle every pending exit after
            # today's open buys. This prevents a close sale (including a
            # retry from an earlier data gap) from funding the same day's
            # open purchase.
            if str(strategy.execution.get("exit_price", "next_tradable_raw_open")) == "same_session_raw_close":
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

        if previous in selection_map and previous in candidate_map:
            decision = selection_map[previous]
            candidates = candidate_map[previous]
            slots = max(0, int(strategy.portfolio["max_open_positions"]) - len(positions))
            desired = min(int(decision.desired_entries), slots)
            filled = 0
            current_open_value = _open_market_value(positions, by_key, day)
            decision_nav = float(nav_rows[-1]["nav"]) if nav_rows else initial_cash
            exposure_budget = max(0.0, float(strategy.portfolio["target_gross_exposure_cap"]) * decision_nav - current_open_value)
            weight_basis = str(getattr(decision, "target_weight_basis", "decision_nav"))
            if weight_basis not in {"decision_nav", "available_cash"}:
                raise ValueError(f"unsupported target_weight_basis: {weight_basis}")
            cash_fraction_policy = float(getattr(decision, "cash_fraction_policy", 1.0))
            if not 0.0 < cash_fraction_policy <= 1.0:
                raise ValueError("cash_fraction_policy must be in (0,1]")
            cash_pool = cash
            applied_cash_fraction = 1.0
            no_backfill = str(strategy.execution.get("buy_untradable", "")) == "no_backfill"
            attempted_buy_candidates = []
            if no_backfill:
                # Determine the candidates that can actually be executed at
                # T+1 before applying the weak-breadth single-survivor cap.
                for candidate in candidates.itertuples(index=False):
                    code = str(candidate.ts_code)
                    if code in positions:
                        continue
                    row = _execution_row(by_key, day, code)
                    if str(row.execution_status) not in {"MISSING_REQUIRED_DATA", "SUSPENDED", "LIMIT_UP", "ZERO_VOLUME", "OUT_OF_UNIVERSE"}:
                        attempted_buy_candidates.append(code)
                    if len(attempted_buy_candidates) >= desired:
                        break
                breadth = pd.to_numeric(candidates.get("market_breadth", pd.Series(dtype=float)), errors="coerce").dropna()
                breadth_now = float(breadth.iloc[0]) if not breadth.empty else np.nan
                if len(attempted_buy_candidates) == 1 and np.isfinite(breadth_now) and breadth_now < float(strategy.ranking.get("q70_gate", {}).get("market_breadth_min", 0.40)):
                    cash_pool *= cash_fraction_policy
                    applied_cash_fraction = cash_fraction_policy
            if weight_basis == "available_cash":
                exposure_budget = min(exposure_budget, cash_pool)
            attempted: set[str] = set()
            for attempt_no, candidate in enumerate(candidates.itertuples(index=False), start=1):
                if filled >= desired:
                    break
                if no_backfill and attempt_no > desired:
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
                    candidate_weight = getattr(candidate, "target_weight", np.nan)
                    if pd.notna(candidate_weight):
                        if weight_basis == "available_cash":
                            target_budget = max(0.0, float(candidate_weight)) * cash_pool
                        else:
                            target_budget = max(0.0, float(candidate_weight)) * decision_nav
                    else:
                        if weight_basis == "available_cash":
                            raise ValueError(
                                f"available_cash candidate {code} is missing target_weight"
                            )
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
                    # Preserve optional second-stage evidence without making
                    # the baseline ledger depend on those columns.
                    "stage1_rank": getattr(candidate, "stage1_rank", np.nan),
                    "selected_status": getattr(candidate, "selected_status", None),
                    "model_score": getattr(candidate, "model_score", np.nan),
                    "interaction_strength": getattr(candidate, "interaction_strength", np.nan),
                    "alpha_percentile": getattr(candidate, "alpha_percentile", np.nan),
                    "interaction_percentile": getattr(candidate, "interaction_percentile", np.nan),
                    "stage2_score": getattr(candidate, "stage2_score", np.nan),
                    "alpha_power_weight": getattr(candidate, "alpha_power_weight", np.nan),
                    "target_weight": getattr(candidate, "target_weight", np.nan),
                    "target_weight_basis": weight_basis,
                    "cash_fraction_applied": applied_cash_fraction,
                })

        for code, pos in list(positions.items()):
            row = _execution_row(by_key, day, code)
            status = str(row.execution_status)
            data_eligible = bool(row.execution_data_eligible)
            if not data_eligible:
                event(day, code, "HELD_DATA_GAP", pos["state"], pos["state"], str(row.missing_required_execution))
                continue
            previous_economic_close = float(pos["last_economic_close"])
            previous_raw_open = float(pos["last_raw_open"])
            previous_raw_close = float(pos["last_raw_close"])
            if pd.notna(row.raw_open):
                pos["last_raw_open"] = float(row.raw_open)
            if pd.notna(row.raw_close):
                pos["last_raw_close"] = float(row.raw_close)
            if pd.notna(row.economic_close):
                pos["last_economic_close"] = float(row.economic_close)
            if pos["state"] == "ACTIVE":
                stop_cfg = strategy.execution["stop"]
                stop_pct = float(stop_cfg["threshold_pct"])
                stop_mode = str(stop_cfg.get("mode", "from_entry"))
                if stop_mode == "trailing_from_last_close":
                    pos["trailing_reference_economic_close"] = previous_economic_close
                    reference = previous_economic_close
                else:
                    reference = float(pos["entry_economic_price"])
                if float(pos["last_economic_close"]) / reference - 1.0 <= stop_pct:
                    pos["state"] = "EXIT_PENDING"
                    pos["exit_reason"] = "STOP_LOSS"
                    pos["exit_triggers"] = [f"STOP_LOSS({stop_mode})"]
                    event(day, code, "EXIT_TRIGGER", "ACTIVE", "EXIT_PENDING", "STOP_LOSS")
                else:
                    take_profit_pct = strategy.execution.get("take_profit_pct")
                    if (
                        take_profit_pct is not None
                        and float(pos["last_economic_close"]) / float(pos["entry_economic_price"]) - 1.0
                        >= float(take_profit_pct)
                    ):
                        pos["state"] = "EXIT_PENDING"
                        pos["exit_reason"] = "TAKE_PROFIT"
                        pos["exit_triggers"] = ["TAKE_PROFIT"]
                        event(day, code, "EXIT_TRIGGER", "ACTIVE", "EXIT_PENDING", "TAKE_PROFIT")
                    else:
                        technical = _technical_exit_trigger(
                            strategy.execution.get("sell_conditions", ()),
                            previous_raw_open,
                            previous_raw_close,
                            row,
                        )
                        if technical is not None:
                            pos["state"] = "EXIT_PENDING"
                            pos["exit_reason"] = "TECHNICAL_EXIT"
                            pos["exit_triggers"] = [f"TECHNICAL_EXIT({technical})"]
                            event(day, code, "EXIT_TRIGGER", "ACTIVE", "EXIT_PENDING", technical)
                early_path = strategy.execution.get("early_path_exit", {})
                observation_index = int(
                    early_path.get("observation_sessions_from_fill", 0)
                ) - 1
                if (
                    pos["state"] == "ACTIVE"
                    and bool(early_path.get("enabled", False))
                    and session_index - int(pos["entry_session_index"])
                    == observation_index
                    and float(pos["last_economic_close"])
                    <= float(pos["entry_economic_price"])
                ):
                    pos["state"] = "EXIT_PENDING"
                    pos["exit_reason"] = "EARLY_PATH_EXIT"
                    pos["exit_triggers"] = ["EARLY_PATH_EXIT(E2_NONPOSITIVE)"]
                    event(
                        day,
                        code,
                        "EXIT_TRIGGER",
                        "ACTIVE",
                        "EXIT_PENDING",
                        "EARLY_PATH_EXIT(E2_NONPOSITIVE)",
                    )

        # Some contracts (including delta's q70 contract) execute an exit on
        # the same session close that produced the trigger.  The first exit
        # pass above remains the next-open path for positions that were
        # already pending; this pass closes newly-triggered positions without
        # introducing a second backtest engine.
        if str(strategy.execution.get("exit_price", "next_tradable_raw_open")) == "same_session_raw_close":
            for code, pos in list(positions.items()):
                if pos["state"] != "EXIT_PENDING":
                    continue
                row = _execution_row(by_key, day, code)
                status = str(row.execution_status)
                order = new_order(day, str(pos["decision_id"]), code, "SELL", float(pos["shares"]), str(pos["exit_reason"]))
                if status == "MISSING_REQUIRED_DATA":
                    reject(order, day, "SELL_MISSING_REQUIRED_DATA")
                    event(day, code, "EXIT_RETRY", "EXIT_PENDING", "EXIT_PENDING", status)
                    continue
                if status in {"SUSPENDED", "LIMIT_DOWN", "ZERO_VOLUME", "OUT_OF_UNIVERSE"}:
                    reject(order, day, f"SELL_{status}")
                    event(day, code, "EXIT_RETRY", "EXIT_PENDING", "EXIT_PENDING", status)
                    continue
                try:
                    fill_sell(order, day, code, row)
                except BacktestDataError:
                    reject(order, day, "SELL_MISSING_EXIT_PRICE")
                    event(day, code, "EXIT_RETRY", "EXIT_PENDING", "EXIT_PENDING", "MISSING_EXIT_PRICE")

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
                "exit_triggers": pos.get("exit_triggers", []),
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


def _technical_exit_trigger(
    names: Any,
    previous_open: float,
    previous_close: float,
    row: pd.Series,
) -> str | None:
    """Evaluate close-known candle exits without consulting future sessions."""
    if not names or not np.isfinite(previous_open) or not np.isfinite(previous_close):
        return None
    current_open = float(row.raw_open) if pd.notna(row.raw_open) else np.nan
    current_high = float(row.raw_high) if pd.notna(row.raw_high) else np.nan
    current_close = float(row.raw_close) if pd.notna(row.raw_close) else np.nan
    if not np.isfinite(current_open) or not np.isfinite(current_high) or not np.isfinite(current_close):
        return None
    for name in names:
        name = str(name)
        if name == "shadow_upper":
            body = abs(current_close - current_open)
            upper_shadow = current_high - max(current_close, current_open)
            if body > 0 and upper_shadow > body * 2:
                return name
        elif name == "yin_bao_yang":
            if (
                np.isfinite(previous_open)
                and current_close < current_open
                and previous_close > previous_open
                and current_open >= previous_close
                and current_close <= previous_open
            ):
                return name
    return None


__all__ = ["BacktestDataError", "CostScenario", "run_backtest"]
