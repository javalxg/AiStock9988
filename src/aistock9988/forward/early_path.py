"""Deterministic paired repricing for the CAP1 early-path forward shadow."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from ..configuration import StrategyConfig


_TERMINAL_STATES = {
    "CONFIRMED",
    "BREAK_PENDING",
    "AMBIGUOUS_SAME_SESSION",
    "UNSCORABLE_DATA_GAP",
    "CONTROL_EXIT_BEFORE_WINDOW_END",
    "NEUTRAL",
}
_SELL_BLOCKED = {
    "MISSING_REQUIRED_DATA",
    "SUSPENDED",
    "LIMIT_DOWN",
    "ZERO_VOLUME",
    "OUT_OF_UNIVERSE",
}


class EarlyPathFailure(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class EarlyPathConfig:
    identity: Mapping[str, Any]
    control: Mapping[str, Any]
    path: Mapping[str, Any]
    paired_capital: Mapping[str, Any]
    evaluation: Mapping[str, Any]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EarlyPathConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        required = {"identity", "control", "path", "paired_capital", "evaluation"}
        if set(raw) != required or any(not isinstance(raw[name], Mapping) for name in required):
            raise ValueError(f"early-path config requires exactly {sorted(required)}")
        result = cls(**{name: dict(raw[name]) for name in required})
        result.validate()
        return result

    def validate(self) -> None:
        if str(self.identity.get("research_status")) != "forward_only":
            raise ValueError("early-path overlay must be forward_only")
        if int(self.identity.get("version", 0)) <= 0:
            raise ValueError("early-path overlay version must be positive")
        if int(self.path.get("observation_sessions", 0)) != 3:
            raise ValueError("early-path observation_sessions is frozen at 3")
        if float(self.path.get("confirm_pct", 0.0)) != 0.03:
            raise ValueError("early-path confirm_pct is frozen at 0.03")
        if float(self.path.get("break_pct", 0.0)) != -0.05:
            raise ValueError("early-path break_pct is frozen at -0.05")
        frozen = {
            "entry_reference": "raw_open",
            "same_session_both": "ambiguous_follow_control",
            "missing_data": "unscorable_follow_control",
            "control_exit_before_window": "follow_control",
            "exit": "next_tradable_raw_open",
        }
        if any(str(self.path.get(key)) != value for key, value in frozen.items()):
            raise ValueError("early-path path semantics differ from preregistration")
        capital = self.paired_capital
        required_true = {
            "inherit_control_buys",
            "restrict_early_proceeds_until_control_exit",
            "reserve_slot_until_control_exit",
            "fail_on_negative_unrestricted_cash",
        }
        if any(capital.get(key) is not True for key in required_true):
            raise ValueError("early-path paired-capital safeguards must remain enabled")
        if capital.get("allow_shadow_only_buys") is not False or capital.get("resize_control_buys") is not False:
            raise ValueError("early-path cannot create or resize control buys")
        if self.evaluation.get("parameter_sweep") is not False:
            raise ValueError("early-path parameter sweeps are forbidden")

    def validate_control(self, strategy: StrategyConfig) -> None:
        if str(self.control.get("strategy_id")) != strategy.strategy_id:
            raise ValueError("early-path control strategy_id mismatch")
        forward_start = pd.Timestamp(self.identity["forward_start"])
        control_start = pd.Timestamp(strategy.identity["forward_start"])
        if forward_start.normalize() != control_start.normalize():
            raise ValueError("early-path and control forward_start differ")

    @property
    def config_hash(self) -> str:
        payload = {
            "identity": dict(self.identity),
            "control": dict(self.control),
            "path": dict(self.path),
            "paired_capital": dict(self.paired_capital),
            "evaluation": dict(self.evaluation),
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()


def apply_early_path_overlay(
    *,
    control_result: Mapping[str, pd.DataFrame],
    execution_panel: pd.DataFrame,
    execution_sessions: tuple[str, ...],
    control_strategy: StrategyConfig,
    overlay: EarlyPathConfig,
    scenario_name: str,
) -> dict[str, pd.DataFrame]:
    """Reprice control trades without changing any control buy or slot decision."""
    overlay.validate_control(control_strategy)
    sessions = pd.DatetimeIndex(pd.to_datetime(execution_sessions, utc=True)).normalize()
    panel = execution_panel.copy()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], utc=True).dt.normalize()
    if panel.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("execution panel contains duplicate security/session keys")
    by_key = panel.set_index(["trade_date", "ts_code"])

    control_fills = control_result["fills"].copy().reset_index(drop=True)
    if control_fills.empty:
        return _empty_overlay(control_result)
    control_fills["trade_date"] = pd.to_datetime(control_fills["trade_date"], utc=True).dt.normalize()
    control_fills["trade_key"] = _trade_keys(control_fills)
    buys = control_fills[control_fills["side"].eq("BUY")].copy()
    sells = control_fills[control_fills["side"].eq("SELL")].copy()
    if buys["trade_key"].duplicated().any() or sells["trade_key"].duplicated().any():
        raise ValueError("paired overlay requires exactly one buy and one sell per trade key")
    if set(buys["trade_key"]) != set(sells["trade_key"]):
        raise ValueError("paired overlay requires every control buy to be closed")

    positions = control_result["positions"].copy()
    positions["trade_date"] = pd.to_datetime(positions["trade_date"], utc=True).dt.normalize()
    positions["trade_key"] = _trade_keys(positions)
    actions = control_result["corporate_actions"].copy()
    if not actions.empty:
        actions["trade_date"] = pd.to_datetime(actions["trade_date"], utc=True).dt.normalize()
        actions["trade_key"] = _trade_keys(actions)

    scenario = control_strategy.execution["cost_scenarios"][scenario_name]
    slippage = float(scenario["slippage_each_side"])
    sell_commission = float(scenario["sell_commission"])
    stamp_duty = float(scenario["stamp_duty"])
    path_rows: list[dict[str, Any]] = []
    replacements: dict[str, dict[str, Any]] = {}

    sell_map = sells.set_index("trade_key")
    for buy in buys.sort_values(["trade_date", "decision_id", "ts_code"], kind="mergesort").itertuples(index=False):
        trade_key = str(buy.trade_key)
        control_sell = sell_map.loc[trade_key]
        entry_day = pd.Timestamp(buy.trade_date)
        control_exit = pd.Timestamp(control_sell.trade_date)
        entry_index = sessions.get_indexer([entry_day])[0]
        if entry_index < 0:
            raise ValueError(f"control entry is outside execution sessions: {trade_key}")
        entry_row = _row(by_key, entry_day, str(buy.ts_code))
        entry_raw_open = float(entry_row.raw_open)
        confirm_level = entry_raw_open * (1.0 + float(overlay.path["confirm_pct"]))
        break_level = entry_raw_open * (1.0 + float(overlay.path["break_pct"]))
        state = "OBSERVING"
        break_trigger: pd.Timestamp | None = None

        for e_index in range(int(overlay.path["observation_sessions"])):
            session_index = entry_index + e_index
            if session_index >= len(sessions):
                state = "UNSCORABLE_DATA_GAP"
                day = sessions[-1]
                row = None
                reason = "OBSERVATION_SESSION_UNAVAILABLE"
            else:
                day = sessions[session_index]
                row = None
                reason = ""
                if day >= control_exit:
                    state = "CONTROL_EXIT_BEFORE_WINDOW_END"
                    reason = "CONTROL_SELL_OPEN_PRECEDES_PATH_OBSERVATION"
                else:
                    try:
                        row = _row(by_key, day, str(buy.ts_code))
                    except KeyError:
                        state = "UNSCORABLE_DATA_GAP"
                        reason = "EXECUTION_KEY_ABSENT"
                    if row is not None:
                        eligible = bool(row.execution_data_eligible)
                        high = pd.to_numeric(pd.Series([row.raw_high]), errors="coerce").iloc[0]
                        low = pd.to_numeric(pd.Series([row.raw_low]), errors="coerce").iloc[0]
                        if not eligible or not np.isfinite(high) or not np.isfinite(low):
                            state = "UNSCORABLE_DATA_GAP"
                            reason = "REQUIRED_RAW_HIGH_LOW_UNAVAILABLE"
                        else:
                            touch_confirm = bool(float(high) >= confirm_level)
                            touch_break = bool(float(low) <= break_level)
                            if touch_confirm and touch_break:
                                state = "AMBIGUOUS_SAME_SESSION"
                                reason = "BOTH_LEVELS_TOUCHED_ORDER_UNKNOWN"
                            elif touch_confirm:
                                state = "CONFIRMED"
                                reason = "CONFIRM_LEVEL_TOUCHED_FIRST"
                            elif touch_break:
                                state = "BREAK_PENDING"
                                break_trigger = day
                                reason = "BREAK_LEVEL_TOUCHED_BEFORE_CONFIRM"
                            elif e_index == int(overlay.path["observation_sessions"]) - 1:
                                state = "NEUTRAL"
                                reason = "NO_LEVEL_TOUCHED_THROUGH_E2_CLOSE"
                            else:
                                reason = "NO_LEVEL_TOUCHED_CONTINUE"

            high_value = np.nan if row is None else row.raw_high
            low_value = np.nan if row is None else row.raw_low
            path_rows.append({
                "trade_key": trade_key,
                "decision_id": str(buy.decision_id),
                "ts_code": str(buy.ts_code),
                "entry_session": entry_day,
                "e_index": e_index,
                "session": day,
                "prior_state": "OBSERVING",
                "raw_open": np.nan if row is None else row.raw_open,
                "raw_high": high_value,
                "raw_low": low_value,
                "raw_close": np.nan if row is None else row.raw_close,
                "confirm_level": confirm_level,
                "break_level": break_level,
                "touch_confirm": bool(np.isfinite(high_value) and float(high_value) >= confirm_level),
                "touch_break": bool(np.isfinite(low_value) and float(low_value) <= break_level),
                "resulting_state": state,
                "trigger_reason": reason,
                "decision_timestamp": _close_timestamp(day),
                "linked_control_exit_session": control_exit,
                "linked_control_exit_order_id": str(control_sell.order_id),
            })
            if state in _TERMINAL_STATES:
                break

        if state == "BREAK_PENDING" and break_trigger is not None:
            overlay_exit = _next_sell_session(
                by_key, sessions, str(buy.ts_code), break_trigger, control_exit
            )
            if overlay_exit < control_exit:
                exit_row = _row(by_key, overlay_exit, str(buy.ts_code))
                share_rows = positions[
                    positions["trade_key"].eq(trade_key)
                    & positions["trade_date"].le(overlay_exit)
                ].sort_values("trade_date", kind="mergesort")
                if share_rows.empty:
                    raise ValueError(f"control position ledger missing at overlay exit: {trade_key}")
                shares = float(share_rows.iloc[-1]["shares"])
                raw_price = float(exit_row.raw_open) * (1.0 - slippage)
                economic_price = float(exit_row.economic_open) * (1.0 - slippage)
                gross = raw_price * shares
                commission = gross * sell_commission
                duty = gross * stamp_duty
                replacements[trade_key] = {
                    "break_trigger_session": break_trigger,
                    "early_exit_session": overlay_exit,
                    "control_exit_session": control_exit,
                    "shares": shares,
                    "raw_price": raw_price,
                    "economic_price": economic_price,
                    "gross_value": gross,
                    "commission": commission,
                    "stamp_duty": duty,
                    "net_proceeds": gross - commission - duty,
                    "control_sell_order_id": str(control_sell.order_id),
                }

    shadow_fills = _replace_sells(control_fills, replacements, actions, buys)
    shadow_actions = _shadow_actions(actions, replacements)
    shadow_positions = positions[
        ~positions.apply(
            lambda row: str(row["trade_key"]) in replacements
            and pd.Timestamp(row["trade_date"]) >= replacements[str(row["trade_key"])]["early_exit_session"],
            axis=1,
        )
    ].copy()
    nav, shadow_fills = _rebuild_cash_nav(
        sessions=sessions,
        initial_cash=float(control_strategy.execution["initial_cash"]),
        fills=shadow_fills,
        positions=shadow_positions,
        actions=shadow_actions,
        replacements=replacements,
    )
    if bool(overlay.paired_capital["fail_on_negative_unrestricted_cash"]) and float(nav["unrestricted_cash"].min()) < -1e-8:
        raise EarlyPathFailure(
            "FAIL_PAIRED_FUNDING",
            "unrestricted shadow cash became negative",
        )

    paired_capital = pd.DataFrame([
        {
            "trade_key": key,
            "early_exit_session": value["early_exit_session"],
            "restricted_cash": value["net_proceeds"],
            "release_session": value["control_exit_session"],
            "linked_control_exit_order_id": value["control_sell_order_id"],
            "slot_reserved_until_release": True,
            "control_decision_nav": _decision_nav(control_result["nav"], buys, key),
            "minimum_portfolio_unrestricted_cash": float(nav["unrestricted_cash"].min()),
        }
        for key, value in sorted(replacements.items())
    ])
    reconciliation = _reconcile(control_fills, shadow_fills, path_rows)
    return {
        "orders": _shadow_orders(control_result["orders"], replacements),
        "fills": shadow_fills.drop(columns=["trade_key"], errors="ignore"),
        "position_events": _shadow_position_events(
            control_result["position_events"], replacements
        ),
        "positions": shadow_positions.drop(columns=["trade_key"], errors="ignore"),
        "nav": nav,
        "execution_decisions": control_result["execution_decisions"].copy(),
        "corporate_actions": shadow_actions.drop(columns=["trade_key"], errors="ignore"),
        "open_positions": pd.DataFrame(),
        "path_events": pd.DataFrame(path_rows),
        "paired_capital": paired_capital,
        "reconciliation": pd.DataFrame([reconciliation]),
    }


def _empty_overlay(control: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    result = {name: frame.copy() for name, frame in control.items()}
    result["path_events"] = pd.DataFrame()
    result["paired_capital"] = pd.DataFrame()
    result["reconciliation"] = pd.DataFrame([{
        "control_trade_count": 0,
        "shadow_trade_count": 0,
        "identical_buys": True,
        "one_sell_per_trade": True,
        "path_terminal_count": 0,
        "passed": True,
    }])
    return result


def _trade_keys(frame: pd.DataFrame) -> pd.Series:
    required = {"decision_id", "ts_code"}
    if not required.issubset(frame.columns):
        raise ValueError(f"trade identity columns missing: {sorted(required - set(frame.columns))}")
    return frame["decision_id"].astype(str) + "|" + frame["ts_code"].astype(str)


def _row(by_key: pd.DataFrame, day: pd.Timestamp, code: str) -> pd.Series:
    row = by_key.loc[(day, code)]
    if isinstance(row, pd.DataFrame):
        raise ValueError(f"duplicate execution key: {day.date()} {code}")
    return row


def _close_timestamp(day: pd.Timestamp) -> str:
    local = pd.Timestamp(day.date(), tz="Asia/Shanghai") + pd.Timedelta(hours=15)
    return local.isoformat()


def _next_sell_session(
    by_key: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    code: str,
    trigger: pd.Timestamp,
    control_exit: pd.Timestamp,
) -> pd.Timestamp:
    start = sessions.get_indexer([trigger])[0] + 1
    for day in sessions[start:]:
        if day > control_exit:
            break
        try:
            row = _row(by_key, day, code)
        except KeyError:
            continue
        if str(row.execution_status) in _SELL_BLOCKED:
            continue
        if pd.notna(row.raw_open) and pd.notna(row.economic_open):
            return day
    return control_exit


def _replace_sells(
    control_fills: pd.DataFrame,
    replacements: Mapping[str, Mapping[str, Any]],
    actions: pd.DataFrame,
    buys: pd.DataFrame,
) -> pd.DataFrame:
    shadow = control_fills.copy()
    buy_map = buys.set_index("trade_key")
    for key, replacement in replacements.items():
        index = shadow.index[shadow["trade_key"].eq(key) & shadow["side"].eq("SELL")]
        if len(index) != 1:
            raise ValueError(f"control sell reconciliation failed: {key}")
        buy = buy_map.loc[key]
        dividends = 0.0
        if not actions.empty:
            dividends = float(actions[
                actions["trade_key"].eq(key)
                & actions["trade_date"].le(replacement["early_exit_session"])
            ]["cash_dividend"].sum())
        proceeds = float(replacement["net_proceeds"])
        total_cost = float(buy.gross_value) + float(buy.commission)
        shadow.loc[index, "order_id"] = "overlay-" + shadow.loc[index, "order_id"].astype(str)
        shadow.loc[index, "trade_date"] = replacement["early_exit_session"]
        shadow.loc[index, "price"] = replacement["raw_price"]
        shadow.loc[index, "shares"] = replacement["shares"]
        shadow.loc[index, "gross_value"] = replacement["gross_value"]
        shadow.loc[index, "commission"] = replacement["commission"]
        shadow.loc[index, "stamp_duty"] = replacement["stamp_duty"]
        shadow.loc[index, "reason"] = "EARLY_PATH_BREAK"
        shadow.loc[index, "trigger_type"] = "EARLY_PATH_BREAK"
        shadow.loc[index, "economic_price"] = replacement["economic_price"]
        shadow.loc[index, "economic_return"] = (
            float(replacement["economic_price"]) / float(buy.economic_price) - 1.0
        )
        shadow.loc[index, "realized_pnl"] = proceeds + dividends - total_cost
        shadow.loc[index, "gap_return"] = np.nan
        shadow.loc[index, "gap_flag"] = True
        shadow.at[index[0], "exit_triggers"] = ["EARLY_PATH_BREAK"]
    return shadow


def _shadow_actions(
    actions: pd.DataFrame,
    replacements: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    if actions.empty:
        return actions.copy()
    keep = actions.apply(
        lambda row: str(row["trade_key"]) not in replacements
        or pd.Timestamp(row["trade_date"]) <= replacements[str(row["trade_key"])]["early_exit_session"],
        axis=1,
    )
    return actions[keep].copy()


def _rebuild_cash_nav(
    *,
    sessions: pd.DatetimeIndex,
    initial_cash: float,
    fills: pd.DataFrame,
    positions: pd.DataFrame,
    actions: pd.DataFrame,
    replacements: Mapping[str, Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_fills = fills.copy()
    cash = initial_cash
    rows: list[dict[str, Any]] = []
    for day in sessions:
        if not actions.empty:
            cash += float(actions.loc[actions["trade_date"].eq(day), "cash_dividend"].sum())
        day_fills = result_fills[result_fills["trade_date"].eq(day)]
        ordered = pd.concat([
            day_fills[day_fills["side"].eq("SELL")],
            day_fills[day_fills["side"].eq("BUY")],
        ])
        for index, fill in ordered.iterrows():
            if str(fill["side"]) == "BUY":
                cash -= float(fill["gross_value"]) + float(fill["commission"])
            else:
                cash += float(fill["gross_value"]) - float(fill["commission"]) - float(fill["stamp_duty"])
            result_fills.loc[index, "cash_after"] = cash
        restricted = sum(
            float(value["net_proceeds"])
            for value in replacements.values()
            if value["early_exit_session"] <= day < value["control_exit_session"]
        )
        reserved = sum(
            1 for value in replacements.values()
            if value["early_exit_session"] <= day < value["control_exit_session"]
        )
        day_positions = positions[positions["trade_date"].eq(day)]
        market_value = float(day_positions["market_value"].sum()) if not day_positions.empty else 0.0
        nav = cash + market_value
        rows.append({
            "trade_date": day,
            "cash": cash,
            "restricted_cash": restricted,
            "unrestricted_cash": cash - restricted,
            "market_value": market_value,
            "nav": nav,
            "open_positions": int(day_positions["trade_key"].nunique()) if not day_positions.empty else 0,
            "reserved_slots": reserved,
            "occupied_slots": (int(day_positions["trade_key"].nunique()) if not day_positions.empty else 0) + reserved,
            "gross_exposure": market_value / nav if nav > 0 else np.nan,
        })
    return pd.DataFrame(rows), result_fills


def _decision_nav(nav: pd.DataFrame, buys: pd.DataFrame, trade_key: str) -> float:
    buy = buys.set_index("trade_key").loc[trade_key]
    entry = pd.Timestamp(buy.trade_date)
    ordered = nav.copy()
    ordered["trade_date"] = pd.to_datetime(ordered["trade_date"], utc=True).dt.normalize()
    previous = ordered[ordered["trade_date"].lt(entry)].sort_values("trade_date", kind="mergesort")
    if previous.empty:
        raise EarlyPathFailure(
            "FAIL_RECONCILIATION",
            f"signal-close decision NAV is missing before entry: {trade_key}",
        )
    return float(previous.iloc[-1]["nav"])


def _shadow_position_events(
    control_events: pd.DataFrame,
    replacements: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    result = control_events.copy()
    if not result.empty:
        result["event_scope"] = "CONTROL_REFERENCE"
    overlay_rows = []
    for key, value in sorted(replacements.items()):
        decision_id, code = key.split("|", 1)
        overlay_rows.append({
            "event_id": f"early-path-{hashlib.sha256(key.encode()).hexdigest()[:16]}",
            "trade_date": value["early_exit_session"],
            "ts_code": code,
            "decision_id": decision_id,
            "event_type": "EARLY_PATH_EXIT_FILL",
            "state_before": "BREAK_PENDING",
            "state_after": "CLOSED_SLOT_RESERVED",
            "reason": "EARLY_PATH_BREAK",
            "event_scope": "SHADOW_OVERLAY",
        })
    if overlay_rows:
        result = pd.concat([result, pd.DataFrame(overlay_rows)], ignore_index=True, sort=False)
    return result


def _shadow_orders(
    orders: pd.DataFrame,
    replacements: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    result = orders.copy()
    if result.empty:
        return result
    keys = _trade_keys(result)
    for key, replacement in replacements.items():
        mask = keys.eq(key) & result["side"].eq("SELL") & result["status"].eq("FILLED")
        if mask.any():
            result.loc[mask, "order_id"] = "overlay-" + result.loc[mask, "order_id"].astype(str)
            result.loc[mask, "decision_session"] = replacement["break_trigger_session"]
            result.loc[mask, "execution_session"] = replacement["early_exit_session"]
            result.loc[mask, "execution_price"] = replacement["raw_price"]
            result.loc[mask, "filled_shares"] = replacement["shares"]
            result.loc[mask, "reason"] = "EARLY_PATH_BREAK"
    return result


def _reconcile(
    control: pd.DataFrame,
    shadow: pd.DataFrame,
    path_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    control_buys = control[control["side"].eq("BUY")].sort_values("trade_key", kind="mergesort")
    shadow_buys = shadow[shadow["side"].eq("BUY")].sort_values("trade_key", kind="mergesort")
    columns = ["trade_key", "trade_date", "ts_code", "decision_id", "shares", "price"]
    identical_buys = control_buys[columns].reset_index(drop=True).equals(
        shadow_buys[columns].reset_index(drop=True)
    )
    shadow_sells = shadow[shadow["side"].eq("SELL")]
    terminals = pd.DataFrame(path_rows)
    terminal_count = 0 if terminals.empty else int(terminals["resulting_state"].isin(_TERMINAL_STATES).sum())
    trade_count = int(len(control_buys))
    one_sell = bool(len(shadow_sells) == trade_count and not shadow_sells["trade_key"].duplicated().any())
    passed = bool(identical_buys and one_sell and terminal_count == trade_count)
    if not passed:
        raise EarlyPathFailure(
            "FAIL_RECONCILIATION",
            "paired overlay trade or path reconciliation failed",
        )
    return {
        "control_trade_count": trade_count,
        "shadow_trade_count": int(len(shadow_buys)),
        "identical_buys": identical_buys,
        "one_sell_per_trade": one_sell,
        "path_terminal_count": terminal_count,
        "passed": passed,
    }


__all__ = ["EarlyPathConfig", "EarlyPathFailure", "apply_early_path_overlay"]
