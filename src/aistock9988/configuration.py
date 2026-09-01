"""The versioned, user-editable strategy configuration."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import pandas as pd
import yaml


_SAFE_ID = re.compile(r"^[A-Za-z0-9_.:-]{3,96}$")
_MODES = {"rules"}
_OPS = {"gt", "ge", "lt", "le", "between", "cross_above"}
_TECHNICAL_EXIT_NAMES = {"shadow_upper", "yin_bao_yang"}


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _identifier(value: Any, name: str) -> str:
    result = str(value or "")
    if not _SAFE_ID.fullmatch(result):
        raise ValueError(f"{name} must be a safe non-empty identifier")
    return result


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(_thaw(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _validate_expression(node: Any, path: str = "stage1.expression") -> None:
    expression = _mapping(node, path)
    boolean_keys = [key for key in ("all", "any", "not") if key in expression]
    if boolean_keys:
        if len(boolean_keys) != 1 or len(expression) != 1:
            raise ValueError(f"{path} must contain exactly one boolean operator")
        key = boolean_keys[0]
        children = expression[key]
        if key == "not":
            _validate_expression(children, f"{path}.not")
            return
        if not isinstance(children, (list, tuple)) or not children:
            raise ValueError(f"{path}.{key} must be a non-empty list")
        for index, child in enumerate(children):
            _validate_expression(child, f"{path}.{key}[{index}]")
        return
    if not {"left", "op"} <= set(expression):
        raise ValueError(f"{path} condition requires left/op")
    if expression["op"] not in _OPS:
        raise ValueError(f"{path} uses unsupported operator: {expression['op']}")
    if expression["op"] == "between":
        if not {"lower", "upper"} <= set(expression):
            raise ValueError(f"{path} between condition requires lower/upper")
    elif "right" not in expression and "value" not in expression:
        raise ValueError(f"{path} condition requires right or value")


@dataclass(frozen=True)
class StrategyConfig:
    identity: Mapping[str, Any]
    universe: Mapping[str, Any]
    data_policy: Mapping[str, Any]
    decision: Mapping[str, Any]
    features: Mapping[str, Any]
    stage1: Mapping[str, Any]
    ranking: Mapping[str, Any]
    portfolio: Mapping[str, Any]
    execution: Mapping[str, Any]
    acceptance: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "StrategyConfig":
        source = _mapping(raw, "strategy config")
        if "identity" not in source:
            raise ValueError("strategy config requires the current identity-based contract")
        sections = {
            name: _mapping(source.get(name, {}), name)
            for name in (
                "identity", "universe", "data_policy", "decision", "features", "stage1",
                "ranking", "portfolio", "execution", "acceptance",
            )
        }
        identity = sections["identity"]
        _identifier(identity.get("strategy_id"), "identity.strategy_id")
        mode = str(identity.get("mode", "rules"))
        if mode not in _MODES:
            raise ValueError(f"identity.mode must be one of {sorted(_MODES)}")
        if int(identity.get("version", 0)) <= 0:
            raise ValueError("identity.version must be positive")
        research_status = str(identity.get("research_status", "historical"))
        if research_status == "forward_only":
            forward_start = identity.get("forward_start")
            if not forward_start:
                raise ValueError("forward_only strategies require identity.forward_start")
            try:
                pd_timestamp = pd.Timestamp(forward_start)
            except Exception as exc:
                raise ValueError("identity.forward_start must be a valid date") from exc
            if pd_timestamp.tzinfo is not None:
                pd_timestamp = pd_timestamp.tz_convert("UTC")
            if pd_timestamp.normalize() != pd_timestamp:
                raise ValueError("identity.forward_start must be normalized to a session day")

        universe = sections["universe"]
        min_listed_sessions = int(universe.get("min_listed_sessions", 0))
        if min_listed_sessions < 0:
            raise ValueError("universe.min_listed_sessions must be non-negative")

        decision = sections["decision"]
        if str(decision.get("frequency", "")) not in {"daily", "weekly"}:
            raise ValueError("decision.frequency must be daily or weekly")
        if int(decision.get("entry_delay_sessions", 0)) != 1:
            raise ValueError("the first production contract requires entry_delay_sessions=1")

        data_policy = sections["data_policy"]
        dense = _mapping(data_policy.get("dense_required", {}), "data_policy.dense_required")
        if not {"selection", "training", "execution"} <= set(dense):
            raise ValueError("data_policy.dense_required requires selection/training/execution")
        for stage in ("selection", "training", "execution"):
            if not isinstance(dense[stage], (list, tuple)) or not dense[stage]:
                raise ValueError(f"data_policy.dense_required.{stage} must be a non-empty list")
        sparse = data_policy.get("sparse_event", ())
        optional = data_policy.get("optional_enrichment", ())
        source_availability = _mapping(data_policy.get("source_availability", {}), "data_policy.source_availability")
        if not isinstance(sparse, (list, tuple)) or not isinstance(optional, (list, tuple)):
            raise ValueError("data_policy sparse_event and optional_enrichment must be lists")
        groups = {
            "dense_required": set(str(item) for stage in dense.values() for item in stage),
            "sparse_event": set(str(item) for item in sparse),
            "optional_enrichment": set(str(item) for item in optional),
        }
        overlap = (groups["dense_required"] & groups["sparse_event"]) | (
            groups["dense_required"] & groups["optional_enrichment"]
        ) | (groups["sparse_event"] & groups["optional_enrichment"])
        if overlap:
            raise ValueError(f"data sources cannot belong to multiple policy classes: {sorted(overlap)}")
        missing_actions = _mapping(data_policy.get("missing_action", {}), "data_policy.missing_action")
        expected_actions = {
            "selection": "exclude_stock_session",
            "training": "exclude_sample",
            "entry": "skip_and_ranked_fallback",
            "held_position": "carry_last_mark_and_retry",
        }
        if any(missing_actions.get(key) != value for key, value in expected_actions.items()):
            raise ValueError("data_policy.missing_action does not match the V3 eligibility contract")
        if "index_daily_ts" in set(str(value) for value in dense["selection"]):
            if source_availability.get("index_daily_ts") != "eod_trade_date_close":
                raise ValueError("index_daily_ts selection input requires eod_trade_date_close availability")

        portfolio = sections["portfolio"]
        candidate_n = int(portfolio.get("candidate_view_size", 0))
        entry_n = int(portfolio.get("entries_per_decision", 0))
        position_n = int(portfolio.get("max_open_positions", 0))
        if not (candidate_n >= entry_n > 0 and position_n > 0):
            raise ValueError("candidate_view_size >= entries_per_decision > 0 and max_open_positions > 0 are required")
        gross_cap = float(portfolio.get("target_gross_exposure_cap", 0.0))
        if not 0 < gross_cap <= 1:
            raise ValueError("target_gross_exposure_cap must be in (0,1]")
        sizing = _mapping(portfolio.get("sizing", {}), "portfolio.sizing")
        if sizing.get("method") != "fixed_fraction_of_decision_nav":
            raise ValueError("the first production contract requires fixed_fraction_of_decision_nav sizing")
        if not 0 < float(sizing.get("value", 0.0)) <= 1:
            raise ValueError("portfolio.sizing.value must be in (0,1]")

        execution = sections["execution"]
        if execution.get("entry_price") not in {
            "next_session_raw_open",
            "next_session_5min_confirmed_open",
        }:
            raise ValueError(
                "execution.entry_price must be next_session_raw_open or "
                "next_session_5min_confirmed_open"
            )
        if int(execution.get("hold_sessions_from_fill", 0)) <= 0:
            raise ValueError("execution.hold_sessions_from_fill must be positive")
        extension = _mapping(
            execution.get("time_exit_extension", {}),
            "execution.time_exit_extension",
        )
        if bool(extension.get("enabled", False)):
            if str(extension.get("condition")) != "prior_close_unrealized_positive":
                raise ValueError(
                    "time_exit_extension.condition must be prior_close_unrealized_positive"
                )
            extended_hold = int(extension.get("extended_hold_sessions_from_fill", 0))
            if extended_hold <= int(execution["hold_sessions_from_fill"]):
                raise ValueError(
                    "time_exit_extension extended hold must exceed the base hold"
                )
        exit_price = str(execution.get("exit_price", "next_tradable_raw_open"))
        if exit_price not in {"next_tradable_raw_open", "same_session_raw_close"}:
            raise ValueError(
                "execution.exit_price must be next_tradable_raw_open or same_session_raw_close"
            )
        stop = _mapping(execution.get("stop", {}), "execution.stop")
        stop_mode = str(stop.get("mode", "from_entry"))
        if stop_mode not in {"from_entry", "trailing_from_last_close"}:
            raise ValueError("execution.stop.mode must be from_entry or trailing_from_last_close")
        if float(stop.get("threshold_pct", 0.0)) >= 0:
            raise ValueError("execution.stop.threshold_pct must be negative")
        take_profit = execution.get("take_profit_pct")
        if take_profit is not None and float(take_profit) < 0:
            raise ValueError("execution.take_profit_pct must be non-negative when enabled")
        sell_conditions = execution.get("sell_conditions", ())
        if not isinstance(sell_conditions, (list, tuple)):
            raise ValueError("execution.sell_conditions must be a list")
        unknown_exits = set(str(item) for item in sell_conditions) - _TECHNICAL_EXIT_NAMES
        if unknown_exits:
            raise ValueError(f"unsupported execution.sell_conditions: {sorted(unknown_exits)}")
        if int(execution.get("lot_size", 0)) <= 0:
            raise ValueError("execution.lot_size must be positive")
        if execution.get("accounting_basis") != "raw_with_corporate_actions":
            raise ValueError("formal backtests require raw_with_corporate_actions")
        buy_untradable = str(execution.get("buy_untradable", "same_decision_ranked_replacement"))
        if buy_untradable not in {"no_backfill", "same_decision_ranked_replacement"}:
            raise ValueError(
                "execution.buy_untradable must be no_backfill or "
                "same_decision_ranked_replacement"
            )
        if research_status == "forward_only":
            amount_unit = str(execution.get("amount_unit", ""))
            if amount_unit not in {"thousand_rmb", "rmb"}:
                raise ValueError("forward_only strategies require execution.amount_unit=thousand_rmb or rmb")
            if float(execution.get("amount_unit_multiplier", 0.0)) <= 0:
                raise ValueError("execution.amount_unit_multiplier must be positive")
        scenarios = _mapping(execution.get("cost_scenarios", {}), "execution.cost_scenarios")
        if not {"base", "stress"} <= set(scenarios):
            raise ValueError("execution.cost_scenarios requires base and stress")

        stage1 = sections["stage1"]
        if "expression" not in stage1:
            raise ValueError("the first strategy requires stage1.expression")
        _validate_expression(stage1["expression"])
        terms = sections["ranking"].get("terms")
        if not isinstance(terms, (list, tuple)) or not terms:
            raise ValueError("ranking.terms must be a non-empty list")
        total_weight = sum(float(_mapping(term, "ranking term").get("weight", 0.0)) for term in terms)
        if abs(total_weight - 1.0) > 1e-9:
            raise ValueError("ranking term weights must sum to 1")
        # Optional second-stage reranking is deliberately opt-in.  Keeping the
        # contract under ``ranking`` makes the experiment reproducible while
        # preserving the baseline behaviour when the section is absent.
        stage2 = _mapping(sections["ranking"].get("stage2", {}), "ranking.stage2")
        if bool(stage2.get("enabled", False)):
            if str(stage2.get("method", "native_interaction_strength")) != "native_interaction_strength":
                raise ValueError("ranking.stage2.method must be native_interaction_strength")
            interaction_weight = float(stage2.get("interaction_weight", 0.10))
            if not 0.0 <= interaction_weight <= 1.0:
                raise ValueError("ranking.stage2.interaction_weight must be in [0,1]")
            if int(stage2.get("interaction_top_k", 5)) <= 0:
                raise ValueError("ranking.stage2.interaction_top_k must be positive")
            stage2_view = int(stage2.get("candidate_view_size", candidate_n))
            if stage2_view < entry_n or stage2_view > candidate_n:
                raise ValueError("ranking.stage2.candidate_view_size must be within portfolio candidate view")
            stage2_entries = int(stage2.get("entries_per_decision", entry_n))
            if not 0 < stage2_entries <= position_n:
                raise ValueError("ranking.stage2.entries_per_decision must be within max_open_positions")
            alpha_power = float(stage2.get("alpha_power", 7.0))
            if alpha_power < 0.0:
                raise ValueError("ranking.stage2.alpha_power must be non-negative")

        return cls(**{name: _freeze(value) for name, value in sections.items()})

    @classmethod
    def from_yaml(cls, path: str | Path) -> "StrategyConfig":
        return cls.from_mapping(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})

    @property
    def strategy_id(self) -> str:
        return str(self.identity["strategy_id"])

    @property
    def mode(self) -> str:
        return str(self.identity["mode"])

    @property
    def strategy_type(self) -> str:
        return self.mode

    @property
    def feature_set_id(self) -> str:
        return str(self.identity.get("feature_set_id", f"{self.strategy_id}.features"))

    @property
    def selection(self) -> Mapping[str, Any]:
        return self.portfolio

    @property
    def candidate(self) -> Mapping[str, Any]:
        return self.stage1

    def to_dict(self) -> dict[str, Any]:
        return {name: _thaw(getattr(self, name)) for name in self.__dataclass_fields__}

    @property
    def config_hash(self) -> str:
        return _canonical_hash(self.to_dict())


__all__ = ["StrategyConfig"]
