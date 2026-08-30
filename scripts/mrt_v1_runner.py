"""Run the frozen MRT-V1 mild-rebound state-sequence rule.

MRT is deliberately a transparent, one-pass diagnostic.  Every security/session
in the PIT coverage grid receives an event-ledger row, including rejected rows;
only rows that pass both the trend setup and same-day shock become ranked
candidates for the shared V3 accounting engine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aistock9988.backtest.v3_engine import run_v3_backtest
from aistock9988.configuration import ModelConfig, StrategyConfig
from aistock9988.data.bundle import build_data_bundle, load_trading_calendar
from aistock9988.features.mrt import build_mrt_feature_ledger
from aistock9988.planning import RunRequest, compile_run_plan
from aistock9988.reporting.v3_metrics import summarize_v3
from aistock9988.selection.pipeline import evaluate_expression
from aistock9988.time.session import session_close


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STRATEGY = ROOT / "configs/strategy/mrt_v1.yaml"
DEFAULT_MODEL = ROOT / "configs/model/disabled.yaml"
DEFAULT_OUTPUT = ROOT / "docs/council_20260828/MRT_V1_DIAGNOSTIC_2026_TO_0828"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any, *, replace: bool = False) -> None:
    if path.exists() and not replace:
        raise FileExistsError(f"immutable artifact already exists: {path}")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _frozen_thresholds(strategy: StrategyConfig) -> dict[str, float]:
    """Read the single frozen MRT contract from YAML and reject drift."""
    expression = strategy.stage1.get("expression", {})
    conditions = expression.get("all", ()) if hasattr(expression, "get") else ()
    by_left: dict[str, list[Any]] = {}
    for item in conditions:
        if hasattr(item, "get"):
            by_left.setdefault(str(item.get("left")), []).append(item)
    expected = {
        "market_excess_ret10": ("gt", 0.0),
        "industry_excess_ret10": ("gt", 0.0),
        "ret60": ("gt", 0.0),
        "ret20": ("le", 0.25),
        "dist_ma60": ("gt", 0.0),
        "vol20_pct": ("le", 0.85),
        "shock_close_lt_open": ("ge", 1.0),
        "pct_chg": ("le", -5.0),
        "shock_amount_ratio": ("ge", 1.5),
        "shock_open_ok": ("ge", 1.0),
        "shock_close_ok": ("ge", 1.0),
        "pit_st": ("le", 0.0),
        "list_age_sessions": ("ge", 60.0),
        "execution_data_eligible": ("ge", 1.0),
        "shock_tradable": ("ge", 1.0),
    }
    for left, (op, value) in expected.items():
        items = by_left.get(left, [])
        if not any(str(item.get("op")) == op and float(item.get("value")) == value for item in items):
            raise ValueError(f"MRT frozen contract drift at stage1.expression.{left}")
    if not any(
        str(item.get("left")) == "dist_ma60"
        and str(item.get("op")) == "le"
        and float(item.get("value")) == 0.15
        for item in conditions if hasattr(item, "get")
    ):
        raise ValueError("MRT frozen contract drift at stage1.expression.dist_ma60 upper bound")
    terms = strategy.ranking.get("terms", ())
    if len(terms) != 1 or str(terms[0].get("feature")) != "shock_amount_ratio" or str(terms[0].get("direction")) != "desc":
        raise ValueError("MRT frozen ranking contract drift")
    values = {left: value for left, (_, value) in expected.items()}
    values["dist_ma60_upper"] = 0.15
    return values


def _build_features(
    bundle: Any,
    strategy: StrategyConfig,
    signal_sessions: pd.DatetimeIndex,
    output: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Delegate feature construction to the versioned MRT provider.

    The provider owns the PIT industry resolution and exposes its audit frame
    through ``DataFrame.attrs``; this runner only adds the explicit list-age
    exclusion used by the frozen non-new-stock shock rule.
    """
    out = build_mrt_feature_ledger(bundle, strategy)
    provider_industry_audit = out.attrs.get("industry_audit", pd.DataFrame())
    out["asof"] = pd.to_datetime(out["asof"], utc=True).dt.normalize()
    out["ts_code"] = out["ts_code"].astype(str).str.upper()
    universe = bundle.universe.copy()
    universe["asof"] = pd.to_datetime(universe["asof"], utc=True).dt.normalize()
    universe["ts_code"] = universe["ts_code"].astype(str).str.upper()
    out = out.merge(universe[["asof", "ts_code", "list_date", "name", "pit_st"]], on=["asof", "ts_code"], how="left", validate="one_to_one")
    execution_meta = bundle.execution.copy()
    execution_meta["asof"] = pd.to_datetime(execution_meta.pop("trade_date"), utc=True).dt.normalize()
    execution_meta["ts_code"] = execution_meta["ts_code"].astype(str).str.upper()
    out = out.merge(
        execution_meta[["asof", "ts_code", "suspension_evidence"]],
        on=["asof", "ts_code"], how="left", validate="one_to_one",
    )
    out["list_date"] = pd.to_datetime(out["list_date"], utc=True, errors="coerce").dt.normalize()
    out["list_age_days"] = (out["asof"] - out["list_date"]).dt.days
    calendar = pd.DatetimeIndex(pd.to_datetime(bundle.calendar["session"], utc=True)).normalize()
    calendar_ns = calendar.asi8
    asof_ns = pd.DatetimeIndex(out["asof"]).asi8
    listed_ns = pd.DatetimeIndex(out["list_date"].fillna(pd.Timestamp("1900-01-01", tz="UTC"))).asi8
    age_sessions = np.searchsorted(calendar_ns, asof_ns, side="right") - np.searchsorted(calendar_ns, listed_ns, side="left")
    out["list_age_sessions"] = age_sessions.astype(float)
    out.loc[out["list_date"].isna(), "list_age_sessions"] = np.nan
    out["industry_covered"] = out["industry"].notna()
    out["mrt_feature_ready"] = out["feature_ready"].astype(bool)
    out["shock_tradable"] = (
        out["execution_status"].eq("TRADABLE") & ~out["suspension_evidence"].fillna(False).astype(bool)
    ).astype(float)
    out["feature_ready"] = out["mrt_feature_ready"]
    industry_audit = provider_industry_audit
    if isinstance(industry_audit, pd.DataFrame):
        audits = industry_audit.to_dict("records")
    else:
        audits = list(industry_audit or [])
    membership_map = out[["asof", "ts_code", "industry"]].copy()
    membership_map["industry_covered"] = membership_map["industry"].notna()
    membership_map.to_parquet(output / "ledgers/industry_membership_resolution.parquet", index=False)
    return out.sort_values(["asof", "ts_code"], kind="mergesort").reset_index(drop=True), audits


def _build_event_ledger(features: pd.DataFrame, strategy: StrategyConfig) -> pd.DataFrame:
    """Evaluate frozen setup/shock predicates and retain all rejection causes."""
    f = features.copy()
    _frozen_thresholds(strategy)
    if "shock_tradable" not in f:
        f["shock_tradable"] = (~f["execution_status"].astype(str).isin(
            {"SUSPENDED", "ZERO_VOLUME", "MISSING_REQUIRED_DATA", "LIMIT_DOWN"}
        )).astype(float)
    configured = pd.Series(False, index=f.index, dtype=bool)
    ready = f["mrt_feature_ready"].astype(bool)
    if ready.any():
        configured.loc[ready] = evaluate_expression(f.loc[ready], strategy.stage1["expression"])
    reasons: list[list[str]] = []
    setup_pass: list[bool] = []
    shock_pass: list[bool] = []
    for row in f.itertuples(index=False):
        r: list[str] = []
        if not bool(row.mrt_feature_ready):
            r.append("MISSING_REQUIRED_FEATURE")
        if not bool(row.universe_pass):
            r.append("UNIVERSE_REJECTED")
        if not bool(row.selection_data_eligible):
            r.append(str(row.missing_required_selection or "SELECTION_DATA_MISSING"))
        if not bool(row.execution_data_eligible):
            r.append(str(row.missing_required_execution or "EXECUTION_DATA_MISSING"))
        if not bool(row.industry_covered):
            r.append("PIT_INDUSTRY_MISSING")
        setup = (
            bool(row.market_excess_ret10 > 0)
            and bool(row.industry_excess_ret10 > 0)
            and bool(row.ret60 > 0)
            and bool(row.ret20 <= 0.25)
            and bool(0 < row.dist_ma60 <= 0.15)
            and bool(row.vol20_pct <= 0.85)
        ) if bool(row.mrt_feature_ready) else False
        if not bool(row.market_excess_ret10 > 0): r.append("MARKET_EXCESS_RET10_NONPOSITIVE")
        if not bool(row.industry_excess_ret10 > 0): r.append("INDUSTRY_EXCESS_RET10_NONPOSITIVE")
        if not bool(row.ret60 > 0): r.append("RET60_NONPOSITIVE")
        if not bool(row.ret20 <= 0.25): r.append("RET20_ABOVE_25PCT")
        if not bool(0 < row.dist_ma60 <= 0.15): r.append("DIST_MA60_OUTSIDE_0_15")
        if not bool(row.vol20_pct <= 0.85): r.append("VOL20_ABOVE_CROSS_SECTIONAL_P85")
        shock = (
            bool(row.shock_close_lt_open >= 1.0)
            and bool(row.pct_chg <= -5.0)
            and bool(row.shock_amount_ratio >= 1.5)
            and bool(row.shock_open_ok >= 1.0)
            and bool(row.shock_close_ok >= 1.0)
            and not bool(row.pit_st)
            and bool(row.execution_data_eligible)
            and bool(row.shock_tradable >= 1.0)
            and bool(row.list_age_sessions >= 60)
        ) if bool(row.mrt_feature_ready) else False
        if not bool(row.shock_close_lt_open >= 1.0): r.append("CLOSE_NOT_BELOW_OPEN")
        if not bool(row.pct_chg <= -5.0): r.append("PCT_CHG_ABOVE_NEG5PCT")
        if not bool(row.shock_amount_ratio >= 1.5): r.append("AMOUNT_BELOW_1P5_ADV20_PRIOR")
        if not bool(row.shock_open_ok >= 1.0): r.append("OPEN_AT_OR_BELOW_DOWN_LIMIT")
        if not bool(row.shock_close_ok >= 1.0): r.append("CLOSE_AT_OR_BELOW_DOWN_LIMIT")
        if bool(row.pit_st): r.append("PIT_ST")
        if str(row.execution_status) in {"SUSPENDED", "ZERO_VOLUME", "MISSING_REQUIRED_DATA"}: r.append(f"EXECUTION_{row.execution_status}")
        if not bool(row.execution_data_eligible): r.append("EXECUTION_DATA_INELIGIBLE")
        if not bool(row.shock_tradable >= 1.0): r.append("SHOCK_NOT_TRADABLE")
        if not bool(row.list_age_sessions >= 60): r.append("NEW_STOCK_LT_60_TRADING_SESSIONS")
        setup_pass.append(setup)
        shock_pass.append(shock)
        reasons.append(sorted(set(r)))
    f["setup_pass"] = setup_pass
    f["shock_pass"] = shock_pass
    f["event_pass"] = f["setup_pass"] & f["shock_pass"]
    f["config_stage1_pass"] = configured
    if not f["event_pass"].eq(f["config_stage1_pass"]).all():
        mismatch = f.loc[
            f["event_pass"].ne(f["config_stage1_pass"]),
            ["asof", "ts_code", "event_pass", "config_stage1_pass"],
        ].head(5)
        raise AssertionError(
            "MRT hard-coded event contract diverges from strategy config: "
            f"{mismatch.to_dict('records')}"
        )
    f["event_status"] = np.where(f["event_pass"], "PASS", "REJECTED")
    f["rejection_reasons"] = ["" if not r and ok else ";".join(r) for r, ok in zip(reasons, f["event_pass"])]
    f["event_id"] = [hashlib.sha256(f"{strategy.config_hash}|{d.date()}|{c}".encode()).hexdigest() for d, c in zip(f["asof"], f["ts_code"])]
    f["candidate_rank"] = np.nan
    f["candidate_status"] = "NOT_IN_VIEW"
    f["candidate_snapshot_id"] = ""
    for day, group in f[f["event_pass"]].groupby("asof", sort=True):
        ranked = group.sort_values(
            ["shock_amount_ratio", "ts_code"], ascending=[False, True], kind="mergesort"
        )
        top = ranked.head(int(strategy.portfolio["candidate_view_size"]))
        snapshot = hashlib.sha256("|".join(f"{r.ts_code}:{i}" for i, r in enumerate(top.itertuples(), 1)).encode()).hexdigest()
        for rank, idx in enumerate(top.index, 1):
            f.loc[idx, "candidate_rank"] = rank
            f.loc[idx, "candidate_status"] = "IN_VIEW"
            f.loc[idx, "candidate_snapshot_id"] = snapshot
    f["amount_to_adv20_prior"] = pd.to_numeric(f["shock_amount_ratio"], errors="coerce")
    return f


def _build_selection_ledger(events: pd.DataFrame, strategy: StrategyConfig, signal_sessions: pd.DatetimeIndex) -> pd.DataFrame:
    policy_hash = hashlib.sha256(strategy.config_hash.encode()).hexdigest()
    rows: list[dict[str, Any]] = []
    for day in signal_sessions:
        group = events[(events["asof"].eq(day)) & events["candidate_status"].eq("IN_VIEW")]
        snapshot = str(group["candidate_snapshot_id"].iloc[0]) if not group.empty else ""
        rows.append({
            "decision_id": hashlib.sha256(f"{policy_hash}|{day.date()}|{snapshot}".encode()).hexdigest(),
            "asof": day,
            "desired_entries": int(strategy.portfolio["entries_per_decision"]),
            "target_weight_each": float(strategy.portfolio["sizing"]["value"]),
            "primary_rank_end": int(strategy.portfolio["entries_per_decision"]),
            "replacement_rank_end": int(strategy.portfolio["candidate_view_size"]),
            "candidate_snapshot_id": snapshot,
            "policy_id": strategy.strategy_id,
            "policy_hash": policy_hash,
            "context_hash": hashlib.sha256(f"{day.date()}|{strategy.config_hash}".encode()).hexdigest(),
        })
    return pd.DataFrame(rows)


def _acceptance(metrics: dict[str, Any], strategy: StrategyConfig) -> dict[str, Any]:
    pf = metrics["portfolio_profit_factor"]
    tests = {
        "profit_factor": pf is not None and float(pf) >= float(strategy.acceptance["portfolio_profit_factor_min"]),
        "max_drawdown": abs(float(metrics["max_drawdown"])) <= float(strategy.acceptance["max_drawdown_abs_max"]),
        "excluding_best_week": float(metrics["return_excluding_best_week"]) > float(strategy.acceptance["return_excluding_best_week_min_exclusive"]),
    }
    return {"passed": all(tests.values()), "tests": tests}


def _attach_execution_audit(
    events: pd.DataFrame,
    selection: pd.DataFrame,
    result: dict[str, pd.DataFrame],
    execution_sessions: tuple[str, ...],
) -> pd.DataFrame:
    """Join execution decisions and realized exits back to every event row."""
    out = events.copy()
    out = out.merge(selection[["asof", "decision_id"]], on="asof", how="left", validate="many_to_one")
    sessions = pd.DatetimeIndex(pd.to_datetime(execution_sessions, utc=True)).normalize()
    next_session = {sessions[i]: sessions[i + 1] for i in range(len(sessions) - 1)}
    out["execution_session"] = out["asof"].map(next_session)
    decisions = result.get("execution_decisions", pd.DataFrame()).copy()
    if decisions.empty:
        decisions = pd.DataFrame(columns=["asof", "ts_code", "chosen", "t1_execution_status", "reject_reason", "attempt_no"])
    else:
        decisions["signal_session"] = pd.to_datetime(decisions["signal_session"], utc=True).dt.normalize()
        decisions["ts_code"] = decisions["ts_code"].astype(str).str.upper()
        decisions = decisions.rename(columns={"signal_session": "asof", "execution_status": "t1_execution_status"})
        decisions = decisions[["asof", "ts_code", "chosen", "t1_execution_status", "reject_reason", "attempt_no"]]
    out = out.merge(decisions, on=["asof", "ts_code"], how="left", validate="one_to_one")
    out["execution_attempted"] = out["chosen"].notna()
    out["execution_audit_status"] = np.select(
        [~out["event_pass"].astype(bool), out["candidate_status"].ne("IN_VIEW"), out["execution_attempted"]],
        ["NOT_EVENT", "NOT_IN_VIEW", "ATTEMPTED"],
        default="IN_VIEW_NO_ATTEMPT",
    )
    fills = result.get("fills", pd.DataFrame()).copy()
    sells = fills[fills.get("side", pd.Series(dtype=str)).eq("SELL")].copy() if not fills.empty else pd.DataFrame()
    if sells.empty:
        sells = pd.DataFrame(columns=["decision_id", "ts_code", "trade_date", "reason", "economic_return", "realized_pnl"])
    else:
        sells["ts_code"] = sells["ts_code"].astype(str).str.upper()
        sells = sells[["decision_id", "ts_code", "trade_date", "reason", "economic_return", "realized_pnl"]].rename(
            columns={"trade_date": "exit_date", "reason": "exit_reason"}
        )
    out = out.merge(sells, on=["decision_id", "ts_code"], how="left", validate="one_to_one")
    return out


def run(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"immutable output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("configs", "manifests", "ledgers", "backtests", "diagnostics", "logs"):
        (output / name).mkdir()
    strategy = StrategyConfig.from_yaml(args.strategy)
    model = ModelConfig.from_yaml(args.model)
    calendar_start = str((pd.Timestamp(args.signal_start) - pd.Timedelta(days=500)).date())
    calendar = load_trading_calendar(calendar_start, args.execution_end)
    plan = compile_run_plan(strategy, model, RunRequest(args.signal_start, args.signal_end, args.execution_end, str(output), args.run_name), calendar["session"])
    shutil.copyfile(args.strategy, output / "configs/strategy.yaml")
    shutil.copyfile(args.model, output / "configs/model.yaml")
    _write_json(output / "plan.json", plan.to_dict())
    _write_json(output / "RUN_STATUS.json", {
        "run_name": args.run_name, "status": "RUNNING", "created_at": datetime.now(timezone.utc).isoformat(),
        "strategy_id": strategy.strategy_id, "strategy_hash": strategy.config_hash,
        "model_id": model.model_id, "model_hash": model.config_hash, "python": sys.version,
        "parameter_sweep": False, "diagnostic_history": True,
        "strategy_research_status": strategy.identity.get("research_status", "historical"),
    })
    print("phase=snapshot start", flush=True)
    bundle = build_data_bundle(plan, strategy, output)
    _write_json(output / "data_manifest.json", bundle.manifest)
    bundle.universe.to_parquet(output / "ledgers/universe_ledger.parquet", index=False)
    bundle.availability.to_parquet(output / "ledgers/data_availability_ledger.parquet", index=False)
    bundle.execution.to_parquet(output / "ledgers/execution_panel.parquet", index=False)
    signal_sessions = pd.DatetimeIndex(pd.to_datetime(plan.signal_sessions, utc=True)).normalize()
    print("phase=features start", flush=True)
    features, industry_audit = _build_features(bundle, strategy, signal_sessions, output)
    features.to_parquet(output / "ledgers/feature_ledger.parquet", index=False)
    events = _build_event_ledger(features[features["asof"].isin(signal_sessions)].copy(), strategy)
    events.to_parquet(output / "ledgers/event_ledger.parquet", index=False)
    candidates = events[events["candidate_status"].eq("IN_VIEW")].copy()
    selection = _build_selection_ledger(events, strategy, signal_sessions)
    events.to_parquet(output / "ledgers/score_ledger.parquet", index=False)
    candidates.to_parquet(output / "ledgers/candidate_ledger.parquet", index=False)
    selection.to_parquet(output / "ledgers/selection_ledger.parquet", index=False)
    selection_summary = {
        "signal_dates": int(len(signal_sessions)),
        "event_rows": int(len(events)),
        "event_pass_rows": int(events["event_pass"].sum()),
        "event_pass_days": int(events.loc[events["event_pass"], "asof"].nunique()),
        "candidate_view_rows": int(len(candidates)),
        "rejection_reason_counts": {str(k): int(v) for k, v in events.loc[events["event_status"].eq("REJECTED"), "rejection_reasons"].str.split(";").explode().replace("", np.nan).dropna().value_counts().sort_index().items()},
        "industry_resolution": industry_audit,
    }
    _write_json(output / "diagnostics/selection_summary.json", selection_summary)
    # The engine's participation cap is based on adv20_amount.  MRT explicitly
    # uses the prior-20-session median, so pass a local causal override.
    execution_for_backtest = bundle.execution.copy()
    execution_for_backtest["trade_date"] = pd.to_datetime(execution_for_backtest["trade_date"], utc=True).dt.normalize()
    execution_for_backtest = execution_for_backtest.sort_values(["ts_code", "trade_date"], kind="mergesort")
    execution_for_backtest["adv20_amount"] = execution_for_backtest.groupby("ts_code", sort=False)["amount"].transform(lambda s: s.shift(1).rolling(20, min_periods=20).median())
    execution_for_backtest = execution_for_backtest.sort_values(["trade_date", "ts_code"], kind="mergesort")
    execution_for_backtest.to_parquet(output / "ledgers/execution_panel_mrt_prior_adv20.parquet", index=False)
    portfolios: dict[str, Any] = {}
    execution_event_ledgers: dict[str, pd.DataFrame] = {}
    for scenario in ("base", "stress"):
        print(f"phase=backtest scenario={scenario} start", flush=True)
        result = run_v3_backtest(
            candidate_ledger=candidates,
            selection_ledger=selection,
            execution_panel=execution_for_backtest,
            corporate_actions=bundle.corporate_actions,
            strategy=strategy,
            execution_sessions=plan.execution_sessions,
            scenario_name=scenario,
        )
        target = output / "backtests" / scenario
        target.mkdir()
        for name, frame in result.items():
            frame.to_parquet(target / f"{name}.parquet", index=False)
        metrics = summarize_v3(result["nav"], result["fills"], initial_cash=float(strategy.execution["initial_cash"]))
        metrics.update({"scenario": scenario, "entry_attempts": int(len(result["execution_decisions"])), "entry_fills": int(result["execution_decisions"]["chosen"].sum()) if not result["execution_decisions"].empty else 0, "open_positions_at_end": int(len(result["open_positions"]))})
        metrics["acceptance"] = _acceptance(metrics, strategy)
        _write_json(target / "metrics.json", metrics)
        execution_events = _attach_execution_audit(events, selection, result, plan.execution_sessions)
        execution_events.to_parquet(target / "event_ledger.parquet", index=False)
        execution_event_ledgers[scenario] = execution_events
        portfolios[scenario] = metrics
        print(f"phase=backtest scenario={scenario} return={metrics['total_return']:+.6f} pf={metrics['portfolio_profit_factor']} maxdd={metrics['max_drawdown']:+.6f}", flush=True)
    summary = {"strategy": strategy.strategy_id, "bundle_id": bundle.bundle_id, "selection": selection_summary, "portfolios": portfolios, "parameter_sweep": False, "decision": "ADVANCE" if all(item["acceptance"]["passed"] for item in portfolios.values()) else "ABANDON_DIAGNOSTIC_ONLY"}
    _write_json(output / "SUMMARY.json", summary)
    # The top-level event ledger is the base-cost audit view; stress has its
    # own copy under backtests/stress/event_ledger.parquet.
    execution_event_ledgers["base"].to_parquet(output / "ledgers/event_ledger.parquet", index=False)
    _write_json(output / "manifests/config_manifest.json", {"strategy_hash": strategy.config_hash, "model_hash": model.config_hash, "plan_hash": _sha(output / "plan.json"), "parameter_sweep": False})
    code_paths = [
        Path(__file__).resolve(), ROOT / "configs/strategy/mrt_v1.yaml",
        ROOT / "src/aistock9988/backtest/v3_engine.py",
        ROOT / "src/aistock9988/features/engine.py",
        ROOT / "src/aistock9988/features/mrt.py",
        ROOT / "src/aistock9988/data/bundle.py",
        ROOT / "src/aistock9988/selection/pipeline.py",
    ]
    _write_json(output / "manifests/code_manifest.json", {str(path.relative_to(ROOT)): _sha(path) for path in code_paths})
    lines = [f"# MRT-V1 frozen state-sequence diagnostic", "", f"- Signal range: `{plan.signal_start}` to `{plan.signal_end}`; execution through `{plan.execution_end}`.", f"- Event rows: `{len(events)}`; pass rows: `{int(events['event_pass'].sum())}`; pass days: `{events.loc[events['event_pass'], 'asof'].nunique()}`.", "- Rules: market/industry excess ret10 > 0, ret60 > 0, ret20 <= 25%, 0 < dist_ma60 <= 15%, vol20 <= cross-sectional P85; same-day raw down shock, >=1.5x prior ADV20, above down-limit, non-ST/non-new.", "- T+1 raw open, Top20 by amount/ADV20-prior, Top5, 10% each, max 5 positions, H10, -8% close/next-open stop.", "", "| Cost | Return | PF | MaxDD | Ex-best-week | Trades | Pass |", "|---|---:|---:|---:|---:|---:|---|"]
    for name in ("base", "stress"):
        item = portfolios[name]
        pf = "NA" if item["portfolio_profit_factor"] is None else f"{item['portfolio_profit_factor']:.3f}"
        lines.append(f"| {name} | {item['total_return']:+.2%} | {pf} | {item['max_drawdown']:.2%} | {item['return_excluding_best_week']:+.2%} | {item['trade_count']} | {item['acceptance']['passed']} |")
    lines.extend(["", "Decision: `ADVANCE` only if both base and stress satisfy PF>=2, MaxDD<=15%, and return excluding best week > 0. No parameter sweep."])
    (output / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(output / "RUN_STATUS.json", {
        "status": "DIAGNOSTIC_COMPLETED", "bundle_id": bundle.bundle_id,
        "overall_acceptance_passed": summary["decision"] == "ADVANCE",
        "diagnostic_history": True,
        "strategy_research_status": strategy.identity.get("research_status", "historical"),
    }, replace=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=Path, default=DEFAULT_STRATEGY)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--signal-start", default="2026-01-01")
    parser.add_argument("--signal-end", default="2026-08-13")
    parser.add_argument("--execution-end", default="2026-08-28")
    parser.add_argument("--run-name", default="MRT_V1_DIAGNOSTIC_2026_TO_0828")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()
