"""Freeze and settle one date in the append-only forward lockbox."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from aistock9988.backtest.v3_engine import run_v3_backtest
from aistock9988.configuration import ModelConfig, StrategyConfig
from aistock9988.data.bundle import build_data_bundle, load_trading_calendar
from aistock9988.features.engine import build_feature_ledger
from aistock9988.forward.lockbox import ForwardLockbox
from aistock9988.planning import RunRequest, compile_run_plan
from aistock9988.reporting.v3_metrics import summarize_v3
from aistock9988.time.session import session_close
from aistock9988.selection.pipeline import build_rule_ledgers

ROOT = Path(__file__).resolve().parents[1]


def _enforce_forward_only_contract(strategy: StrategyConfig, asof: pd.Timestamp, mode: str) -> None:
    """Enforce forward-only dates at the shared runner boundary, not just wrappers."""
    status = str(strategy.identity.get("research_status", "historical"))
    if status == "abandoned":
        raise ValueError(
            f"strategy {strategy.strategy_id} is abandoned; refusing {mode} lockbox operation"
        )
    if status != "forward_only":
        return
    forward_start = pd.Timestamp(strategy.identity["forward_start"])
    forward_start = (
        forward_start.tz_localize("UTC")
        if forward_start.tzinfo is None
        else forward_start.tz_convert("UTC")
    ).normalize()
    if asof < forward_start:
        raise ValueError(
            f"forward-only strategy rejects historical asof {asof.date()}; "
            f"first permitted session is {forward_start.date()}"
        )
    now = pd.Timestamp.now(tz="Asia/Shanghai")
    today = now.normalize()
    asof_local = asof.tz_convert("Asia/Shanghai").normalize()
    if asof_local > today:
        raise ValueError(
            f"forward-only strategy rejects future asof {asof.date()}; local date is {today.date()}"
        )
    if mode == "freeze" and asof_local == today:
        close = session_close(asof).tz_convert("Asia/Shanghai")
        if now < close:
            raise ValueError(
                f"freeze requires the completed session close at {close.isoformat()}; now is {now.isoformat()}"
            )


def _code_manifest(strategy_path: Path, model_path: Path) -> dict[str, object]:
    """Hash the executable closure used to create or settle a forward batch."""
    paths = list(sorted((ROOT / "src/aistock9988").rglob("*.py"))) + [
        Path(__file__).resolve(), strategy_path.resolve(), model_path.resolve()
    ]
    entrypoint = Path(sys.argv[0]).resolve()
    if entrypoint.is_file():
        paths.append(entrypoint)
    files: dict[str, str] = {}
    for path in dict.fromkeys(paths):
        if not path.is_file():
            raise FileNotFoundError(f"code manifest path is missing: {path}")
        try:
            name = str(path.relative_to(ROOT))
        except ValueError:
            name = str(path)
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload: dict[str, object] = {"schema_version": "code-manifest-v1", "files": files}
    payload["manifest_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def _code_manifest_hash(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.get("manifest_sha256")
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    actual = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if expected != actual:
        raise ValueError(f"code manifest self-hash mismatch: {path}")
    return str(expected)


def _verify_code_manifest(path: Path, *, expected_hash: str | None = None) -> str:
    """Verify a frozen closure against the current filesystem without rebuilding it."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    actual_manifest_hash = _code_manifest_hash(path)
    if expected_hash is not None and actual_manifest_hash != str(expected_hash):
        raise ValueError(f"code manifest hash differs from lockbox metadata: {path}")
    files = payload.get("files", {})
    if not isinstance(files, dict) or not files:
        raise ValueError(f"code manifest has no files: {path}")
    for name, expected in files.items():
        candidate = Path(str(name))
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        if not candidate.is_file() or hashlib.sha256(candidate.read_bytes()).hexdigest() != str(expected):
            raise ValueError(f"forward code closure drift: {candidate}")
    return actual_manifest_hash


def _plan(strategy: StrategyConfig, model: ModelConfig, asof: pd.Timestamp, execution_end: pd.Timestamp, output: Path):
    calendar = load_trading_calendar(
        str((asof - pd.Timedelta(days=500)).date()), str(execution_end.date())
    )
    request = RunRequest(
        signal_start=str(asof.date()), signal_end=str(asof.date()),
        execution_end=str(execution_end.date()), output_dir=str(output),
        run_name=f"{strategy.strategy_id}-{asof.strftime('%Y%m%d')}",
    )
    return compile_run_plan(strategy, model, request, calendar["session"])


def _write_waiting(path: Path, asof: pd.Timestamp, reason: str, bundle_id: str) -> str:
    path.mkdir(parents=True, exist_ok=True)
    (path / "WAITING_FOR_DATA.json").write_text(
        json.dumps({
            "asof": str(asof.date()), "status": "WAITING_FOR_DATA",
            "reason": reason, "bundle_id": bundle_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2) + chr(10), encoding="utf-8"
    )
    return "WAITING_FOR_DATA"


def freeze(args: argparse.Namespace, strategy: StrategyConfig, model: ModelConfig, asof: pd.Timestamp, execution_end: pd.Timestamp) -> str:
    root = args.output.resolve()
    formal = root / "batches" / asof.strftime("%Y-%m-%d")
    lockbox = ForwardLockbox(root, experiment_id=strategy.strategy_id, config_sha256=strategy.config_hash)
    if formal.exists():
        # Recover a process that moved the batch before committing its manifest.
        try:
            lockbox.read_day(asof)
            return str(formal)
        except FileNotFoundError:
            required = {"score_ledger.parquet", "candidate_ledger.parquet", "selection_ledger.parquet"}
            if not required.issubset({path.name for path in formal.glob("*.parquet")}):
                raise FileExistsError(f"immutable forward batch exists: {formal}")
            stored_strategy = StrategyConfig.from_yaml(formal / "strategy.yaml")
            if stored_strategy.config_hash != strategy.config_hash:
                raise ValueError("formal batch strategy differs from recovery strategy")
            code_manifest_path = formal / "code_manifest.json"
            code_hash = _code_manifest_hash(code_manifest_path) if code_manifest_path.exists() else None
            lockbox.append(
                {"score": pd.read_parquet(formal / "score_ledger.parquet"),
                 "candidate": pd.read_parquet(formal / "candidate_ledger.parquet"),
                 "selection": pd.read_parquet(formal / "selection_ledger.parquet")},
                bundle_id=json.loads((formal / "data_manifest.json").read_text(encoding="utf-8"))["bundle_id"],
                source_end=str(execution_end.date()),
                metadata={
                    "research_status": str(strategy.identity.get("research_status", "historical")),
                    **({"code_manifest_sha256": code_hash} if code_hash else {}),
                },
            )
            return str(formal)
    attempt = root / "pending" / asof.strftime("%Y-%m-%d") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    attempt.mkdir(parents=True, exist_ok=False)
    plan = _plan(strategy, model, asof, execution_end, attempt)
    bundle = build_data_bundle(plan, strategy, attempt)
    day = bundle.execution["trade_date"].eq(asof)
    universe = bundle.execution.loc[day & bundle.execution["universe_pass"].astype(bool)]
    eligible = universe["selection_data_eligible"].astype(bool)
    coverage = float(eligible.mean()) if len(universe) else 0.0
    min_coverage = float(strategy.data_policy.get("forward_min_coverage", 0.90))
    if len(universe) == 0 or coverage < min_coverage:
        return _write_waiting(attempt, asof, f"selection_coverage_{coverage:.4f}_below_{min_coverage:.4f}", bundle.bundle_id)

    features = build_feature_ledger(bundle, strategy)
    ledgers = build_rule_ledgers(features, strategy, plan.signal_sessions)
    candidate_count = int(ledgers["candidate"]["candidate_status"].eq("IN_VIEW").sum())
    view_size = int(strategy.portfolio["candidate_view_size"])
    if candidate_count < view_size:
        return _write_waiting(attempt, asof, f"candidate_view_{candidate_count}_below_{view_size}", bundle.bundle_id)
    features.to_parquet(attempt / "feature_ledger.parquet", index=False)
    for name, frame in ledgers.items():
        frame.to_parquet(attempt / f"{name}_ledger.parquet", index=False)
    (attempt / "plan.json").write_text(json.dumps(plan.to_dict(), indent=2, default=str) + chr(10), encoding="utf-8")
    (attempt / "data_manifest.json").write_text(json.dumps(bundle.manifest, indent=2, default=str) + chr(10), encoding="utf-8")
    shutil.copyfile(args.strategy, attempt / "strategy.yaml")
    shutil.copyfile(args.model, attempt / "model.yaml")
    code_manifest = _code_manifest(Path(args.strategy), Path(args.model))
    (attempt / "code_manifest.json").write_text(
        json.dumps(code_manifest, indent=2, sort_keys=True) + chr(10), encoding="utf-8"
    )
    formal.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(attempt), str(formal))
    lockbox.append(
        {"score": ledgers["score"], "candidate": ledgers["candidate"], "selection": ledgers["selection"]},
        bundle_id=bundle.bundle_id,
        source_end=str(execution_end.date()),
        metadata={
            "research_status": str(strategy.identity.get("research_status", "historical")),
            "code_manifest_sha256": str(code_manifest["manifest_sha256"]),
        },
    )
    return str(formal)


def settle(args: argparse.Namespace, strategy: StrategyConfig, model: ModelConfig, asof: pd.Timestamp, execution_end: pd.Timestamp) -> str:
    root = args.output.resolve()
    batch = root / "batches" / asof.strftime("%Y-%m-%d")
    if not batch.exists():
        raise FileNotFoundError(f"no frozen batch for {asof.date()}: run freeze first")
    target = root / "settlements" / asof.strftime("%Y-%m-%d")
    if target.exists():
        raise FileExistsError(f"immutable settlement exists: {target}")
    lockbox = ForwardLockbox(root, experiment_id=strategy.strategy_id, config_sha256=strategy.config_hash)
    manifest = lockbox.manifest_for_day(asof)
    expected_code_hash = manifest.get("code_manifest_sha256")
    if not expected_code_hash:
        # Legacy batches were frozen before code-closure metadata was
        # mandatory.  Never turn them into a formal OOS return by silently
        # skipping the closure check; preserve them as diagnostic-only data.
        raise ValueError(
            "forward batch is legacy and has no code_manifest_sha256; "
            "formal settlement is refused"
        )
    current_code_hash = _code_manifest(Path(args.strategy), Path(args.model))["manifest_sha256"]
    if str(current_code_hash) != str(expected_code_hash):
        raise ValueError("forward settlement code closure differs from freeze")
    if str(manifest.get("source_end")) != str(execution_end.date()):
        raise ValueError("settle execution_end differs from freeze source_end")
    committed = lockbox.read_day(asof)
    candidate = committed["candidate"]
    selection = committed["selection"]
    attempt = root / "settlements" / "pending" / asof.strftime("%Y-%m-%d") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    attempt.mkdir(parents=True, exist_ok=False)
    plan = _plan(strategy, model, asof, execution_end, attempt)
    bundle = build_data_bundle(plan, strategy, attempt)
    selected_codes = sorted(candidate.loc[candidate["candidate_status"].eq("IN_VIEW"), "ts_code"].astype(str).unique())
    # RunPlan stores ISO date strings; normalize them to the same UTC index
    # used by the as-of timestamp and execution panel.
    sessions = pd.DatetimeIndex(pd.to_datetime(plan.execution_sessions, utc=True)).normalize()
    start_idx = sessions.get_indexer([asof])[0]
    if start_idx < 0:
        raise ValueError("asof is not in execution calendar")
    needed = sessions[start_idx + 1 : start_idx + 1 + int(strategy.execution["hold_sessions_from_fill"] + 1)]
    panel = bundle.execution[bundle.execution["ts_code"].isin(selected_codes) & bundle.execution["trade_date"].isin(needed)]
    if len(needed) == 0 or panel.empty or not panel["execution_data_eligible"].astype(bool).all():
        return _write_waiting(attempt, asof, "execution_horizon_not_complete", bundle.bundle_id)
    results = {}
    for scenario in ("base", "stress"):
        result = run_v3_backtest(
            candidate_ledger=candidate,
            selection_ledger=selection,
            execution_panel=bundle.execution,
            corporate_actions=bundle.corporate_actions,
            strategy=strategy,
            execution_sessions=plan.execution_sessions,
            scenario_name=scenario,
        )
        for name, frame in result.items():
            frame.to_parquet(attempt / scenario / f"{name}.parquet", index=False)
        metrics = summarize_v3(result["nav"], result["fills"], initial_cash=float(strategy.execution["initial_cash"]))
        metrics["scenario"] = scenario
        metrics["single_day_diagnostic_only"] = True
        metrics["acceptance"] = {
            "pf_ge_2": bool(metrics["portfolio_profit_factor"] is not None and float(metrics["portfolio_profit_factor"]) >= float(strategy.acceptance["portfolio_profit_factor_min"])),
            "maxdd_le_15": abs(float(metrics["max_drawdown"])) <= float(strategy.acceptance["max_drawdown_abs_max"]),
            "ex_best_week_positive": float(metrics["return_excluding_best_week"]) > 0.0,
            "ex_top3_profit_positive": float(metrics["return_excluding_top3_profit"]) > 0.0,
            "trade_win_rate_ge_70": bool(metrics["trade_win_rate"] is not None and float(metrics["trade_win_rate"]) >= 0.70),
        }
        metrics["acceptance"]["passed"] = bool(all(metrics["acceptance"].values()))
        (attempt / scenario / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + chr(10), encoding="utf-8")
        results[scenario] = metrics
    (attempt / "SETTLEMENT.json").write_text(json.dumps({"asof": str(asof.date()), "bundle_id": bundle.bundle_id, "results": results}, indent=2, default=str) + chr(10), encoding="utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(attempt), str(target))
    return str(target)


def run(args: argparse.Namespace) -> str:
    strategy = StrategyConfig.from_yaml(args.strategy)
    model = ModelConfig.from_yaml(args.model)
    asof = pd.Timestamp(args.asof, tz="UTC").normalize()
    execution_end = pd.Timestamp(args.execution_end, tz="UTC").normalize()
    if execution_end <= asof:
        raise ValueError("execution_end must be after asof")
    _enforce_forward_only_contract(strategy, asof, str(args.mode))
    return freeze(args, strategy, model, asof, execution_end) if args.mode == "freeze" else settle(args, strategy, model, asof, execution_end)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("freeze", "settle"), default="freeze")
    parser.add_argument("--asof", required=True)
    parser.add_argument("--execution-end", required=True)
    parser.add_argument("--strategy", type=Path, default=ROOT / "configs/strategy/quiet_confirmed_v1.yaml")
    parser.add_argument("--model", type=Path, default=ROOT / "configs/model/disabled.yaml")
    # Keep formal forward batches isolated from legacy and diagnostic roots.
    parser.add_argument("--output", type=Path, default=ROOT / "docs/council_20260828/S49_QUIET_FORWARD_LOCKBOX_FORMAL")
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()
