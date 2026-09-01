"""Freeze and settle one date in the append-only forward lockbox."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from aistock9988.backtest.engine import run_backtest
from aistock9988.configuration import StrategyConfig
from aistock9988.data.bundle import (
    build_data_bundle,
    load_source_max_dates,
    load_trading_calendar,
)
from aistock9988.features.engine import build_feature_ledger
from aistock9988.forward.early_path import (
    EarlyPathConfig,
    EarlyPathFailure,
    apply_early_path_overlay,
)
from aistock9988.forward.lockbox import ForwardLockbox
from aistock9988.planning import RunRequest, compile_run_plan
from aistock9988.reporting.metrics import summarize
from aistock9988.time.session import session_close
from aistock9988.selection.pipeline import build_rule_ledgers

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
SCRIPTS_ROOT = ROOT / "scripts"


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


def _code_manifest(
    strategy_path: Path,
    extra_paths: tuple[Path, ...] = (),
) -> dict[str, object]:
    """Hash the executable closure used to create or settle a forward batch."""
    seeds = {
        Path(__file__).resolve(),
        strategy_path.resolve(),
        *(path.resolve() for path in extra_paths),
    }
    paths = _local_import_closure(seeds)
    files: dict[str, str] = {}
    for path in sorted(paths):
        if not path.is_file():
            raise FileNotFoundError(f"code manifest path is missing: {path}")
        try:
            name = str(path.relative_to(ROOT))
        except ValueError:
            name = str(path)
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload: dict[str, object] = {"schema_version": "code-manifest-v2", "files": files}
    payload["manifest_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def _local_import_closure(seeds: set[Path]) -> set[Path]:
    """Resolve local Python imports without sweeping unrelated source files."""
    closure = {path.resolve() for path in seeds}
    pending = [path for path in closure if path.suffix == ".py"]
    parsed: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in parsed:
            continue
        parsed.add(path)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as exc:
            raise ValueError(f"cannot parse forward dependency: {path}") from exc
        for module in _imported_modules(tree, path):
            target = _resolve_local_module(module)
            if target is None:
                continue
            for dependency in _package_chain(target):
                if dependency not in closure:
                    closure.add(dependency)
                    if dependency.suffix == ".py":
                        pending.append(dependency)
    return closure


def _imported_modules(tree: ast.AST, source: Path) -> set[str]:
    modules: set[str] = set()
    current = _module_name(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        base = _absolute_import_name(current, source, node.module, node.level)
        if base:
            modules.add(base)
            modules.update(f"{base}.{alias.name}" for alias in node.names if alias.name != "*")
    return modules


def _module_name(path: Path) -> str:
    path = path.resolve()
    if path.is_relative_to(SRC_ROOT):
        relative = path.relative_to(SRC_ROOT).with_suffix("")
    elif path.is_relative_to(SCRIPTS_ROOT):
        relative = path.relative_to(SCRIPTS_ROOT).with_suffix("")
    else:
        return ""
    parts = list(relative.parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _absolute_import_name(
    current: str,
    source: Path,
    imported: str | None,
    level: int,
) -> str:
    if level == 0:
        return imported or ""
    if not current or not source.is_relative_to(SRC_ROOT):
        return ""
    package = current.split(".")
    if source.name != "__init__.py":
        package = package[:-1]
    trim = level - 1
    if trim > len(package):
        return ""
    prefix = package[: len(package) - trim]
    suffix = imported.split(".") if imported else []
    return ".".join(prefix + suffix)


def _resolve_local_module(module: str) -> Path | None:
    if not module:
        return None
    parts = module.split(".")
    roots = (SRC_ROOT,) if parts[0] == "aistock9988" else (SCRIPTS_ROOT,)
    for root in roots:
        file_path = root.joinpath(*parts).with_suffix(".py")
        if file_path.is_file():
            return file_path.resolve()
        init_path = root.joinpath(*parts, "__init__.py")
        if init_path.is_file():
            return init_path.resolve()
    return None


def _package_chain(path: Path) -> set[Path]:
    chain = {path.resolve()}
    if not path.is_relative_to(SRC_ROOT):
        return chain
    parent = path.parent
    while parent != SRC_ROOT:
        init_path = parent / "__init__.py"
        if init_path.is_file():
            chain.add(init_path.resolve())
        parent = parent.parent
    return chain


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


def _verify_overlay_registration(
    args: argparse.Namespace,
    strategy: StrategyConfig,
    overlay: EarlyPathConfig,
) -> dict[str, str]:
    """Require the pre-signal registration to match the current executable closure."""
    path = Path(args.overlay_registration).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"early-path registration is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_manifest = payload.get("code_manifest")
    if not isinstance(expected_manifest, dict):
        raise ValueError(f"early-path registration has no code manifest: {path}")
    expected_hash = expected_manifest.get("manifest_sha256")
    expected_body = {
        key: value for key, value in expected_manifest.items() if key != "manifest_sha256"
    }
    actual_expected_hash = hashlib.sha256(
        json.dumps(expected_body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if expected_hash != actual_expected_hash:
        raise ValueError(f"early-path registration code manifest self-hash mismatch: {path}")

    current_manifest = _code_manifest(Path(args.strategy), _overlay_closure_paths(args))
    checks = {
        "registration_schema_version": payload.get("registration_schema_version")
        == "cap1-early-path-registration-v2",
        "status": payload.get("status") == "PREREGISTERED_NOT_STARTED",
        "historical_backtest_executed": payload.get("historical_backtest_executed") is False,
        "control_strategy_id": payload.get("control_strategy_id") == strategy.strategy_id,
        "control_strategy_config_sha256": payload.get("control_strategy_config_sha256")
        == strategy.config_hash,
        "overlay_id": payload.get("overlay_id") == str(overlay.identity["overlay_id"]),
        "overlay_config_sha256": payload.get("overlay_config_sha256") == overlay.config_hash,
        "overlay_prereg_sha256": payload.get("overlay_prereg_sha256")
        == hashlib.sha256(Path(args.overlay_prereg).read_bytes()).hexdigest(),
        "code_manifest_sha256": expected_hash == current_manifest["manifest_sha256"],
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"early-path registration drift ({', '.join(failed)}): {path}")
    return {
        "registration_path": str(path),
        "registration_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "code_manifest_sha256": str(current_manifest["manifest_sha256"]),
    }


def _plan(
    strategy: StrategyConfig,
    asof: pd.Timestamp,
    execution_end: pd.Timestamp,
    output: Path,
    *,
    require_complete_horizon: bool,
):
    calendar = load_trading_calendar(
        str((asof - pd.Timedelta(days=500)).date()), str(execution_end.date())
    )
    request = RunRequest(
        signal_start=str(asof.date()), signal_end=str(asof.date()),
        execution_end=str(execution_end.date()), output_dir=str(output),
        run_name=f"{strategy.strategy_id}-{asof.strftime('%Y%m%d')}",
    )
    return compile_run_plan(
        strategy,
        request,
        calendar["session"],
        require_complete_horizon=require_complete_horizon,
    )


def _cutoff_status(
    strategy: StrategyConfig,
    cutoff: pd.Timestamp,
    *,
    stage: str,
) -> tuple[dict[str, str], dict[str, str]]:
    sources = set(str(value) for value in strategy.data_policy["dense_required"][stage])
    if stage == "selection":
        sources.update(str(value) for value in strategy.data_policy.get("optional_enrichment", ()))
    cutoffs = load_source_max_dates(sources)
    stale = {
        name: value
        for name, value in cutoffs.items()
        if pd.Timestamp(value).date() < cutoff.date()
    }
    return cutoffs, stale


def _write_waiting(
    path: Path,
    asof: pd.Timestamp,
    reason: str,
    bundle_id: str | None,
    details: dict[str, object] | None = None,
) -> str:
    path.mkdir(parents=True, exist_ok=True)
    (path / "WAITING_FOR_DATA.json").write_text(
        json.dumps({
            "asof": str(asof.date()), "status": "WAITING_FOR_DATA",
            "reason": reason, "bundle_id": bundle_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **(details or {}),
        }, indent=2) + chr(10), encoding="utf-8"
    )
    return "WAITING_FOR_DATA"


def freeze(args: argparse.Namespace, strategy: StrategyConfig, asof: pd.Timestamp) -> str:
    overlay = EarlyPathConfig.from_yaml(args.overlay_config)
    overlay.validate_control(strategy)
    registration_info = _verify_overlay_registration(args, strategy, overlay)
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
                freeze_data_cutoff=str(asof.date()),
                metadata={
                    "research_status": str(strategy.identity.get("research_status", "historical")),
                    **({"code_manifest_sha256": code_hash} if code_hash else {}),
                },
            )
            return str(formal)
    attempt = root / "pending" / asof.strftime("%Y-%m-%d") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    attempt.mkdir(parents=True, exist_ok=False)
    selection_cutoffs, stale = _cutoff_status(strategy, asof, stage="selection")
    if stale:
        return _write_waiting(
            attempt,
            asof,
            "selection_cutoff_after_database_cutoff",
            None,
            {"requested_cutoff": str(asof.date()), "source_cutoffs": selection_cutoffs},
        )
    plan = _plan(
        strategy,
        asof,
        asof,
        attempt,
        require_complete_horizon=False,
    )
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
    allow_sparse = bool(strategy.portfolio.get("allow_sparse_candidate_view", False))
    min_ratio = float(strategy.portfolio.get("candidate_min_daily_view_ratio", 1.0))
    minimum_candidates = int(np.ceil(view_size * min_ratio))
    if not allow_sparse and candidate_count < minimum_candidates:
        return _write_waiting(
            attempt,
            asof,
            f"candidate_view_{candidate_count}_below_{minimum_candidates}",
            bundle.bundle_id,
        )
    features.to_parquet(attempt / "feature_ledger.parquet", index=False)
    for name, frame in ledgers.items():
        frame.to_parquet(attempt / f"{name}_ledger.parquet", index=False)
    (attempt / "plan.json").write_text(json.dumps(plan.to_dict(), indent=2, default=str) + chr(10), encoding="utf-8")
    (attempt / "data_manifest.json").write_text(json.dumps(bundle.manifest, indent=2, default=str) + chr(10), encoding="utf-8")
    shutil.copyfile(args.strategy, attempt / "strategy.yaml")
    shutil.copyfile(args.overlay_config, attempt / "early_path_overlay.yaml")
    shutil.copyfile(args.overlay_prereg, attempt / "early_path_preregistration.md")
    shutil.copyfile(args.overlay_registration, attempt / "early_path_registration.json")
    registration_info = _verify_overlay_registration(args, strategy, overlay)
    code_manifest = _code_manifest(Path(args.strategy), _overlay_closure_paths(args))
    (attempt / "code_manifest.json").write_text(
        json.dumps(code_manifest, indent=2, sort_keys=True) + chr(10), encoding="utf-8"
    )
    formal.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(attempt), str(formal))
    lockbox.append(
        {"score": ledgers["score"], "candidate": ledgers["candidate"], "selection": ledgers["selection"]},
        bundle_id=bundle.bundle_id,
        freeze_data_cutoff=str(asof.date()),
        metadata={
            "research_status": str(strategy.identity.get("research_status", "historical")),
            "forward_start": str(strategy.identity.get("forward_start", "")),
            "code_manifest_sha256": str(code_manifest["manifest_sha256"]),
            "overlay_config_sha256": overlay.config_hash,
            "overlay_prereg_sha256": hashlib.sha256(args.overlay_prereg.read_bytes()).hexdigest(),
            "overlay_registration_sha256": str(
                registration_info["registration_sha256"]
            ),
            "candidate_count": candidate_count,
            "abstention": candidate_count == 0,
        },
    )
    return str(formal)


def settle(args: argparse.Namespace, strategy: StrategyConfig, asof: pd.Timestamp, execution_end: pd.Timestamp) -> str:
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
        raise ValueError(
            "forward batch is unsealed and has no code_manifest_sha256; "
            "formal settlement is refused"
        )
    current_code_hash = _code_manifest(
        Path(args.strategy), _overlay_closure_paths(args)
    )["manifest_sha256"]
    if str(current_code_hash) != str(expected_code_hash):
        raise ValueError("forward settlement code closure differs from freeze")
    if str(manifest.get("freeze_data_cutoff")) != str(asof.date()):
        raise ValueError("forward batch freeze_data_cutoff differs from signal asof")
    committed = lockbox.read_day(asof)
    candidate = committed["candidate"]
    selection = committed["selection"]
    attempt = root / "settlements" / "pending" / asof.strftime("%Y-%m-%d") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    attempt.mkdir(parents=True, exist_ok=False)
    source_cutoffs, stale = _cutoff_status(
        strategy, execution_end, stage="execution"
    )
    if stale:
        return _write_waiting(
            attempt,
            asof,
            "execution_end_after_database_cutoff",
            None,
            {
                "requested_execution_end": str(execution_end.date()),
                "source_cutoffs": source_cutoffs,
            },
        )
    plan = _plan(
        strategy,
        asof,
        execution_end,
        attempt,
        require_complete_horizon=False,
    )
    bundle = build_data_bundle(plan, strategy, attempt)
    overlay = EarlyPathConfig.from_yaml(args.overlay_config)
    overlay.validate_control(strategy)
    results: dict[str, dict[str, object]] = {"control": {}, "shadow": {}}
    scenario_outputs: dict[str, dict[str, dict[str, pd.DataFrame]]] = {
        "control": {}, "shadow": {},
    }
    open_positions: dict[str, int] = {}
    for scenario in ("base", "stress"):
        result = run_backtest(
            candidate_ledger=candidate,
            selection_ledger=selection,
            execution_panel=bundle.execution,
            corporate_actions=bundle.corporate_actions,
            strategy=strategy,
            execution_sessions=plan.execution_sessions,
            scenario_name=scenario,
        )
        scenario_outputs["control"][scenario] = result
        try:
            scenario_outputs["shadow"][scenario] = apply_early_path_overlay(
                control_result=result,
                execution_panel=bundle.execution,
                execution_sessions=plan.execution_sessions,
                control_strategy=strategy,
                overlay=overlay,
                scenario_name=scenario,
            )
        except EarlyPathFailure as exc:
            return _seal_overlay_failure(
                attempt=attempt,
                target=target,
                args=args,
                asof=asof,
                execution_end=execution_end,
                scenario=scenario,
                failure=exc,
                bundle_id=bundle.bundle_id,
                source_cutoffs=source_cutoffs,
            )
        final_open = int(result["nav"].sort_values("trade_date").iloc[-1]["open_positions"])
        open_positions[scenario] = final_open
    if any(value > 0 for value in open_positions.values()):
        return _write_waiting(
            attempt,
            asof,
            "actual_positions_still_open",
            bundle.bundle_id,
        )

    for arm, arm_outputs in scenario_outputs.items():
        for scenario, result in arm_outputs.items():
            scenario_dir = attempt / arm / scenario
            scenario_dir.mkdir(parents=True, exist_ok=False)
            for name, frame in result.items():
                frame.to_parquet(scenario_dir / f"{name}.parquet", index=False)
            metrics = summarize(
                result["nav"], result["fills"],
                initial_cash=float(strategy.execution["initial_cash"]),
                positions=result["positions"],
                corporate_actions=result["corporate_actions"],
            )
            metrics.update({
                "arm": arm,
                "scenario": scenario,
                "single_day_diagnostic_only": True,
                "sample_ready": False,
                "objectives": {
                    "weekly_return_target": 0.05,
                    "trade_win_rate_target": 0.70,
                    "trade_win_rate_target_met": bool(
                        metrics["trade_win_rate"] is not None
                        and float(metrics["trade_win_rate"]) >= 0.70
                    ),
                },
                "acceptance": {"passed": None},
            })
            if arm == "shadow":
                metrics["paired_right_tail"] = _right_tail_recall(
                    arm_outputs=scenario_outputs,
                    scenario=scenario,
                )
                path = result["path_events"]
                terminals = path.drop_duplicates("trade_key", keep="last") if not path.empty else path
                metrics["path_state_counts"] = (
                    terminals["resulting_state"].value_counts().sort_index().to_dict()
                    if not terminals.empty else {}
                )
                metrics["effective_early_exit_count"] = int(len(result["paired_capital"]))
            (scenario_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2, default=str) + chr(10), encoding="utf-8"
            )
            results[arm][scenario] = metrics
    (attempt / "SETTLEMENT.json").write_text(json.dumps({
        "asof": str(asof.date()),
        "freeze_data_cutoff": str(manifest["freeze_data_cutoff"]),
        "settlement_data_cutoff": str(execution_end.date()),
        "source_cutoffs": source_cutoffs,
        "bundle_id": bundle.bundle_id,
        "actual_positions_closed": True,
        "overlay_config_sha256": overlay.config_hash,
        "results": results,
    }, indent=2, default=str) + chr(10), encoding="utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(attempt), str(target))
    return str(target)


def run(args: argparse.Namespace) -> str:
    strategy = StrategyConfig.from_yaml(args.strategy)
    overlay = EarlyPathConfig.from_yaml(args.overlay_config)
    overlay.validate_control(strategy)
    if not args.overlay_prereg.is_file():
        raise FileNotFoundError(f"early-path preregistration is missing: {args.overlay_prereg}")
    asof = pd.Timestamp(args.asof, tz="UTC").normalize()
    _enforce_forward_only_contract(strategy, asof, str(args.mode))
    if args.mode == "freeze":
        if args.execution_end is not None and pd.Timestamp(args.execution_end).date() != asof.date():
            raise ValueError("freeze uses only asof-close data; omit --execution-end or set it equal to asof")
        return freeze(args, strategy, asof)
    if args.execution_end is None:
        raise ValueError("settle requires --execution-end")
    execution_end = pd.Timestamp(args.execution_end, tz="UTC").normalize()
    if execution_end <= asof:
        raise ValueError("settle execution_end must be after asof")
    if execution_end.tz_convert("Asia/Shanghai").normalize() > pd.Timestamp.now(tz="Asia/Shanghai").normalize():
        raise ValueError("settle execution_end cannot be a future local date")
    return settle(args, strategy, asof, execution_end)


def _overlay_closure_paths(args: argparse.Namespace) -> tuple[Path, ...]:
    return (
        Path(args.overlay_config),
        Path(args.overlay_prereg),
        ROOT / "scripts/quiet_forward_rollup_runner.py",
        ROOT / "scripts/quiet_forward_preflight.py",
    )


def _seal_overlay_failure(
    *,
    attempt: Path,
    target: Path,
    args: argparse.Namespace,
    asof: pd.Timestamp,
    execution_end: pd.Timestamp,
    scenario: str,
    failure: EarlyPathFailure,
    bundle_id: str,
    source_cutoffs: dict[str, str],
) -> str:
    payload = {
        "status": failure.code,
        "reason": str(failure),
        "asof": str(asof.date()),
        "settlement_data_cutoff": str(execution_end.date()),
        "scenario": scenario,
        "bundle_id": bundle_id,
        "source_cutoffs": source_cutoffs,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (attempt / "FAILURE.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.copyfile(args.strategy, attempt / "strategy.yaml")
    shutil.copyfile(args.overlay_config, attempt / "early_path_overlay.yaml")
    shutil.copyfile(args.overlay_prereg, attempt / "early_path_preregistration.md")
    shutil.copyfile(args.overlay_registration, attempt / "early_path_registration.json")
    manifest = _code_manifest(Path(args.strategy), _overlay_closure_paths(args))
    (attempt / "code_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(attempt), str(target))
    return str(target)


def _right_tail_recall(
    *,
    arm_outputs: dict[str, dict[str, dict[str, pd.DataFrame]]],
    scenario: str,
) -> dict[str, object]:
    control = arm_outputs["control"][scenario]["fills"]
    shadow = arm_outputs["shadow"][scenario]["fills"]
    if control.empty or "side" not in control.columns:
        return {
            "control_ge_10_count": 0,
            "shadow_retained_ge_10_count": 0,
            "not_decreased": True,
        }
    control_sells = control[control["side"].eq("SELL")].copy()
    shadow_sells = (
        shadow[shadow["side"].eq("SELL")].copy()
        if not shadow.empty and "side" in shadow.columns else pd.DataFrame()
    )
    control_sells["trade_key"] = (
        control_sells["decision_id"].astype(str) + "|" + control_sells["ts_code"].astype(str)
    )
    if shadow_sells.empty:
        shadow_sells = pd.DataFrame(columns=["trade_key", "economic_return"])
    else:
        shadow_sells["trade_key"] = (
            shadow_sells["decision_id"].astype(str) + "|" + shadow_sells["ts_code"].astype(str)
        )
    winners = set(control_sells.loc[control_sells["economic_return"].ge(0.10), "trade_key"])
    retained = int(shadow_sells[
        shadow_sells["trade_key"].isin(winners)
        & shadow_sells["economic_return"].ge(0.10)
    ]["trade_key"].nunique())
    return {
        "control_ge_10_count": len(winners),
        "shadow_retained_ge_10_count": retained,
        "not_decreased": retained >= len(winners),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("freeze", "settle"), default="freeze")
    parser.add_argument("--asof", required=True)
    parser.add_argument("--execution-end")
    parser.add_argument("--strategy", type=Path, default=ROOT / "configs/strategy/reset_weak_confirm_v3_cap1_20_forward.yaml")
    parser.add_argument(
        "--overlay-config", type=Path,
        default=ROOT / "configs/strategy/cap1_early_path_forward_overlay.yaml",
    )
    parser.add_argument(
        "--overlay-prereg", type=Path,
        default=ROOT / "docs/council_20260828/CAP1_EARLY_PATH_FORWARD_PREREG_20260901.md",
    )
    parser.add_argument(
        "--overlay-registration", type=Path,
        default=ROOT / "docs/council_20260828/CAP1_EARLY_PATH_FORWARD_REGISTRATION_20260901_R2.json",
    )
    # Keep formal forward batches isolated from unsealed and diagnostic roots.
    parser.add_argument("--output", type=Path, default=ROOT / "docs/council_20260828/CAP1_20_FORWARD_LOCKBOX")
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()
