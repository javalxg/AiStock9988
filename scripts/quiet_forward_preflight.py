"""Read-only preflight for selecting the latest complete forward signal day."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from aistock9988.configuration import StrategyConfig
from aistock9988.data.quantdb import readonly_connection
from aistock9988.forward.early_path import EarlyPathConfig
from aistock9988.time.session import session_close

from quiet_forward_shadow_runner import _verify_overlay_registration

ROOT = Path(__file__).resolve().parents[1]

_SOURCE_TABLES = {
    "market_daily_ts": "market_daily_ts",
    "adj_factor_ts": "adj_factor_ts",
    "stk_limit_ts": "stk_limit_ts",
    "daily_basic_ts": "daily_basic_ts",
}


def _normalize(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.normalize()


def _required_sources(strategy: StrategyConfig) -> list[str]:
    dense = strategy.data_policy["dense_required"]
    names = sorted({str(source) for stage in dense.values() for source in stage})
    unknown = [name for name in names if name not in _SOURCE_TABLES]
    if unknown:
        raise ValueError(f"preflight has no max-date query for dense sources: {unknown}")
    return names


def _max_dates(sources: list[str]) -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    with readonly_connection() as connection:
        for source in sources:
            table = _SOURCE_TABLES[source]
            value = pd.read_sql_query(
                f"SELECT MAX(trade_date) AS max_date FROM {table}", connection
            ).iloc[0, 0]
            values[source] = None if pd.isna(value) else str(pd.Timestamp(value).date())
    return values


def _is_frozen(lockbox_root: Path, strategy: StrategyConfig, day: pd.Timestamp) -> bool:
    manifests = lockbox_root / "manifests"
    if not manifests.exists():
        return False
    target = str(day.date())
    for path in sorted(manifests.glob("manifest-*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("experiment_id") != strategy.strategy_id or payload.get("config_sha256") != strategy.config_hash:
            continue
        if any(str(part.get("asof")) == target for part in payload.get("parts", ())):
            return True
    return False


def preflight(args: argparse.Namespace) -> dict[str, object]:
    strategy = StrategyConfig.from_yaml(args.strategy)
    overlay = EarlyPathConfig.from_yaml(args.overlay_config)
    overlay.validate_control(strategy)
    if not args.overlay_prereg.is_file():
        raise FileNotFoundError(f"early-path preregistration is missing: {args.overlay_prereg}")
    try:
        registration = _verify_overlay_registration(args, strategy, overlay)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        return {
            "status": "BLOCKED_REGISTRATION_DRIFT",
            "reason": str(exc),
            "database_query_executed": False,
        }
    sources = _required_sources(strategy)
    max_dates = _max_dates(sources)
    missing_sources = sorted(name for name, value in max_dates.items() if value is None)
    if missing_sources:
        return {
            "status": "WAITING_FOR_DATA",
            "reason": "required_dense_source_has_no_rows",
            "missing_sources": missing_sources,
            "required_sources": sources,
            "data_max_by_source": max_dates,
        }
    available = [pd.Timestamp(value) for value in max_dates.values() if value is not None]
    if not available:
        return {
            "status": "WAITING_FOR_DATA",
            "reason": "no_dense_source_rows",
            "required_sources": sources,
            "data_max_by_source": max_dates,
        }

    cutoff = min(_normalize(value) for value in available)
    requested = _normalize(args.asof) if args.asof else cutoff
    research_status = str(strategy.identity.get("research_status", "historical"))
    forward_start = (
        _normalize(strategy.identity["forward_start"])
        if research_status == "forward_only" else None
    )
    payload: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "strategy_id": strategy.strategy_id,
        "strategy_config_sha256": strategy.config_hash,
        "overlay_id": str(overlay.identity["overlay_id"]),
        "overlay_config_sha256": overlay.config_hash,
        "overlay_prereg_sha256": hashlib.sha256(args.overlay_prereg.read_bytes()).hexdigest(),
        "overlay_registration_sha256": registration["registration_sha256"],
        "registered_code_manifest_sha256": registration["code_manifest_sha256"],
        "required_sources": sources,
        "data_max_by_source": max_dates,
        "common_data_cutoff": str(cutoff.date()),
        "target_asof": str(requested.date()),
        "research_status": research_status,
        "forward_start": str(forward_start.date()) if forward_start is not None else None,
        "database_query_read_only": True,
        "no_future_rows_used": True,
    }

    if forward_start is not None and requested < forward_start:
        payload.update(status="WAITING_FOR_DATA", reason="target_before_forward_start")
    elif requested > cutoff:
        payload.update(status="WAITING_FOR_DATA", reason="target_after_common_cutoff")
    elif requested.tz_convert("Asia/Shanghai").normalize() > pd.Timestamp.now(tz="Asia/Shanghai").normalize():
        payload.update(status="WAITING_FOR_DATA", reason="target_is_future_date")
    else:
        # Use the exchange calendar already persisted by the application only
        # through the common data cutoff; no future trading session is needed.
        from aistock9988.data.bundle import load_trading_calendar

        calendar = load_trading_calendar(
            str((requested - pd.Timedelta(days=10)).date()), str(cutoff.date())
        )
        sessions = pd.DatetimeIndex(pd.to_datetime(calendar["session"], utc=True)).normalize()
        if requested not in sessions:
            payload.update(status="WAITING_FOR_DATA", reason="target_not_completed_exchange_session")
        elif requested.tz_convert("Asia/Shanghai").normalize() == pd.Timestamp.now(tz="Asia/Shanghai").normalize() and pd.Timestamp.now(tz="Asia/Shanghai") < session_close(requested).tz_convert("Asia/Shanghai"):
            payload.update(status="WAITING_FOR_DATA", reason="session_close_not_complete")
        elif _is_frozen(args.lockbox.resolve(), strategy, requested):
            payload.update(status="ALREADY_FROZEN", reason="target_exists_in_lockbox")
        else:
            payload.update(status="READY_TO_FREEZE", reason="common_cutoff_is_complete_session")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asof", help="UTC/session date to check; defaults to latest common cutoff")
    parser.add_argument(
        "--strategy",
        type=Path,
        default=ROOT / "configs/strategy/reset_weak_confirm_v3_cap1_20_forward.yaml",
    )
    parser.add_argument(
        "--overlay-config",
        type=Path,
        default=ROOT / "configs/strategy/cap1_early_path_forward_overlay.yaml",
    )
    parser.add_argument(
        "--overlay-prereg",
        type=Path,
        default=ROOT / "docs/council_20260828/CAP1_EARLY_PATH_FORWARD_PREREG_20260901.md",
    )
    parser.add_argument(
        "--overlay-registration",
        type=Path,
        default=ROOT / "docs/council_20260828/CAP1_EARLY_PATH_FORWARD_REGISTRATION_20260901_R2.json",
    )
    # Formal forward evidence must never share a root with unsealed batches.
    parser.add_argument(
        "--lockbox",
        type=Path,
        default=ROOT / "docs/council_20260828/CAP1_20_FORWARD_LOCKBOX",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = preflight(args)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output:
        output = args.output.resolve()
        if output.exists():
            raise FileExistsError(f"preflight output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
