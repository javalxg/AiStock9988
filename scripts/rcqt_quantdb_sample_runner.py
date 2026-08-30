"""Reproducible small quant_db RCQT replay used before full-universe rollout."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from aistock9988.data.quantdb import readonly_connection
from aistock9988.data.q70_source import load_f0_panel
from aistock9988.data.execution_source import load_execution_panel
from aistock9988.data.corporate_actions_source import load_corporate_actions
from aistock9988.selection.rcqt import score_rcqt, select_rcqt
from aistock9988.backtest.engine import run_backtest, BacktestConfig
from aistock9988.reporting.metrics import summarize_backtest
from aistock9988.data.universe import (STBlacklistManifest, filter_current_st_history,
                                       build_universe_exclusion_ledger)

def _features(panel: pd.DataFrame, prices: pd.DataFrame, start: str) -> pd.DataFrame:
    f = panel.copy(); f["asof"] = pd.to_datetime(f.event_time, utc=True).dt.strftime("%Y-%m-%d"); f["close"] = f.economic_close
    p = prices.copy(); p["asof"] = pd.to_datetime(p.trade_date, utc=True).dt.strftime("%Y-%m-%d")
    if "amount" not in p.columns:
        raise ValueError("execution panel must contain amount; refusing implicit liquidity values")
    p["amount"] = pd.to_numeric(p["amount"], errors="coerce")
    if p["amount"].isna().any() or not np.isfinite(p["amount"].to_numpy(dtype=float)).all():
        raise ValueError("execution panel contains non-numeric or missing amount")
    p["liq20"] = p.groupby("ts_code").amount.transform(lambda s: s.rolling(20).median())
    p["volume_ratio_20"] = p.groupby("ts_code").amount.transform(lambda s: s / s.rolling(20).median())
    g = f.sort_values(["ts_code", "event_time"]).reset_index(drop=True); gp = g.groupby("ts_code", group_keys=False)
    g["ma5"] = gp.economic_close.transform(lambda s: s.rolling(5).mean()); g["ma60"] = gp.economic_close.transform(lambda s: s.rolling(60).mean())
    g["dist_ma60"] = g.economic_close / g.ma60 - 1; g["ret1"] = gp.economic_close.transform(lambda s: s.pct_change())
    g["ret5"] = gp.economic_close.transform(lambda s: s.pct_change(5)); g["ret20"] = gp.economic_close.transform(lambda s: s.pct_change(20)); g["ret60"] = gp.economic_close.transform(lambda s: s.pct_change(60))
    # Drawdown is measured from economic highs, not closes, to match the RCQT contract.
    highs = p[["asof", "ts_code", "economic_high"]].copy() if "economic_high" in p.columns else None
    if highs is not None:
        g = g.merge(highs, on=["asof", "ts_code"], how="left")
    else:
        g["economic_high"] = g["economic_close"]
    g = g.sort_values(["ts_code", "event_time"]).reset_index(drop=True)
    gh = g.groupby("ts_code", group_keys=False)
    g["dd20"] = gh.economic_high.transform(lambda s: s / s.rolling(20).max() - 1)
    g["dd60"] = gh.economic_high.transform(lambda s: s / s.rolling(60).max() - 1)
    # Rebuild the groupby after merge/reset; reusing ``gp`` would align the
    # rolling series by stale row indexes and silently mix securities.
    g["vol20"] = gh.ret1.transform(lambda s: s.rolling(20).std())
    g["prev3_high"] = gh.economic_close.transform(lambda s: s.shift(1).rolling(3).max())
    g = g.merge(p[["asof", "ts_code", "liq20", "volume_ratio_20"]], on=["asof", "ts_code"], how="left")
    event_cols = [c for c in ("kdj_k_bfq", "cci_bfq", "wr_bfq") if c in g.columns]
    cols = ["asof", "ts_code", "dist_ma60", "ret1", "ret5", "ret20", "ret60", "dd20", "dd60", "vol20", "liq20", "volume_ratio_20", "close", "ma5", "prev3_high", *event_cols]
    return g[cols + ["available_time"]].dropna().loc[lambda x: x["asof"] >= start]

def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, required=True); ap.add_argument("--limit", type=int, default=50); ap.add_argument("--start", default="2026-01-01"); ap.add_argument("--end", default="2026-01-31"); ap.add_argument("--hold-sessions", type=int, default=10); ap.add_argument("--max-positions", type=int, default=6); ap.add_argument("--single-weight-cap", type=float, default=0.15); ap.add_argument("--equity-cap", type=float, default=0.72); ap.add_argument("--no-trailing", action="store_true"); ap.add_argument("--no-stop", action="store_true"); ap.add_argument("--stop-loss-pct", type=float, default=-0.08); ap.add_argument("--no-right-confirm", action="store_true"); ap.add_argument("--max-order-to-adv20", type=float, default=None); ap.add_argument("--slippage", type=float, default=0.0); ap.add_argument("--event-factors", action="store_true"); ap.add_argument("--regime-cap", action="store_true", help="cap new-entry equity at 0.40 when market proxy is below MA60"); ap.add_argument("--codes-file", type=Path); a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    with readonly_connection() as c:
        st_rows = pd.read_sql_query("SELECT ts_code, name FROM stock_basic_ts", c)
        st_codes = set(st_rows.loc[st_rows["name"].fillna("").str.upper().str.contains("ST|退市风险", regex=True), "ts_code"].astype(str).str.upper())
        if a.codes_file:
            candidates = pd.read_csv(a.codes_file)["ts_code"].astype(str).str.upper()
        else:
            candidates = pd.read_sql_query("SELECT DISTINCT ts_code FROM market_daily_ts WHERE source='daily' AND trade_date=(SELECT MAX(trade_date) FROM market_daily_ts WHERE source='daily' AND trade_date<=%s) AND amount>0 AND ts_code NOT LIKE %s ORDER BY ts_code", c, params=(a.start, "%.BJ"))["ts_code"].astype(str).str.upper()
        codes = [code for code in candidates if code not in st_codes][:a.limit]
    st_manifest = STBlacklistManifest.build(st_codes, source="quant_db.stock_basic_ts.name", extracted_at=pd.Timestamp.utcnow().isoformat())
    (a.output / "run_config.json").write_text(json.dumps({
        "start": a.start, "end": a.end, "limit": a.limit,
        "hold_sessions": a.hold_sessions, "max_positions": a.max_positions,
        "stop_loss_pct": None if a.no_stop else a.stop_loss_pct,
        "trailing": not a.no_trailing, "right_confirmation": not a.no_right_confirm,
        "max_order_to_adv20": a.max_order_to_adv20, "slippage": a.slippage,
        "event_factors": a.event_factors, "event_weight": getattr(a, "event_weight", 0.15),
        "regime_cap": a.regime_cap, "codes_file": str(a.codes_file) if a.codes_file else None,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (a.output / "st_manifest.json").write_text(json.dumps(st_manifest.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame({"ts_code": codes}).to_csv(a.output / "codes.csv", index=False)
    lookback_start = (pd.Timestamp(a.start) - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    panel = load_f0_panel(lookback_start, a.end, ts_codes=codes); prices = load_execution_panel(lookback_start, a.end, ts_codes=codes)
    panel = filter_current_st_history(panel, st_codes); prices = filter_current_st_history(prices, st_codes)
    feat = _features(panel, prices, a.start)
    if "available_time" not in feat.columns:
        raise RuntimeError("strict PIT violation: quant_db features must include available_time")
    feat = feat[feat["asof"] <= a.end]; scored = score_rcqt(feat, require_right_confirmation=not a.no_right_confirm)
    if a.event_factors and {"kdj_k_bfq", "cci_bfq", "wr_bfq"} <= set(scored.columns):
        # Lower KDJ/CCI and higher WR are the pre-registered oversold direction.
        def pct(s): return s.groupby(scored["asof"]).rank(pct=True, method="average")
        oversold = (1 - pct(scored["kdj_k_bfq"]) + 1 - pct(scored["cci_bfq"]) + pct(scored["wr_bfq"])) / 3
        scored["reset_score"] = 0.85 * scored["reset_score"] + 0.15 * oversold
    if a.regime_cap:
        # Quant DB deployments may not contain an index series. Use a
        # point-in-time equal-weight universe proxy (median close) instead of
        # silently disabling the gate.
        idx = panel.assign(asof=pd.to_datetime(panel.event_time, utc=True).dt.strftime("%Y-%m-%d")) \
                  .groupby("asof", as_index=False)["economic_close"].median() \
                  .rename(columns={"economic_close": "close"}).sort_values("asof")
        idx["close"] = pd.to_numeric(idx["close"], errors="coerce"); idx["ma60"] = idx["close"].rolling(60, min_periods=60).mean()
        caps = dict(zip(idx["asof"], idx.apply(lambda r: 0.40 if pd.notna(r.ma60) and r.close < r.ma60 else 0.72, axis=1)))
        selected = pd.concat([select_rcqt(g, equity_cap=caps.get(str(day), 0.72)) for day, g in scored.groupby("asof", sort=True)], ignore_index=True)
    else:
        selected = select_rcqt(scored, single_weight_cap=a.single_weight_cap, equity_cap=a.equity_cap)
    sig = selected[["asof", "ts_code", "candidate_rank", "selected", "selection_decision_id", "policy_id", "target_weight", "context_hash", "sleeve"]]
    px = prices.copy()
    px["amount"] = pd.to_numeric(px["amount"], errors="raise")
    # Compute ADV20 on the full lookback panel before slicing the test window,
    # so the first test sessions have a continuous PIT liquidity history.
    px["adv20"] = px.groupby("ts_code")["amount"].transform(lambda s: s.rolling(20, min_periods=1).median())
    px = px[(pd.to_datetime(px.trade_date, utc=True).dt.strftime("%Y-%m-%d") >= a.start) & (pd.to_datetime(px.trade_date, utc=True).dt.strftime("%Y-%m-%d") <= a.end)]
    actions = load_corporate_actions(a.start, a.end, ts_codes=codes)
    actions.to_csv(a.output / "corporate_actions.csv", index=False)
    result = run_backtest(sig, px, config=BacktestConfig(max_positions=a.max_positions, hold_sessions=a.hold_sessions, lot_size=100, stop_loss_pct=None if a.no_stop else a.stop_loss_pct, trailing_arm_pct=None if a.no_trailing else .10, trailing_drawdown_pct=None if a.no_trailing else .08, max_order_to_adv20=a.max_order_to_adv20, buy_slippage=a.slippage, sell_slippage=a.slippage), corporate_actions=actions)
    feat.to_csv(a.output / "score_ledger.csv", index=False); selected.to_csv(a.output / "selection_ledger.csv", index=False)
    selected.assign(candidate_status=lambda d: d["selected"].map({True: "SELECTED", False: "REJECTED_BY_POLICY"})).to_csv(a.output / "candidate_ledger.csv", index=False)
    build_universe_exclusion_ledger(pd.DataFrame({"ts_code": sorted(st_codes)}), st_codes, asof=a.start).to_csv(a.output / "universe_exclusion_ledger.csv", index=False)
    def _sha(path: Path) -> str:
        import hashlib
        return hashlib.sha256(path.read_bytes()).hexdigest()
    for name, frame in result.items(): frame.to_csv(a.output / f"{name}.csv", index=False)
    metrics = summarize_backtest(result["nav"], result["trades"], initial_cash=1_000_000)
    metrics["forced_final_liquidation_count"] = int((result["trades"].get("reason", pd.Series(dtype=str)) == "end_of_test_liquidation").sum())
    nav_dates = pd.to_datetime(result["nav"]["trade_date"], utc=True)
    weekly_nav = result["nav"].assign(_week=nav_dates.dt.to_period("W-SUN")).groupby("_week")["nav"].last()
    weekly_returns = weekly_nav.pct_change(); weekly_returns.iloc[0] = weekly_nav.iloc[0] / 1_000_000 - 1.0
    metrics["weekly_target"] = 0.05
    metrics["weekly_target_hit_ratio"] = float((weekly_returns >= 0.05).mean()) if len(weekly_returns) else None
    metrics["max_weekly_return"] = float(weekly_returns.max()) if len(weekly_returns) else None
    (a.output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str) + "\n")
    (a.output / "RESULT.md").write_text("# RCQT quant_db sample replay\n\n" + json.dumps({"codes": len(codes), "features": len(feat), "selected": len(selected), **metrics}, ensure_ascii=False, indent=2, default=str) + "\n")
    artifacts = [p for p in a.output.iterdir() if p.is_file() and p.name != "data_manifest.json"]
    (a.output / "data_manifest.json").write_text(json.dumps({
        "source_mode": "quant_db", "st_blacklist": st_manifest.to_dict(),
        "st_policy": "current_name_applied_to_full_history", "code_count": len(codes),
        "start": a.start, "end": a.end, "hold_sessions": a.hold_sessions,
        "stop_loss_enabled": not a.no_stop, "trailing_enabled": not a.no_trailing,
        "adv20_definition": "rolling_20_session_median_amount",
        "artifacts": {p.name: {"sha256": _sha(p)} for p in sorted(artifacts, key=lambda x: x.name)},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__": main()
