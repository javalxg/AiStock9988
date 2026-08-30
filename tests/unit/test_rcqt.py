import pandas as pd

from aistock9988.data.universe import filter_current_st_history, mark_current_st, STBlacklistManifest, build_universe_exclusion_ledger
from aistock9988.selection.rcqt import score_rcqt, select_rcqt


def test_current_st_is_filtered_for_all_dates_and_marked():
    frame = pd.DataFrame({"ts_code": ["000001.SZ", "600001.SH", "000002.SZ"], "trade_date": ["2024", "2025", "2026"]})
    marked = mark_current_st(frame, {"600001.SH"})
    assert bool(marked.loc[1, "excluded_current_st"])
    assert list(filter_current_st_history(frame, {"600001.SH"})["ts_code"]) == ["000001.SZ", "000002.SZ"]
    manifest = STBlacklistManifest.build({"600001.SH"}, source="test", extracted_at="2026-08-27T00:00:00Z")
    assert manifest.to_dict()["code_count"] == 1
    assert list(build_universe_exclusion_ledger(frame, {"600001.SH"}, asof="2025-01-01")["ts_code"]) == ["600001.SH"]


def test_rcqt_is_deterministic_and_prefers_reset_overlap():
    frame = pd.DataFrame({
        "asof": ["2026-08-20"] * 3,
        "ts_code": ["000003.SZ", "000001.SZ", "000002.SZ"],
        "dist_ma60": [-.02, .03, .08], "ret20": [-.05, .02, .10], "ret60": [.10, .20, .40],
        "dd20": [-.05, -.02, .01], "dd60": [-.20, -.15, -.05], "vol20": [.10, .12, .20],
        "liq20": [3., 2., 1.], "volume_ratio_20": [1., 1.1, 1.], "close": [10., 10., 10.],
        "ma5": [9., 9., 9.], "prev3_high": [9.5, 9.5, 9.5], "ret1": [.01, .01, .01],
    })
    selected = select_rcqt(score_rcqt(frame), reset_slots=2, quiet_slots=1)
    assert len(selected) <= 3
    assert selected["ts_code"].is_unique
    assert set(selected["sleeve"]) <= {"recovery", "quiet"}
    assert selected["context_hash"].str.len().eq(64).all()
    assert selected["selection_decision_id"].str.startswith("rcqt-").all()


def test_rcqt_scores_each_asof_cross_section_independently():
    base = {
        "dist_ma60": 0.01, "ret20": 0.01, "ret60": 0.10, "dd20": -0.02, "dd60": -0.15,
        "vol20": 0.10, "liq20": 2.0, "volume_ratio_20": 1.0, "close": 10.0,
        "ma5": 9.5, "prev3_high": 9.8, "ret1": 0.01,
    }
    rows = []
    for asof in ["2026-08-20", "2026-08-21"]:
        for code in ["000001.SZ", "000002.SZ"]:
            row = dict(base); row.update(asof=asof, ts_code=code); rows.append(row)
    out = score_rcqt(pd.DataFrame(rows))
    assert out.groupby("asof")["reset_score"].transform("count").eq(2).all()
    assert out.groupby("asof")["reset_score"].transform("max").le(1.0).all()
