from signal_trust.scorer import compute_scores, trust_tag
from signal_trust.constants import (
    TAG_GREEN, TAG_YELLOW, TAG_RED, TAG_NO_DATA,
)
from signal_trust.db import connect


def _insert_samples(db_path: str, rows: list[dict]):
    conn = connect(db_path)
    try:
        for r in rows:
            conn.execute(
                "INSERT INTO signal_trust_samples "
                "(code, trade_date, sample_end_date, pred_10d, actual_10d, version) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (r["code"], r["trade_date"], r["sample_end_date"],
                 r["pred_10d"], r.get("actual_10d"), r.get("version", "ng106")),
            )
        conn.commit()
    finally:
        conn.close()


def test_min_samples_returns_no_data_tag(tmp_db):
    # 只有 5 个样本 < 10
    rows = [{
        "code": "A.SZ", "trade_date": f"2026-01-{i+1:02d}",
        "sample_end_date": f"2026-01-{i+15:02d}",
        "pred_10d": 0.02, "actual_10d": 0.03,
    } for i in range(5)]
    _insert_samples(tmp_db, rows)
    scores = compute_scores(tmp_db, as_of_date="2026-03-01")
    assert scores["A.SZ"]["trust_tag"] == TAG_NO_DATA
    assert scores["A.SZ"]["n_samples"] == 5


def test_three_metrics_correct(tmp_db):
    # 10 个样本: 方向命中 7/10 = 0.7, realize rate actual>0.02 4/10=0.4
    rows = []
    data = [
        (0.02,  0.03),
        (0.02,  0.03),
        (0.02,  0.03),
        (0.02,  0.025),
        (0.02,  0.01),
        (0.02,  0.005),
        (0.02,  0.005),
        (0.02, -0.01),
        (0.02, -0.02),
        (0.02, -0.03),
    ]
    for i, (p, a) in enumerate(data):
        rows.append({
            "code": "B.SZ", "trade_date": f"2026-01-{i+1:02d}",
            "sample_end_date": f"2026-01-{i+15:02d}",
            "pred_10d": p, "actual_10d": a,
        })
    _insert_samples(tmp_db, rows)
    scores = compute_scores(tmp_db, as_of_date="2026-03-01")
    s = scores["B.SZ"]
    assert s["n_samples"] == 10
    assert abs(s["direction_hit_rate"] - 0.7) < 1e-9
    assert abs(s["systematic_bias"] - (sum(a - p for p, a in data) / 10)) < 1e-9
    assert abs(s["high_pred_realize_rate"] - 0.4) < 1e-9


def test_leakage_prevention_excludes_unripe_samples(tmp_db):
    """🚨 核心测试: 样本 sample_end_date >= as_of_date 的不参与计算."""
    base = [{
        "code": "C.SZ", "trade_date": f"2025-{m:02d}-10",
        "sample_end_date": f"2025-{m:02d}-24",
        "pred_10d": 0.02, "actual_10d": 0.03,
    } for m in range(1, 12)]  # 11 个已过期样本
    future = [{
        "code": "C.SZ", "trade_date": "2026-04-10",
        "sample_end_date": "2026-04-26",  # 在 as_of_date=2026-04-12 之后
        "pred_10d": 0.02, "actual_10d": -0.05,  # 极端负值: 若泄露会显著拉低指标
    }]
    _insert_samples(tmp_db, base + future)
    scores = compute_scores(tmp_db, as_of_date="2026-04-12")
    s = scores["C.SZ"]
    assert s["n_samples"] == 11  # future 被排除
    assert s["direction_hit_rate"] == 1.0  # 所有历史样本命中


def test_excludes_null_actual(tmp_db):
    rows = [{"code": "D.SZ", "trade_date": f"2026-01-{i+1:02d}",
             "sample_end_date": f"2026-01-{i+15:02d}",
             "pred_10d": 0.02, "actual_10d": 0.03 if i < 10 else None}
            for i in range(12)]
    _insert_samples(tmp_db, rows)
    scores = compute_scores(tmp_db, as_of_date="2026-03-01")
    assert scores["D.SZ"]["n_samples"] == 10  # 2 条 NULL 被排除


def test_trust_tag_green():
    assert trust_tag(hit=0.60, bias=-0.01, realize=0.50, n=20) == TAG_GREEN


def test_trust_tag_red_any_condition():
    assert trust_tag(hit=0.40, bias=0.0, realize=0.5, n=20) == TAG_RED  # hit 触发
    assert trust_tag(hit=0.60, bias=-0.05, realize=0.5, n=20) == TAG_RED  # bias
    assert trust_tag(hit=0.60, bias=0.0, realize=0.10, n=20) == TAG_RED  # realize


def test_trust_tag_yellow_middle():
    assert trust_tag(hit=0.50, bias=-0.025, realize=0.30, n=20) == TAG_YELLOW


def test_trust_tag_no_data_below_min_samples():
    assert trust_tag(hit=0.99, bias=0.0, realize=0.99, n=8) == TAG_NO_DATA
