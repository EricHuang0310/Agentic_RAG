"""數值訊號：重點在能否區分「證據集中」與「證據分散」。"""

from conftest import item

from agentic.fusion import parse_chunks
from agentic.signals import compute_signals


def test_empty():
    signals = compute_signals([])
    assert signals.n_chunks == 0
    assert signals.top1 == 0.0


def test_concentrated_single_source():
    chunks = parse_chunks([
        item("開戶作業.pdf", 3.0, "甲"),
        item("開戶作業.pdf", 2.5, "乙"),
        item("開戶作業.pdf", 2.0, "丙"),
    ])
    signals = compute_signals(chunks)
    assert signals.n_sources == 1
    assert signals.source_entropy == 0.0
    assert signals.max_source_share == 1.0
    assert signals.top1 == 3.0
    assert signals.gap == 0.5


def test_dispersed_multi_source_has_high_entropy():
    chunks = parse_chunks([
        item("久未往來.pdf", 1.2, "甲"),
        item("開戶作業.pdf", 1.1, "乙"),
        item("綜合對帳單.pdf", 1.0, "丙"),
    ])
    signals = compute_signals(chunks)
    assert signals.n_sources == 3
    assert signals.source_entropy > 0.9
    assert signals.max_source_share < 0.45
    assert signals.gap < 0.5  # 互相競爭


def test_negative_scores_do_not_break_weights():
    chunks = parse_chunks([
        item("A.pdf", -3.0, "甲"),
        item("B.pdf", -5.0, "乙"),
    ])
    signals = compute_signals(chunks)
    assert signals.top1 == -3.0
    assert 0.0 <= signals.source_entropy <= 1.0
    assert 0.0 < signals.max_source_share <= 1.0


def test_counts_use_thresholds():
    chunks = parse_chunks([
        item("A.pdf", 2.0, "甲"),   # >= answerable(1.0)
        item("A.pdf", 0.5, "乙"),   # >= supportive(0.0)
        item("B.pdf", -2.0, "丙"),  # 兩者皆否
    ])
    signals = compute_signals(chunks)
    assert signals.n_answerable == 1
    assert signals.n_supportive == 2
