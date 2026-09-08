"""RRF 融合：跨變體去重、取最大分數、記錄命中變體。"""

from conftest import item

from agentic.fusion import parse_chunks, rrf_fuse


def test_parse_chunks_score_fallbacks():
    chunks = parse_chunks([
        {"filename": "A.pdf", "content": "甲", "kwargs": {"relevance_score": 1.5}},
        {"filename": "B.pdf", "content": "乙", "relevance_score": 2.5},
        {"filename": "C.pdf", "content": "丙", "score": 3.5},
        {"filename": "D.pdf", "content": "丁"},
    ])
    assert [c.score for c in chunks] == [1.5, 2.5, 3.5, 0.0]
    assert chunks[3].filename == "D.pdf"


def test_same_content_dedupes_across_variants():
    a = parse_chunks([item("A.pdf", 1.0, "同一段內容")])
    b = parse_chunks([item("A.pdf", 2.5, "同一段內容")])
    fused = rrf_fuse([("original", a), ("rewritten", b)])
    assert len(fused) == 1
    # 分數取各變體最大值，代表最佳證據強度
    assert fused[0].score == 2.5
    assert sorted(fused[0].matched_variants) == ["original", "rewritten"]


def test_consensus_across_variants_ranks_higher():
    """只在單一變體拿高分的 chunk，排名應輸給多變體一致命中的 chunk。"""
    v1 = parse_chunks([item("A.pdf", 1.0, "共識段落"), item("B.pdf", 9.0, "單點高分")])
    v2 = parse_chunks([item("A.pdf", 1.0, "共識段落")])
    v3 = parse_chunks([item("A.pdf", 1.0, "共識段落")])
    fused = rrf_fuse([("original", v1), ("rewritten", v2), ("hyde", v3)])
    assert fused[0].content == "共識段落"
    assert fused[0].rrf_score > fused[1].rrf_score


def test_top_n_truncation():
    chunks = parse_chunks([item(f"F{i}.pdf", float(10 - i), f"內容{i}") for i in range(10)])
    fused = rrf_fuse([("original", chunks)], top_n=3)
    assert len(fused) == 3
