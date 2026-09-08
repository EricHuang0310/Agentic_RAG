"""回答生成的輔助邏輯：引用編號驗證與來源清單。"""

from conftest import item

from agentic.fusion import parse_chunks
from agentic.generate import (
    build_context,
    build_references_summary,
    check_citations,
)


def test_check_citations_splits_valid_and_invalid():
    text = "第一步…[文獻 1]\n第二步…[文獻 3]\n第三步…[文獻 9]"
    valid, invalid = check_citations(text, n_contexts=3)
    assert valid == [1, 3]
    assert invalid == [9]


def test_check_citations_tolerates_spacing_and_duplicates():
    valid, invalid = check_citations("A[文獻1] B[文獻  2] C[文獻 2]", n_contexts=2)
    assert valid == [1, 2]
    assert invalid == []


def test_context_numbering_matches_chunk_order():
    chunks = parse_chunks([item("A.pdf", 2.0, "甲"), item("B.pdf", 1.0, "乙")])
    context = build_context(chunks)
    assert "[文獻 1]" in context and "A.pdf" in context
    assert "[文獻 2]" in context and "B.pdf" in context


def test_empty_content_chunks_are_skipped_in_context():
    chunks = parse_chunks([item("A.pdf", 2.0, ""), item("B.pdf", 1.0, "乙")])
    assert build_context(chunks).count("[文獻") == 1


def test_references_summary_dedupes():
    chunks = parse_chunks([
        item("A.pdf", 2.0, "甲", page="1"),
        item("A.pdf", 1.5, "乙", page="1"),
        item("A.pdf", 1.0, "丙", page="2"),
    ])
    summary = build_references_summary(chunks)
    assert summary.count("A.pdf") == 2
