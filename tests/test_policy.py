"""分支策略：候選少就全列，候選多才反問；反問不可無限重複。"""

from conftest import item

from agentic import config
from agentic.fusion import parse_chunks
from agentic.policy import build_options, choose_branch_action
from agentic.schemas import Branch, Decision


def make(labels, chunk_chars=100):
    chunks = parse_chunks([
        item(f"{label}.pdf", 1.0, "內" * chunk_chars) for label in labels
    ])
    branches = [
        Branch(label=label, key_terms=[label], chunk_ids=[chunk.chunk_id])
        for label, chunk in zip(labels, chunks)
    ]
    return branches, chunks


def test_two_short_branches_are_answered_inline():
    branches, chunks = make(["久未往來", "新開戶"], chunk_chars=200)
    action = choose_branch_action(branches, chunks, "業務類別")
    assert action.decision == Decision.ANSWER_ALL_BRANCHES


def test_many_branches_trigger_clarification():
    branches, chunks = make(["A", "B", "C", "D", "E"], chunk_chars=200)
    action = choose_branch_action(branches, chunks, "業務類別")
    assert action.decision == Decision.CLARIFY
    assert "業務類別" in action.clarification_question
    assert all(b.label in action.clarification_question for b in branches)


def test_long_content_triggers_clarification_even_with_few_branches():
    branches, chunks = make(["A", "B"], chunk_chars=config.INLINE_BRANCH_MAX_CHARS)
    action = choose_branch_action(branches, chunks, "業務類別")
    assert action.decision == Decision.CLARIFY


def test_already_asked_dimension_is_not_asked_again():
    branches, chunks = make(["A", "B", "C", "D", "E"], chunk_chars=200)
    action = choose_branch_action(
        branches, chunks, "業務類別", asked_dimensions={"業務類別"}
    )
    assert action.decision == Decision.ANSWER_ALL_BRANCHES


def test_clarification_limit_switches_to_inline():
    branches, chunks = make(["A", "B", "C", "D", "E"], chunk_chars=200)
    action = choose_branch_action(
        branches, chunks, "業務類別", clarification_count=config.MAX_CLARIFICATIONS
    )
    assert action.decision == Decision.ANSWER_ALL_BRANCHES


def test_no_branches_refuses():
    action = choose_branch_action([], [], "業務類別")
    assert action.decision == Decision.REFUSE


def test_options_include_escape_hatch():
    branches, _ = make(["久未往來", "新開戶"])
    options = build_options(branches)
    assert [o.label for o in options][:2] == ["久未往來", "新開戶"]
    assert options[-1].value == "__other__"
