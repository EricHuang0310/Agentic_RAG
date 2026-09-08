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


# ==================== 重複反問的防護 ====================
def test_resolve_branch_from_message_accepts_option_value():
    branches = [
        Branch(label="獨資戶變更負責人"),
        Branch(label="公司變更負責人"),
        Branch(label="一般銷戶申請"),
    ]
    from agentic.policy import resolve_branch_from_message

    # 前端送回的 option value 帶了括號關鍵詞
    assert resolve_branch_from_message(
        "獨資戶變更負責人（獨資戶、變更負責人）", branches
    ).label == "獨資戶變更負責人"
    # 只送 label
    assert resolve_branch_from_message("獨資戶變更負責人", branches).label == "獨資戶變更負責人"
    # 自由描述但含有分支名稱
    assert resolve_branch_from_message(
        "我要辦獨資戶變更負責人的銷戶", branches
    ).label == "獨資戶變更負責人"


def test_resolve_branch_from_message_rejects_ambiguous_and_unrelated():
    branches = [Branch(label="獨資戶變更負責人"), Branch(label="公司變更負責人")]
    from agentic.policy import resolve_branch_from_message

    # 同時命中兩個分支 -> 仍然模糊，不可自行挑一個
    assert resolve_branch_from_message("變更負責人", branches) is None
    assert resolve_branch_from_message("不太確定耶", branches) is None
    assert resolve_branch_from_message("", branches) is None


def test_same_branches_under_renamed_dimension_are_not_asked_again():
    """維度名稱由 LLM 生成，兩輪之間可能改名；分支相同就不該再問一次。"""
    branches, chunks = make(["A", "B", "C", "D", "E"], chunk_chars=200)
    action = choose_branch_action(
        branches,
        chunks,
        "申請類型",  # 上一輪叫「業務類別」
        asked_dimensions={"業務類別"},
        asked_branch_labels=["A", "B", "C", "D", "E"],
    )
    assert action.decision == Decision.ANSWER_ALL_BRANCHES
    assert "選項重複" in action.reason


def test_dimension_comparison_ignores_whitespace():
    branches, chunks = make(["A", "B", "C", "D", "E"], chunk_chars=200)
    action = choose_branch_action(
        branches, chunks, " 業務類別 ", asked_dimensions={"業務類別"}
    )
    assert action.decision == Decision.ANSWER_ALL_BRANCHES


def test_clarification_records_branch_labels_for_next_round():
    branches, chunks = make(["A", "B", "C", "D", "E"], chunk_chars=200)
    action = choose_branch_action(branches, chunks, "業務類別")
    assert action.decision == Decision.CLARIFY
    assert action.branch_labels == ["A", "B", "C", "D", "E"]
