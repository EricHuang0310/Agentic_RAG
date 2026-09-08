"""Agent 主迴圈的端到端行為（不連外部服務）。

每個測試對應流程圖的一條路徑，重點驗證兩件事：
1. 該自己解決的（措辭、複合問題）不要去問使用者
2. 任何情況下都不要在證據不足時硬答
"""

import asyncio

from conftest import FakeLLM, FakeRetriever, item

from agentic import config
from agentic.budget import Budget
from agentic.orchestrator import Agent
from agentic.schemas import Decision, Diagnosis
from agentic.sessions import SessionState
from agentic.trace import Tracer


def run_agent(retriever, llm, message="遺留物現金怎麼處理？", session=None, budget=None):
    agent = Agent(retriever, llm)
    tracer = Tracer("test")
    result, state = asyncio.run(agent.run(message, session, tracer, budget=budget))
    return result, state, tracer


# 分數採 0～1 正規化尺度，與實際 reranker 一致
# （SCORE_ANSWERABLE=0.7、SCORE_SUPPORTIVE=0.35、GAP_AMBIGUOUS=0.08）
STRONG_HIT = [item("遺留物處理.pdf", 0.92, "遺留物現金應於當日…" * 5)]
WEAK_SPREAD = [
    item("久未往來.pdf", 0.08, "久未往來帳戶…"),
    item("開戶作業.pdf", 0.06, "開戶應…"),
    item("綜合對帳單.pdf", 0.04, "對帳單…"),
]
# 三個來源分數接近（gap=0.03 < 0.08）-> 互相競爭
COMPETING = [
    item("久未往來.pdf", 0.74, "久未往來帳戶的處理…" * 3),
    item("開戶作業.pdf", 0.71, "開戶時的處理…" * 3),
    item("綜合對帳單.pdf", 0.69, "對帳單的處理…" * 3),
]


# ==================== (0) 證據充足：直接回答 ====================
def test_sufficient_answers_without_clarification():
    retriever = FakeRetriever(lambda q, i: STRONG_HIT)
    llm = FakeLLM({
        "grade": {"verdict": "sufficient", "relevant_chunks": ["C1"]},
        "answer": "一、清點現金並登錄…[文獻 1]",
    })
    result, state, _ = run_agent(retriever, llm)

    assert result.decision == Decision.ANSWER
    assert result.diagnosis == Diagnosis.SUFFICIENT
    assert result.status == "completed"
    assert state is None
    # happy path 只花 1 次檢索 + 1 次 grading，沒有多餘的 LLM 呼叫
    assert len(retriever.queries) == 1
    assert llm.calls == ["grade", "answer"]


# ==================== (a) 措辭不匹配：自己改寫重試，不反問 ====================
def test_lexical_mismatch_retries_with_variants_instead_of_asking():
    def responder(query, index):
        # 第一輪用原問法只拿到弱命中，改寫過的問法才命中
        return STRONG_HIT if "遺留物品" in query or "假想" in query else [
            item("遺留物處理.pdf", 0.4, "相關但偏離重點的段落")
        ]

    retriever = FakeRetriever(responder)
    llm = FakeLLM({
        "grade": [
            {"verdict": "lexical_mismatch", "missing_information": "缺少具體步驟"},
            {"verdict": "sufficient", "relevant_chunks": ["C1"]},
        ],
        "analyze": {
            "is_compound": False,
            "sub_questions": [],
            "rewritten_query": "遺留物品中的現金應如何處理",
            "keyword_query": "遺留物品 現金 處理",
            "hyde_passage": "假想條文：遺留物品之現金應…",
        },
        "answer": "一、清點…[文獻 1]",
    })
    result, state, _ = run_agent(retriever, llm)

    assert result.decision == Decision.ANSWER
    assert state is None
    # 第二輪確實用了改寫後的變體
    assert any("遺留物品" in q for q in retriever.queries)
    assert "analyze" in llm.calls
    assert len(retriever.queries) > 1


# ==================== (b) 複合問題：拆解，不反問 ====================
def test_compound_question_is_decomposed_not_asked():
    def responder(query, index):
        if "多久" in query or "證件" in query:
            return STRONG_HIT
        return [item("開戶作業.pdf", 0.3, "偏離重點")]

    retriever = FakeRetriever(responder)
    llm = FakeLLM({
        "grade": [
            {"verdict": "lexical_mismatch"},
            {"verdict": "sufficient", "relevant_chunks": ["C1"]},
        ],
        "analyze": {
            "is_compound": True,
            "sub_questions": ["開戶要準備哪些證件", "開戶多久生效"],
            "rewritten_query": "開戶應備文件與生效時間",
            "keyword_query": "開戶 證件 生效",
            "hyde_passage": "假想條文…",
        },
        "answer": "一、應備證件…[文獻 1]",
    })
    result, state, tracer = run_agent(retriever, llm, message="開戶要準備什麼、多久生效？")

    assert result.decision == Decision.ANSWER
    assert state is None
    # 兩個子問題都各自檢索過
    assert any("證件" in q for q in retriever.queries)
    assert any("多久" in q for q in retriever.queries)
    assert any(s.step == "diagnose" and s.detail.get("diagnosis") == "compound" for s in tracer.steps)


# ==================== (c-1) 分支少：分情境全列，不反問 ====================
def test_few_short_branches_are_answered_inline():
    retriever = FakeRetriever(lambda q, i: COMPETING[:2])
    llm = FakeLLM({
        "grade": {"verdict": "multi_branch", "relevant_chunks": ["C1", "C2"]},
        "analyze": {"is_compound": False, "rewritten_query": "", "keyword_query": "", "hyde_passage": ""},
        "branches": {
            "dimension": "業務類別",
            "branches": [
                {"label": "久未往來帳戶", "key_terms": ["久未往來"], "chunks": ["C1"]},
                {"label": "新開戶", "key_terms": ["開戶"], "chunks": ["C2"]},
            ],
        },
        "branch_answer": "本問題依業務類別不同…\n若屬久未往來帳戶：…[文獻 1]\n若屬新開戶：…[文獻 2]",
    })
    result, state, _ = run_agent(retriever, llm)

    assert result.decision == Decision.ANSWER_ALL_BRANCHES
    assert result.diagnosis == Diagnosis.MULTI_BRANCH
    assert result.status == "completed"
    assert state is None  # 不進入等待反問狀態
    assert "若屬" in result.answer


# ==================== (c-2) 分支多：反問，帶結構化選項 ====================
def test_many_branches_trigger_clarification_with_options():
    many = [
        item(f"業務{i}.pdf", 0.74 - i * 0.01, "內容" * 400) for i in range(5)
    ]
    retriever = FakeRetriever(lambda q, i: many)
    llm = FakeLLM({
        "grade": {"verdict": "multi_branch", "relevant_chunks": [f"C{i}" for i in range(1, 6)]},
        "analyze": {"is_compound": False, "rewritten_query": "", "keyword_query": "", "hyde_passage": ""},
        "branches": {
            "dimension": "業務類別",
            "branches": [
                {"label": f"業務{i}", "key_terms": [f"關鍵{i}"], "chunks": [f"C{i + 1}"]}
                for i in range(5)
            ],
        },
    })
    result, state, _ = run_agent(retriever, llm)

    assert result.decision == Decision.CLARIFY
    assert result.status == "awaiting_clarification"
    assert "業務類別" in result.answer
    # 5 個分支 + 1 個「以上皆非」
    assert len(result.clarification_options) == 6
    assert result.clarification_options[-1].value == "__other__"
    # 反問時不回傳 chunks，避免同仁在未確認情境前就照著讀
    assert result.chunks == []
    assert state is not None
    assert state.awaiting_clarification
    assert state.clarification_count == 1
    assert state.asked_dimensions == ["業務類別"]


# ==================== (d) 語料庫沒有：拒答，不反問 ====================
def test_out_of_scope_refuses_instead_of_asking():
    retriever = FakeRetriever(lambda q, i: WEAK_SPREAD)
    llm = FakeLLM({
        "grade": {"verdict": "not_in_corpus"},
        "analyze": {"is_compound": False, "rewritten_query": "", "keyword_query": "", "hyde_passage": ""},
    })
    result, state, _ = run_agent(retriever, llm, message="信用卡分期利率怎麼算？")

    assert result.decision == Decision.REFUSE
    assert result.status == "refused"
    assert result.clarification_options == []
    assert state is None
    # 拒答時仍保留 chunks，方便維運判斷是語料缺漏還是檢索問題
    assert result.chunks


# ==================== 反問後的第二輪：LLM 重構，不做字串拼接 ====================
def test_followup_uses_standalone_rewrite():
    retriever = FakeRetriever(lambda q, i: STRONG_HIT)
    llm = FakeLLM({
        "standalone": "久未往來帳戶的遺留物現金應如何處理",
        "grade": {"verdict": "sufficient", "relevant_chunks": ["C1"]},
        "answer": "一、…[文獻 1]",
    })
    session = SessionState(
        original_query="遺留物現金怎麼處理？",
        clarification_question="請問屬於哪一種業務類別？",
        asked_dimensions=["業務類別"],
        clarification_count=1,
        status="awaiting_clarification",
    )
    result, state, _ = run_agent(retriever, llm, message="久未往來", session=session)

    assert result.decision == Decision.ANSWER
    assert result.effective_query == "久未往來帳戶的遺留物現金應如何處理"
    # 送去檢索的是重構後的獨立問句，不是「原問題\n補充說明：…」的拼接
    assert retriever.queries[0] == "久未往來帳戶的遺留物現金應如何處理"
    assert "補充說明" not in retriever.queries[0]
    assert llm.calls[0] == "standalone"
    assert state is None  # 完成後清除 session


def test_followup_falls_back_to_concatenation_when_rewrite_fails():
    retriever = FakeRetriever(lambda q, i: STRONG_HIT)
    llm = FakeLLM({
        "standalone": None,  # 模擬重構呼叫失敗
        "grade": {"verdict": "sufficient", "relevant_chunks": ["C1"]},
        "answer": "一、…[文獻 1]",
    })
    session = SessionState(
        original_query="遺留物現金怎麼處理？",
        clarification_question="請問屬於哪一種？",
        clarification_count=1,
        status="awaiting_clarification",
    )
    result, _, _ = run_agent(retriever, llm, message="久未往來", session=session)

    assert "遺留物現金怎麼處理" in result.effective_query
    assert "久未往來" in result.effective_query
    assert result.decision == Decision.ANSWER


# ==================== 反問次數上限 ====================
def test_second_round_does_not_ask_the_same_dimension_again():
    many = [item(f"業務{i}.pdf", 0.74 - i * 0.01, "內容" * 400) for i in range(5)]
    retriever = FakeRetriever(lambda q, i: many)
    llm = FakeLLM({
        "standalone": "補充後的獨立問句",
        "grade": {"verdict": "multi_branch", "relevant_chunks": ["C1", "C2"]},
        "analyze": {"is_compound": False, "rewritten_query": "", "keyword_query": "", "hyde_passage": ""},
        "branches": {
            "dimension": "業務類別",
            "branches": [
                {"label": f"業務{i}", "key_terms": [], "chunks": [f"C{i + 1}"]} for i in range(5)
            ],
        },
        "branch_answer": "若屬業務0：…[文獻 1]",
    })
    session = SessionState(
        original_query="原問題",
        clarification_question="請問屬於哪一種業務類別？",
        asked_dimensions=["業務類別"],
        clarification_count=1,
        status="awaiting_clarification",
    )
    result, state, _ = run_agent(retriever, llm, message="不太確定", session=session)

    # 同一個維度不再問第二次，改為分情境全列
    assert result.decision == Decision.ANSWER_ALL_BRANCHES
    assert state is None


# ==================== 知識庫連線失敗 ====================
def test_retrieve_failure_is_reported_not_answered():
    retriever = FakeRetriever(lambda q, i: None)
    llm = FakeLLM({})
    result, state, _ = run_agent(retriever, llm)

    assert result.decision == Decision.REFUSE
    assert "無法連線" in result.answer
    assert llm.calls == []  # 沒有證據就不該呼叫 LLM
    assert state is None


# ==================== 預算耗盡：反問或拒答，絕不硬答 ====================
def test_budget_exhaustion_never_hard_answers():
    retriever = FakeRetriever(lambda q, i: COMPETING)
    llm = FakeLLM({
        "grade": {"verdict": "lexical_mismatch"},
        "analyze": {
            "is_compound": False,
            "rewritten_query": "改寫後的問句",
            "keyword_query": "關鍵詞",
            "hyde_passage": "假想條文",
        },
        "answer": "不該出現的硬答",
        "branch_answer": "不該出現的硬答",
    })
    # 只夠 grade + analyze，沒有預算再開一輪
    result, state, tracer = run_agent(retriever, llm, budget=Budget(max_llm_calls=2))

    assert result.decision in (Decision.CLARIFY, Decision.REFUSE)
    assert "硬答" not in result.answer
    assert result.budget_exhausted
    assert any(s.step == "budget_exhausted" for s in tracer.steps)


def test_grader_failure_with_weak_scores_refuses():
    retriever = FakeRetriever(lambda q, i: WEAK_SPREAD)
    llm = FakeLLM({
        "grade": "這不是 JSON",
        "analyze": "這也不是 JSON",
        "answer": "不該出現的硬答",
    })
    result, _, _ = run_agent(retriever, llm)

    assert result.decision == Decision.REFUSE
    assert "硬答" not in result.answer


# ==================== trace 完整性 ====================
def test_trace_records_every_decision_step():
    retriever = FakeRetriever(lambda q, i: STRONG_HIT)
    llm = FakeLLM({
        "grade": {"verdict": "sufficient", "relevant_chunks": ["C1"]},
        "answer": "一、…[文獻 1]",
    })
    _, _, tracer = run_agent(retriever, llm)

    steps = [s.step for s in tracer.steps]
    for expected in ("normalize_query", "retrieve", "signals", "grade", "diagnose", "generate_answer"):
        assert expected in steps


def test_invalid_citation_is_flagged_in_answer_and_trace():
    retriever = FakeRetriever(lambda q, i: STRONG_HIT)
    llm = FakeLLM({
        "grade": {"verdict": "sufficient", "relevant_chunks": ["C1"]},
        "answer": "一、清點現金…[文獻 7]",  # 只有 1 段參考資訊，7 不存在
    })
    result, _, tracer = run_agent(retriever, llm)

    assert "系統提醒" in result.answer
    generate_step = next(s for s in tracer.steps if s.step == "generate_answer")
    assert generate_step.detail["invalid_citations"] == [7]


# ==================== 重複反問的防護 ====================
BRANCHES_5 = {
    "dimension": "銷戶申請方式",
    "branches": [
        {"label": "獨資戶變更負責人", "key_terms": ["獨資戶"], "chunks": ["C1"]},
        {"label": "公司變更負責人", "key_terms": ["公司"], "chunks": ["C2"]},
        {"label": "一般銷戶申請", "key_terms": ["銷戶"], "chunks": ["C3"]},
        {"label": "其他情形一", "key_terms": [], "chunks": ["C4"]},
        {"label": "其他情形二", "key_terms": [], "chunks": ["C5"]},
    ],
}
LONG_COMPETING = [
    item(f"銷戶{i}.pdf", 0.74 - i * 0.01, "內容" * 400) for i in range(5)
]


def multi_branch_llm(**overrides):
    responses = {
        "grade": {"verdict": "multi_branch", "relevant_chunks": ["C1", "C2", "C3"]},
        "analyze": {"is_compound": False, "rewritten_query": "", "keyword_query": "", "hyde_passage": ""},
        "branches": BRANCHES_5,
        "standalone": "獨資戶變更負責人的銷戶應如何辦理",
        "answer": "一、應備文件…[文獻 1]",
        "branch_answer": "若屬獨資戶變更負責人：…[文獻 1]",
    }
    responses.update(overrides)
    return FakeLLM(responses)


def test_picking_an_option_is_not_asked_again_even_when_session_is_lost():
    """回報的問題：選了選項之後又被問一模一樣的問題。

    最常見的成因是 session 遺失（前端沒帶回 session_id、TTL 過期、服務重啟），
    此時 asked_dimensions 防護整個失效。改由訊息內容本身判斷情境已指定。
    """
    retriever = FakeRetriever(lambda q, i: LONG_COMPETING)
    llm = multi_branch_llm()
    result, state, tracer = run_agent(
        retriever,
        llm,
        message="獨資戶變更負責人（獨資戶）",
        session=None,  # session 遺失
    )

    assert result.decision == Decision.ANSWER
    assert result.status == "completed"
    assert result.clarification_options == []
    assert state is None
    assert any(s.step == "branch_resolved_from_message" for s in tracer.steps)


def test_ambiguous_reply_still_triggers_clarification():
    """訊息同時命中多個分支時不可自行挑一個，仍要問。"""
    retriever = FakeRetriever(lambda q, i: LONG_COMPETING)
    llm = multi_branch_llm()
    result, state, _ = run_agent(retriever, llm, message="變更負責人", session=None)

    assert result.decision == Decision.CLARIFY
    assert state is not None
    assert state.asked_branch_labels == [b["label"] for b in BRANCHES_5["branches"]]


def test_renamed_dimension_does_not_cause_a_repeat_question():
    """第二輪 LLM 把維度改了名，但分支相同 -> 不再問，改為分情境全列。"""
    retriever = FakeRetriever(lambda q, i: LONG_COMPETING)
    llm = multi_branch_llm(
        branches={**BRANCHES_5, "dimension": "申請類型"},
        standalone="還是不確定的銷戶問題",
    )
    session = SessionState(
        original_query="銷戶要怎麼辦理？",
        clarification_question="請問您要辦理的屬於哪一種？",
        asked_dimensions=["銷戶申請方式"],
        asked_branch_labels=[b["label"] for b in BRANCHES_5["branches"]],
        clarification_count=1,
        status="awaiting_clarification",
    )
    result, state, _ = run_agent(retriever, llm, message="都不是", session=session)

    assert result.decision == Decision.ANSWER_ALL_BRANCHES
    assert state is None


# ==================== analyzer 失效時不要空轉、也不要硬判拒答 ====================
def test_no_new_variants_does_not_waste_a_retrieval_round():
    """analyzer 的 JSON 解析失敗時，變體只剩原問句，重跑必然拿到同一批結果。"""
    retriever = FakeRetriever(lambda q, i: COMPETING)
    llm = FakeLLM({
        "grade": {"verdict": "lexical_mismatch"},
        "analyze": "小模型吐了一段不是 JSON 的話",
        "branches": {
            "dimension": "業務類別",
            "branches": [
                {"label": "久未往來帳戶", "key_terms": [], "chunks": ["C1"]},
                {"label": "新開戶", "key_terms": [], "chunks": ["C2"]},
            ],
        },
        "branch_answer": "若屬久未往來帳戶：…[文獻 1]",
    })
    result, _, tracer = run_agent(retriever, llm)

    assert len(retriever.queries) == 1  # 沒有白跑第二次檢索
    assert any(s.step == "skip_retry" for s in tracer.steps)
    # 也不能因為 analyzer 壞了就判拒答——證據還在，交給後續規則決定
    assert result.decision != Decision.REFUSE


def test_analyzer_failure_with_competing_evidence_goes_to_branch_handling():
    retriever = FakeRetriever(lambda q, i: COMPETING)
    llm = FakeLLM({
        "grade": {"verdict": "lexical_mismatch"},
        "analyze": None,  # 呼叫失敗
        "branches": {
            "dimension": "業務類別",
            "branches": [
                {"label": "久未往來帳戶", "key_terms": [], "chunks": ["C1"]},
                {"label": "新開戶", "key_terms": [], "chunks": ["C2"]},
            ],
        },
        "branch_answer": "若屬久未往來帳戶：…[文獻 1]\n若屬新開戶：…[文獻 2]",
    })
    result, _, _ = run_agent(retriever, llm)

    assert result.decision == Decision.ANSWER_ALL_BRANCHES
    assert result.diagnosis == Diagnosis.MULTI_BRANCH


# ==================== 診斷說要改寫，就必須真的改寫 ====================
def test_score_veto_actually_triggers_a_rewrite(monkeypatch):
    """grader 說足夠、但分數未達門檻被否決時，必須真的改寫重試。

    回歸測試：原本 query 分析的觸發條件綁在「grader 說不足」上，
    而這條路徑的 grader 說的是「足夠」，導致分析被略過、沒有變體可用，
    於是 diagnose 的 reason 寫著「先改寫找更強證據」卻什麼都沒做就收斂。
    """
    monkeypatch.setattr(config, "THRESHOLDS_CALIBRATED", True)

    def responder(query, index):
        # 原問法只拿到 0.5073（未達 SCORE_ANSWERABLE=1.0），改寫後才拿到強命中
        if "正式術語" in query or "假想" in query or "關鍵詞" in query:
            return [item("遺留物處理.pdf", 0.86, "遺留物現金應於當日清點…")]
        return [item("遺留物處理.pdf", 0.5073, "相關但分數偏低的段落")]

    retriever = FakeRetriever(responder)
    llm = FakeLLM({
        "grade": {"verdict": "sufficient", "relevant_chunks": ["C1"]},
        "analyze": {
            "is_compound": False,
            "sub_questions": [],
            "rewritten_query": "遺留物品現金之處理正式術語",
            "keyword_query": "遺留物品 現金 關鍵詞",
            "hyde_passage": "假想條文：遺留物品之現金應…",
        },
        "answer": "一、清點現金…[文獻 1]",
    })
    result, _, tracer = run_agent(retriever, llm)

    assert result.decision == Decision.ANSWER
    # 確實跑了第二輪，而且用的是改寫後的問法
    assert len(retriever.queries) > 1
    assert any("正式術語" in q for q in retriever.queries)
    # 分析是在重試前補做的
    analyze_steps = [s for s in tracer.steps if s.step == "analyze_query"]
    assert analyze_steps
    assert analyze_steps[0].detail["trigger"] == "retry_needs_variants"


def test_happy_path_still_skips_query_analysis(monkeypatch):
    """證據充足時不該多花一次 LLM 呼叫做分析。"""
    monkeypatch.setattr(config, "THRESHOLDS_CALIBRATED", True)
    retriever = FakeRetriever(lambda q, i: STRONG_HIT)
    llm = FakeLLM({
        "grade": {"verdict": "sufficient", "relevant_chunks": ["C1"]},
        "answer": "一、…[文獻 1]",
    })
    result, _, _ = run_agent(retriever, llm)

    assert result.decision == Decision.ANSWER
    assert llm.calls == ["grade", "answer"]
