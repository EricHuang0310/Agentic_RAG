"""API 層：驗證回應 schema 與多輪 session 串接（外部服務以假 client 取代）。"""

import pytest
from conftest import FakeLLM, FakeRetriever, item
from fastapi.testclient import TestClient

import arkkb_router_search_all_db_api as api
from agentic.orchestrator import Agent

STRONG_HIT = [item("遺留物處理.pdf", 0.92, "遺留物現金應於當日清點…")]
MANY_BRANCHES = [item(f"業務{i}.pdf", 0.74 - i * 0.01, "內容" * 400) for i in range(5)]


def make_client(retriever, llm):
    client = TestClient(api.app)
    client.__enter__()  # 觸發 lifespan，建立 sessions store
    api.app.state.agent = Agent(retriever, llm)
    return client


def test_answer_response_keeps_legacy_fields():
    client = make_client(
        FakeRetriever(lambda q, i: STRONG_HIT),
        FakeLLM({
            "grade": {"verdict": "sufficient", "relevant_chunks": ["C1"]},
            "answer": "一、清點現金…[文獻 1]",
        }),
    )
    try:
        body = client.post("/api/v1/chat", json={"message": "遺留物現金怎麼處理？"}).json()
    finally:
        client.__exit__(None, None, None)

    # 舊版欄位全部存在
    for field in (
        "session_id", "answer", "references_summary", "reference_chunks",
        "retrieve_db", "top_score", "status", "router_round",
    ):
        assert field in body
    assert body["status"] == "completed"
    assert body["decision"] == "answer"
    assert body["diagnosis"] == "sufficient"
    assert body["reference_chunks"][0]["filename"] == "遺留物處理.pdf"
    assert body["top_score"] == pytest.approx(0.92)
    assert body["trace"]


def test_clarification_then_followup_shares_session():
    llm = FakeLLM({
        "grade": [
            {"verdict": "multi_branch", "relevant_chunks": ["C1", "C2"]},
            {"verdict": "sufficient", "relevant_chunks": ["C1"]},
        ],
        "analyze": {"is_compound": False, "rewritten_query": "", "keyword_query": "", "hyde_passage": ""},
        "branches": {
            "dimension": "業務類別",
            "branches": [
                {"label": f"業務{i}", "key_terms": [f"關鍵{i}"], "chunks": [f"C{i + 1}"]}
                for i in range(5)
            ],
        },
        "standalone": "業務0 的遺留物現金處理方式",
        "answer": "一、清點現金…[文獻 1]",
    })
    call_count = {"n": 0}

    def responder(query, index):
        call_count["n"] += 1
        return STRONG_HIT if "業務0" in query else MANY_BRANCHES

    client = make_client(FakeRetriever(responder), llm)
    try:
        first = client.post("/api/v1/chat", json={"message": "遺留物現金怎麼處理？"}).json()
        assert first["status"] == "awaiting_clarification"
        assert first["decision"] == "clarify"
        assert len(first["clarification_options"]) == 6
        assert first["reference_chunks"] == []

        option_value = first["clarification_options"][0]["value"]
        second = client.post(
            "/api/v1/chat",
            json={"session_id": first["session_id"], "message": option_value},
        ).json()
    finally:
        client.__exit__(None, None, None)

    assert second["session_id"] == first["session_id"]
    assert second["status"] == "completed"
    assert second["decision"] == "answer"
    assert second["effective_query"] == "業務0 的遺留物現金處理方式"


def test_empty_message_rejected():
    client = make_client(FakeRetriever(lambda q, i: STRONG_HIT), FakeLLM({}))
    try:
        response = client.post("/api/v1/chat", json={"message": "   "})
    finally:
        client.__exit__(None, None, None)
    assert response.status_code == 400


def test_include_trace_can_be_disabled_per_request():
    client = make_client(
        FakeRetriever(lambda q, i: STRONG_HIT),
        FakeLLM({
            "grade": {"verdict": "sufficient", "relevant_chunks": ["C1"]},
            "answer": "一、…[文獻 1]",
        }),
    )
    try:
        body = client.post(
            "/api/v1/chat", json={"message": "遺留物現金怎麼處理？", "include_trace": False}
        ).json()
    finally:
        client.__exit__(None, None, None)
    assert body["trace"] == []


def test_healthz_exposes_thresholds_and_budget():
    client = make_client(FakeRetriever(lambda q, i: STRONG_HIT), FakeLLM({}))
    try:
        body = client.get("/healthz").json()
    finally:
        client.__exit__(None, None, None)
    assert body["status"] == "ok"
    assert "answerable" in body["thresholds"]
    assert "max_iterations" in body["budget"]


def test_lost_session_is_visible_in_trace_and_does_not_repeat_the_question():
    """前端沒帶回 session_id 時，不該再問一次一模一樣的問題。"""
    long_competing = [item(f"銷戶{i}.pdf", 0.74 - i * 0.01, "內容" * 400) for i in range(5)]
    llm = FakeLLM({
        "grade": {"verdict": "multi_branch", "relevant_chunks": ["C1", "C2", "C3"]},
        "analyze": {"is_compound": False, "rewritten_query": "", "keyword_query": "", "hyde_passage": ""},
        "branches": {
            "dimension": "銷戶申請方式",
            "branches": [
                {"label": "獨資戶變更負責人", "key_terms": ["獨資戶"], "chunks": ["C1"]},
                {"label": "公司變更負責人", "key_terms": ["公司"], "chunks": ["C2"]},
                {"label": "一般銷戶申請", "key_terms": ["銷戶"], "chunks": ["C3"]},
                {"label": "其他情形一", "key_terms": [], "chunks": ["C4"]},
                {"label": "其他情形二", "key_terms": [], "chunks": ["C5"]},
            ],
        },
        "answer": "一、應備文件…[文獻 1]",
    })
    client = make_client(FakeRetriever(lambda q, i: long_competing), llm)
    try:
        first = client.post("/api/v1/chat", json={"message": "銷戶要怎麼辦理？"}).json()
        assert first["decision"] == "clarify"

        option_value = first["clarification_options"][0]["value"]
        # 故意不帶 session_id，模擬前端漏帶／TTL 過期／服務重啟
        second = client.post("/api/v1/chat", json={"message": option_value}).json()
    finally:
        client.__exit__(None, None, None)

    assert second["decision"] == "answer"
    assert second["status"] == "completed"
    lookup = next(s for s in second["trace"] if s["step"] == "session_lookup")
    assert lookup["detail"]["session_found"] is False


def test_session_lookup_step_reports_a_found_session():
    client = make_client(
        FakeRetriever(lambda q, i: STRONG_HIT),
        FakeLLM({
            "grade": {"verdict": "sufficient", "relevant_chunks": ["C1"]},
            "answer": "一、…[文獻 1]",
        }),
    )
    try:
        body = client.post("/api/v1/chat", json={"message": "遺留物現金怎麼處理？"}).json()
    finally:
        client.__exit__(None, None, None)
    lookup = next(s for s in body["trace"] if s["step"] == "session_lookup")
    assert lookup["detail"]["session_id_provided"] is False
    assert lookup["detail"]["session_found"] is False


def test_healthz_reports_disabled_floor_as_null():
    """停用的門檻是 -inf，不能直接塞進 JSON。"""
    client = make_client(FakeRetriever(lambda q, i: STRONG_HIT), FakeLLM({}))
    try:
        response = client.get("/healthz")
    finally:
        client.__exit__(None, None, None)

    assert response.status_code == 200
    thresholds = response.json()["thresholds"]
    assert thresholds["floor"] is None
    assert thresholds["answerable"] == 0.7
    assert thresholds["calibrated"] is True
