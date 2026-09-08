"""API 層：驗證回應 schema 與多輪 session 串接（外部服務以假 client 取代）。"""

import pytest
from conftest import FakeLLM, FakeRetriever, item
from fastapi.testclient import TestClient

import arkkb_router_search_all_db_api as api
from agentic.orchestrator import Agent

STRONG_HIT = [item("遺留物處理.pdf", 3.2, "遺留物現金應於當日清點…")]
MANY_BRANCHES = [item(f"業務{i}.pdf", 1.3 - i * 0.01, "內容" * 400) for i in range(5)]


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
    assert body["top_score"] == pytest.approx(3.2)
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
