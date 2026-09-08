"""RAG 反問／拒答 Agent 服務。

與舊版 v2.2.0 的差異
--------------------
舊版：`top_score < RERANK_SCORE_THRESHOLD` 就反問，且只在第一輪反問，
第二輪不論分數多低都硬答。這個設計把「問題模糊」與「語料庫沒有」
混成同一種失敗，前者該反問、後者反問也沒用。

這一版把決策改成明確的四種行動，反問降級為最後手段：

    retrieve → grade
      ├ 夠好 ───────────────────────────→ answer
      └ 不夠 → 診斷失敗類型
           (a) 措辭不匹配   → 改寫 / HyDE / 多變體 fan-out 重試
           (b) 複合問題     → 拆解成子問題各自檢索
           (c) 多分支互斥   → 分支 ≤3 且答案短 → answer_all_branches
                             分支多 / 答案長  → clarify（帶結構化選項）
           (d) 全面低分分散 → refuse（拒答 + 轉人工）
      預算耗盡 → clarify 或 refuse，絕不硬答

API 的回應保留舊版所有欄位，新欄位（decision / diagnosis /
clarification_options / signals / trace 等）都是附加的，舊前端可以不改。
"""

import logging
import uuid
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException

from agentic import config
from agentic.clients import LLMClient, RetrieveClient
from agentic.orchestrator import Agent
from agentic.schemas import ChatRequest, ChatResponse, ReferenceChunk
from agentic.sessions import InMemorySessionStore
from agentic.trace import Tracer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("agentic_rag.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 單一 AsyncClient 共用連線池；agent 迴圈會併發送出多個檢索請求
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=20)) as client:
        app.state.agent = Agent(RetrieveClient(client), LLMClient(client))
        app.state.sessions = InMemorySessionStore()
        logger.info(
            "服務啟動：kb=%s llm=%s 閾值(answerable=%s floor=%s supportive=%s) calibrated=%s",
            config.TARGET_KB_LIST,
            config.LLM_MODEL,
            config.SCORE_ANSWERABLE,
            config.SCORE_FLOOR,
            config.SCORE_SUPPORTIVE,
            config.THRESHOLDS_CALIBRATED,
        )
        if not config.THRESHOLDS_CALIBRATED:
            logger.warning(
                "分數門檻尚未校準（ARKKB_THRESHOLDS_CALIBRATED=0）："
                "不會單憑絕對分數判定拒答，把關全部交給 LLM grader。"
                "請先跑 scripts/probe_scores.py 確認分數尺度後再打開。"
            )
        yield


app = FastAPI(
    title="Agentic RAG Clarification / Refusal Service",
    description=(
        "以 retrieve → grade → diagnose 迴圈決定回答、分情境全列、反問或拒答的 RAG 服務。"
        "反問是最後手段，不是入口閘門。"
    ),
    version="3.0.0",
    lifespan=lifespan,
)


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_api(request: ChatRequest) -> ChatResponse:
    session_id = request.session_id or str(uuid.uuid4())
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="提問訊息不可為空")

    sessions: InMemorySessionStore = app.state.sessions
    agent: Agent = app.state.agent

    session = await sessions.get(session_id)
    tracer = Tracer(session_id)
    # session 遺失是「重複反問」最常見的原因（前端沒帶回 session_id、TTL 過期、
    # 服務重啟或多 worker），但它是靜默的。明確記一筆，讓 trace 看得出來。
    tracer.add(
        "session_lookup",
        session_id_provided=request.session_id is not None,
        session_found=session is not None,
        awaiting_clarification=bool(session and session.awaiting_clarification),
    )
    if request.session_id is not None and session is None:
        logger.warning(
            "帶了 session_id 但查無 session（可能是 TTL 過期、服務重啟或多 worker）: %s",
            session_id,
        )

    result, new_state = await agent.run(message, session, tracer)

    if new_state is not None:
        await sessions.set(session_id, new_state)
    else:
        await sessions.delete(session_id)

    include_trace = config.INCLUDE_TRACE if request.include_trace is None else request.include_trace

    return ChatResponse(
        session_id=session_id,
        answer=result.answer,
        references_summary=result.references_summary,
        reference_chunks=[
            ReferenceChunk(
                filename=c.filename,
                page=c.page,
                score=round(c.score, 4),
                content=c.content,
            )
            for c in result.chunks
        ],
        retrieve_db=config.TARGET_KB_LIST,
        top_score=result.signals.top1,
        status=result.status,
        router_round=result.round_label,
        decision=result.decision,
        diagnosis=result.diagnosis,
        clarification_options=result.clarification_options,
        effective_query=result.effective_query,
        signals=result.signals,
        iterations=result.iterations,
        budget_exhausted=result.budget_exhausted,
        trace=tracer.steps if include_trace else [],
    )


@app.get("/healthz")
async def healthz() -> dict:
    return {
        "status": "ok",
        "version": app.version,
        "kb_list": config.TARGET_KB_LIST,
        "active_sessions": await app.state.sessions.size(),
        "thresholds": {
            "calibrated": config.THRESHOLDS_CALIBRATED,
            "answerable": config.SCORE_ANSWERABLE,
            "supportive": config.SCORE_SUPPORTIVE,
            "floor": config.SCORE_FLOOR,
            "gap_ambiguous": config.GAP_AMBIGUOUS,
        },
        "budget": {
            "max_iterations": config.MAX_ITERATIONS,
            "max_retrieve_calls": config.MAX_RETRIEVE_CALLS,
            "max_llm_calls": config.MAX_LLM_CALLS,
            "deadline_seconds": config.DEADLINE_SECONDS,
        },
    }


# ==================== 啟動入口 (Port 9454) ====================
if __name__ == "__main__":
    uvicorn.run("arkkb_router_search_all_db_api:app", host="0.0.0.0", port=9454, reload=True)
