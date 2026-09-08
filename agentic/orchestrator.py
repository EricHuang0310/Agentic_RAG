"""Agent 主迴圈。

    retrieve → grade
      ├ 夠好 ───────────────────────────→ 回答
      └ 不夠 → 診斷失敗類型
           (a) 措辭不匹配   → 改寫 / HyDE / 多變體 fan-out 重試   ← 自己解決，不問
           (b) 複合問題     → 拆解成子問題各自檢索                ← 自己解決，不問
           (c) 多分支互斥   → 分支 ≤3 且答案短 → 分情境全部列出   ← 也不問
                             分支多 / 答案長  → 反問（帶選項）
           (d) 全面低分分散 → 拒答 + 轉人工                      ← 問了也沒用
      預算耗盡 → 反問或拒答，絕不硬答

反問在這裡是最後手段，而不是 pipeline 的入口閘門。
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from . import config
from .budget import Budget, BudgetExhausted
from .clients import LLMClient, RetrieveClient, gather_limited
from .diagnose import DiagnosisOutcome, diagnose, has_branch_evidence
from .fusion import parse_chunks, rrf_fuse
from .generate import (
    build_references_summary,
    generate_answer,
    generate_branch_answer,
)
from .grading import extract_branches, grade_chunks
from .policy import BranchAction, choose_branch_action, resolve_branch_from_message
from .query_ops import QueryAnalysis, analyze_query, rewrite_followup_to_standalone
from .schemas import (
    Branch,
    Chunk,
    ClarificationOption,
    Decision,
    Diagnosis,
    GradeResult,
    Signals,
)
from .sessions import SessionState
from .signals import compute_signals
from .trace import Tracer

logger = logging.getLogger("agentic_rag.orchestrator")


@dataclass
class AgentResult:
    decision: Decision
    diagnosis: Diagnosis
    answer: str
    effective_query: str
    chunks: List[Chunk] = field(default_factory=list)
    signals: Signals = field(default_factory=Signals)
    clarification_options: List[ClarificationOption] = field(default_factory=list)
    dimension: str = ""
    references_summary: str = ""
    iterations: int = 0
    budget_exhausted: bool = False
    round_label: str = ""

    @property
    def status(self) -> str:
        if self.decision == Decision.CLARIFY:
            return "awaiting_clarification"
        if self.decision == Decision.REFUSE:
            return "refused"
        return "completed"


class RetrievalPlan:
    """下一輪要怎麼檢索。"""

    ORIGINAL = "original"      # 只用原問句（第一輪）
    VARIANTS = "variants"      # 多變體 fan-out + RRF（處理措辭不匹配）
    DECOMPOSE = "decompose"    # 子問題各自檢索 + RRF（處理複合問題）


class Agent:
    def __init__(self, retriever: RetrieveClient, llm: LLMClient) -> None:
        self.retriever = retriever
        self.llm = llm

    async def run(
        self,
        message: str,
        session: Optional[SessionState],
        tracer: Tracer,
        budget: Optional[Budget] = None,
    ) -> Tuple[AgentResult, Optional[SessionState]]:
        """執行一次完整的 agent 流程。

        回傳 (結果, 要寫回的 session 狀態)。session 狀態為 None 代表該刪除。
        budget 可由外部注入，方便測試不同的預算上限。
        """
        budget = budget or Budget()
        is_followup = session is not None and session.awaiting_clarification

        # ---------- Step 0：query 正規化 ----------
        query = await self._normalize_query(message, session, budget, tracer)
        round_label = "反問後補充輪" if is_followup else "首輪提問"

        analysis: Optional[QueryAnalysis] = None
        plan = RetrievalPlan.ORIGINAL
        retried_variants = False
        decomposed = False

        chunks: List[Chunk] = []
        signals = Signals()
        grade = GradeResult()
        outcome = DiagnosisOutcome(Diagnosis.UNKNOWN, "尚未診斷")
        retrieve_failed = False

        # ---------- Agent 迴圈 ----------
        while budget.can_start_iteration():
            budget.start_iteration()

            try:
                chunks, retrieve_failed = await self._retrieve(
                    query, plan, analysis, budget, tracer
                )
            except BudgetExhausted:
                break

            signals = compute_signals(chunks)
            matched_variants = sorted({v for c in chunks for v in c.matched_variants})
            tracer.add(
                "signals",
                **signals.model_dump(),
                plan=plan,
                matched_variants=matched_variants,
            )

            if retrieve_failed and not chunks:
                return self._retrieve_failure_result(query, budget, round_label), None

            # ---------- grade ----------
            with tracer.timed("grade") as extra:
                try:
                    grade = await grade_chunks(query, chunks, self.llm, budget)
                except BudgetExhausted:
                    grade = GradeResult(verdict=Diagnosis.UNKNOWN, parse_ok=False)
                extra.update(grade.model_dump(exclude={"relevant_chunk_ids"}))

            # 第一輪判定不足時才做 query 分析，讓 happy path 只花 1 次檢索 + 1 次 grading。
            # grader 說證據足夠時先略過；若稍後診斷仍判定要改寫，會在重試前補做。
            if analysis is None and grade.verdict != Diagnosis.SUFFICIENT:
                analysis = await self._analyze(query, budget, tracer, trigger="grade_insufficient")

            # ---------- diagnose ----------
            outcome = diagnose(
                signals,
                grade,
                retried_variants=retried_variants,
                decomposed=decomposed,
                is_compound=analysis.is_compound if analysis else None,
            )
            tracer.add("diagnose", **outcome.as_detail())

            # ---------- 依診斷決定下一步 ----------
            if outcome.diagnosis == Diagnosis.SUFFICIENT:
                break

            if outcome.diagnosis == Diagnosis.COMPOUND and not decomposed:
                plan = RetrievalPlan.DECOMPOSE
                decomposed = True
                continue

            if outcome.diagnosis == Diagnosis.LEXICAL_MISMATCH and not retried_variants:
                # grader 說證據足夠、但分數未達門檻而被否決時，前面那一步會略過
                # query 分析，導致這裡沒有變體可用。診斷既然說要改寫就得真的改寫，
                # 否則會出現「reason 寫著先改寫找更強證據，卻什麼都沒做就收斂」。
                if analysis is None:
                    analysis = await self._analyze(
                        query, budget, tracer, trigger="retry_needs_variants"
                    )

                if analysis is not None and analysis.has_new_variants(query):
                    plan = RetrievalPlan.VARIANTS
                    retried_variants = True
                    continue

                # 沒有新的變體可試（多半是 analyzer 的 JSON 解析失敗）。
                # 拿同一個 query 再檢索一次只會拿到同一批結果，所以不重試，
                # 但也不能因此就判拒答——改為就地以「已重試」重新診斷，
                # 讓後續規則決定這到底是多分支還是真的沒有。
                tracer.add(
                    "skip_retry",
                    reason="analyzer 未產出新的 query 變體",
                    analysis_parse_ok=analysis.parse_ok if analysis else None,
                )
                retried_variants = True
                outcome = diagnose(
                    signals,
                    grade,
                    retried_variants=True,
                    decomposed=decomposed,
                    is_compound=analysis.is_compound if analysis else None,
                )
                tracer.add("diagnose", **outcome.as_detail())
                break

            # MULTI_BRANCH / OUT_OF_SCOPE，或已經重試過：跳出迴圈收斂
            break

        # 迴圈是因為預算/輪數上限而結束，且還沒得到終局診斷
        if outcome.diagnosis in (Diagnosis.LEXICAL_MISMATCH, Diagnosis.COMPOUND, Diagnosis.UNKNOWN):
            budget.exhausted = True
            tracer.add("budget_exhausted", **budget.snapshot(), diagnosis=outcome.diagnosis.value)
            return await self._fallback(
                message, query, chunks, signals, budget, session, tracer, round_label
            )

        # ---------- 依終局診斷產出結果 ----------
        if outcome.diagnosis == Diagnosis.SUFFICIENT:
            return await self._answer(query, chunks, signals, budget, tracer, round_label, outcome)

        if outcome.diagnosis == Diagnosis.MULTI_BRANCH:
            return await self._handle_multi_branch(
                message, query, chunks, signals, budget, session, tracer, round_label
            )

        return self._refuse(query, chunks, signals, budget, round_label, outcome.reason), None

    async def _analyze(
        self,
        query: str,
        budget: Budget,
        tracer: Tracer,
        trigger: str,
    ) -> Optional[QueryAnalysis]:
        """做一次 query 分析（術語改寫、關鍵詞、HyDE、複合問題拆解）。

        trigger 會記進 trace，用來區分是「grader 判定不足」時做的，
        還是「診斷判定要改寫但還沒分析過」時補做的。
        """
        if not budget.can_afford_llm():
            tracer.add("analyze_query", trigger=trigger, skipped="LLM 預算不足")
            return None

        with tracer.timed("analyze_query", trigger=trigger) as extra:
            try:
                analysis = await analyze_query(query, self.llm, budget)
            except BudgetExhausted:
                analysis = None
            if analysis:
                extra.update(
                    is_compound=analysis.is_compound,
                    sub_questions=analysis.sub_questions,
                    rewritten=analysis.rewritten_query,
                    keyword=analysis.keyword_query,
                    parse_ok=analysis.parse_ok,
                    has_new_variants=analysis.has_new_variants(query),
                )
        return analysis

    # ==================== Step 0 ====================
    async def _normalize_query(
        self,
        message: str,
        session: Optional[SessionState],
        budget: Budget,
        tracer: Tracer,
    ) -> str:
        """把使用者輸入轉成可直接檢索的 query。

        反問後的補充回答一律走 LLM 重構，不再用字串拼接。
        """
        if session is None or not session.awaiting_clarification:
            tracer.add("normalize_query", mode="passthrough", query=message)
            return message

        with tracer.timed("normalize_query", mode="standalone_rewrite") as extra:
            try:
                query = await rewrite_followup_to_standalone(
                    session.original_query,
                    session.clarification_question,
                    message,
                    self.llm,
                    budget,
                )
            except BudgetExhausted:
                query = f"{session.original_query}（補充條件：{message}）"
            extra.update(original=session.original_query, reply=message, rewritten=query)
        return query

    # ==================== 檢索 ====================
    async def _retrieve(
        self,
        query: str,
        plan: str,
        analysis: Optional[QueryAnalysis],
        budget: Budget,
        tracer: Tracer,
    ) -> Tuple[List[Chunk], bool]:
        """依 plan 執行檢索，回傳 (融合後的 chunks, 是否有呼叫失敗)。"""
        if plan == RetrievalPlan.VARIANTS and analysis is not None:
            variants = analysis.variants(query)
        elif plan == RetrievalPlan.DECOMPOSE and analysis is not None and analysis.sub_questions:
            variants = [(f"sub{i}", q) for i, q in enumerate(analysis.sub_questions, start=1)]
            # 子問題檢索時保留原問句，避免拆解拆錯導致完全失去主軸
            variants.insert(0, ("original", query))
        else:
            variants = [("original", query)]

        # 依剩餘額度裁切變體數量，寧可少跑幾個變體也不要中途爆掉
        affordable = max(1, budget.max_retrieve_calls - budget.retrieve_calls)
        if len(variants) > affordable:
            variants = variants[:affordable]

        with tracer.timed("retrieve", plan=plan) as extra:
            results = await gather_limited(
                [
                    self.retriever.retrieve(text, config.TARGET_KB_LIST, budget)
                    for _, text in variants
                ],
                limit=4,
            )

            variant_chunks: List[Tuple[str, List[Chunk]]] = []
            failed = False
            for (name, text), raw in zip(variants, results):
                if isinstance(raw, BaseException) or raw is None:
                    failed = True
                    continue
                variant_chunks.append((name, parse_chunks(raw)))

            fused = rrf_fuse(variant_chunks) if variant_chunks else []
            extra.update(
                variants=[name for name, _ in variants],
                queries={name: text[:80] for name, text in variants},
                n_fused=len(fused),
                any_call_failed=failed,
            )
        return fused, failed

    # ==================== 各終局分支 ====================
    async def _answer(
        self,
        query: str,
        chunks: Sequence[Chunk],
        signals: Signals,
        budget: Budget,
        tracer: Tracer,
        round_label: str,
        outcome: DiagnosisOutcome,
    ) -> Tuple[AgentResult, Optional[SessionState]]:
        with tracer.timed("generate_answer") as extra:
            try:
                generated = await generate_answer(query, chunks, self.llm, budget)
            except BudgetExhausted:
                generated = None
            if generated:
                extra.update(generated.as_detail())

        answer = generated.text if generated else "回答生成失敗，請重新提問或洽詢作業規範維護窗口。"
        label = f"{round_label}（證據充足，直接回答）"
        if outcome.low_confidence:
            label += "（低信心）"

        return (
            AgentResult(
                decision=Decision.ANSWER,
                diagnosis=Diagnosis.SUFFICIENT,
                answer=answer,
                effective_query=query,
                chunks=list(chunks),
                signals=signals,
                references_summary=build_references_summary(chunks),
                iterations=budget.iterations,
                budget_exhausted=budget.exhausted,
                round_label=label,
            ),
            None,
        )

    async def _handle_multi_branch(
        self,
        message: str,
        query: str,
        chunks: Sequence[Chunk],
        signals: Signals,
        budget: Budget,
        session: Optional[SessionState],
        tracer: Tracer,
        round_label: str,
    ) -> Tuple[AgentResult, Optional[SessionState]]:
        with tracer.timed("extract_branches") as extra:
            try:
                dimension, branches = await extract_branches(query, chunks, self.llm, budget)
            except BudgetExhausted:
                dimension, branches = "", []
            extra.update(dimension=dimension, branches=[b.label for b in branches])

        if not branches:
            # 判定為多分支卻抽不出分支：不硬答，改用通用反問把決定權交回使用者
            return self._generic_clarify(query, chunks, signals, budget, session, round_label)

        # 不依賴 session 的防護：使用者這句話若已經指定了其中一個分支，
        # 就直接回答那個分支。session 遺失（前端沒帶 session_id、TTL 過期、
        # 服務重啟）時，這是唯一還擋得住「問一模一樣問題」的機制。
        resolved = resolve_branch_from_message(message, branches)
        if resolved is not None:
            tracer.add(
                "branch_resolved_from_message",
                branch=resolved.label,
                had_session=session is not None,
            )
            scoped = [c for c in chunks if c.chunk_id in set(resolved.chunk_ids)] or list(chunks)
            return await self._answer(
                query,
                scoped,
                signals,
                budget,
                tracer,
                f"{round_label}（訊息已指定情境「{resolved.label}」）",
                DiagnosisOutcome(Diagnosis.SUFFICIENT, f"使用者訊息已指定分支：{resolved.label}"),
            )

        action = choose_branch_action(
            branches,
            chunks,
            dimension,
            asked_dimensions=set(session.asked_dimensions) if session else set(),
            asked_branch_labels=session.asked_branch_labels if session else None,
            clarification_count=session.clarification_count if session else 0,
        )
        tracer.add("branch_policy", **action.as_detail())

        if action.decision == Decision.ANSWER_ALL_BRANCHES:
            return await self._answer_all_branches(
                query, chunks, branches, dimension, signals, budget, tracer, round_label
            )

        if action.decision == Decision.CLARIFY:
            return self._clarify(query, chunks, signals, budget, session, round_label, action)

        return (
            self._refuse(query, chunks, signals, budget, round_label, action.reason),
            None,
        )

    async def _answer_all_branches(
        self,
        query: str,
        chunks: Sequence[Chunk],
        branches: Sequence[Branch],
        dimension: str,
        signals: Signals,
        budget: Budget,
        tracer: Tracer,
        round_label: str,
    ) -> Tuple[AgentResult, Optional[SessionState]]:
        with tracer.timed("generate_branch_answer") as extra:
            try:
                generated = await generate_branch_answer(
                    query, chunks, branches, dimension, self.llm, budget
                )
            except BudgetExhausted:
                generated = None
            if generated:
                extra.update(generated.as_detail())

        if not generated:
            # 生成失敗時退回反問，不要留下空答案
            action = BranchAction(
                Decision.CLARIFY,
                "分情境回答生成失敗，退回反問",
                clarification_question=config.GENERIC_CLARIFICATION,
                dimension=dimension,
            )
            return self._clarify(query, chunks, signals, budget, None, round_label, action)

        return (
            AgentResult(
                decision=Decision.ANSWER_ALL_BRANCHES,
                diagnosis=Diagnosis.MULTI_BRANCH,
                answer=generated.text,
                effective_query=query,
                chunks=list(chunks),
                signals=signals,
                dimension=dimension,
                references_summary=build_references_summary(chunks),
                iterations=budget.iterations,
                budget_exhausted=budget.exhausted,
                round_label=f"{round_label}（多分支，分情境全部列出，共 {len(branches)} 種）",
            ),
            None,
        )

    def _clarify(
        self,
        query: str,
        chunks: Sequence[Chunk],
        signals: Signals,
        budget: Budget,
        session: Optional[SessionState],
        round_label: str,
        action: BranchAction,
    ) -> Tuple[AgentResult, SessionState]:
        new_state = SessionState(
            original_query=query,
            clarification_question=action.clarification_question,
            asked_dimensions=list(session.asked_dimensions) if session else [],
            asked_branch_labels=list(session.asked_branch_labels) if session else [],
            clarification_count=(session.clarification_count if session else 0) + 1,
            status="awaiting_clarification",
        )
        if action.dimension and action.dimension not in new_state.asked_dimensions:
            new_state.asked_dimensions.append(action.dimension)
        for label in action.branch_labels:
            if label not in new_state.asked_branch_labels:
                new_state.asked_branch_labels.append(label)

        return (
            AgentResult(
                decision=Decision.CLARIFY,
                diagnosis=Diagnosis.MULTI_BRANCH,
                answer=action.clarification_question,
                effective_query=query,
                # 反問時不回傳 chunks，避免同仁在還沒確認情境前就照著讀
                chunks=[],
                signals=signals,
                clarification_options=action.options,
                dimension=action.dimension,
                references_summary="",
                iterations=budget.iterations,
                budget_exhausted=budget.exhausted,
                round_label=f"{round_label}（多分支且不宜全列，觸發反問）",
            ),
            new_state,
        )

    def _generic_clarify(
        self,
        query: str,
        chunks: Sequence[Chunk],
        signals: Signals,
        budget: Budget,
        session: Optional[SessionState],
        round_label: str,
    ) -> Tuple[AgentResult, Optional[SessionState]]:
        """沒有分支資訊時的最後手段反問。"""
        count = session.clarification_count if session else 0
        if count >= config.MAX_CLARIFICATIONS:
            return (
                self._refuse(
                    query, chunks, signals, budget, round_label, "反問次數已達上限且仍無法確定情境"
                ),
                None,
            )
        action = BranchAction(
            Decision.CLARIFY,
            "訊號顯示問題模糊但無法抽出具體分支",
            clarification_question=config.GENERIC_CLARIFICATION,
        )
        return self._clarify(query, chunks, signals, budget, session, round_label, action)

    def _refuse(
        self,
        query: str,
        chunks: Sequence[Chunk],
        signals: Signals,
        budget: Budget,
        round_label: str,
        reason: str,
    ) -> AgentResult:
        return AgentResult(
            decision=Decision.REFUSE,
            diagnosis=Diagnosis.OUT_OF_SCOPE,
            answer=config.REFUSAL_MESSAGE,
            effective_query=query,
            # 拒答時仍回傳 chunks，方便維運人員判斷是語料缺漏還是檢索問題
            chunks=list(chunks),
            signals=signals,
            references_summary=build_references_summary(chunks),
            iterations=budget.iterations,
            budget_exhausted=budget.exhausted,
            round_label=f"{round_label}（拒答：{reason}）",
        )

    def _retrieve_failure_result(
        self, query: str, budget: Budget, round_label: str
    ) -> AgentResult:
        return AgentResult(
            decision=Decision.REFUSE,
            diagnosis=Diagnosis.UNKNOWN,
            answer=config.RETRIEVE_FAILED_MESSAGE,
            effective_query=query,
            iterations=budget.iterations,
            budget_exhausted=budget.exhausted,
            round_label=f"{round_label}（知識庫連線失敗）",
        )

    async def _fallback(
        self,
        message: str,
        query: str,
        chunks: Sequence[Chunk],
        signals: Signals,
        budget: Budget,
        session: Optional[SessionState],
        tracer: Tracer,
        round_label: str,
    ) -> Tuple[AgentResult, Optional[SessionState]]:
        """預算耗盡時的收斂：反問或拒答，絕不硬答。

        若訊號顯示「有證據但互相競爭」，代表問題確實模糊，反問是有意義的；
        若訊號顯示證據普遍不足，反問只會浪費使用者一輪，直接拒答。
        """
        if has_branch_evidence(signals):
            if budget.can_afford_llm() and chunks:
                return await self._handle_multi_branch(
                    message, query, chunks, signals, budget, session, tracer, round_label
                )
            return self._generic_clarify(query, chunks, signals, budget, session, round_label)

        return (
            self._refuse(query, chunks, signals, budget, round_label, "預算耗盡且證據不足"),
            None,
        )
