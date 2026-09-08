"""LLM grader 與分支抽取。

設計原則是「LLM 提議、數值訊號仲裁」：grader 只負責讀懂內容並提出判斷，
最終診斷由 diagnose.py 結合分數訊號決定。這樣可以避免小模型過度自信地
說「證據足夠」（LLM 對自己的檢索結果評估是有偏的），也避免純分數
無法區分模糊與無資料的問題。
"""

import logging
from typing import Dict, List, Sequence, Tuple

from .budget import Budget
from .clients import LLMClient
from .schemas import Branch, Chunk, Diagnosis, GradeResult

logger = logging.getLogger("agentic_rag.grading")

# grading 用的 chunk 內容截斷長度。grader 只需判斷相關性，不需要完整條文。
GRADE_CONTENT_LIMIT = 600

_GRADE_SYSTEM = (
    "你是內部作業流程知識庫的檢索品質評估器。給你一個使用者問題與數段檢索到的規範內容，\n"
    "請判斷這些內容是否足以完整回答問題，並指出失敗類型。\n\n"
    "請「只」輸出一個 JSON 物件，格式如下：\n"
    "{\n"
    '  "verdict": "sufficient" 或 "lexical_mismatch" 或 "multi_branch" 或 "not_in_corpus",\n'
    '  "relevant_chunks": ["C1", "C3"],\n'
    '  "missing_information": "若不足，說明缺少什麼資訊",\n'
    '  "reasoning": "一句話說明判斷依據"\n'
    "}\n\n"
    "verdict 的判斷標準：\n"
    '- "sufficient"：內容明確且完整涵蓋問題所問的流程或規定，可以直接據此作答。\n'
    '- "lexical_mismatch"：內容的主題與問題相關，但檢索到的段落偏離重點，'
    "看得出知識庫應該有更貼切的章節只是沒被找出來（通常是使用者用口語、"
    "而規範用正式術語）。\n"
    '- "multi_branch"：檢索到多個彼此互斥的作業情境（例如不同業務類別、不同通路、'
    "不同客戶身分），每一個都可能是使用者要問的，必須知道使用者屬於哪一種才能正確回答。\n"
    '- "not_in_corpus"：這些內容與問題無關，知識庫看起來沒有涵蓋這個主題。\n\n'
    "重要：不要因為內容看起來很像就判定 sufficient。若問題沒有指明情境而內容分屬不同情境，"
    "應判為 multi_branch。使用繁體中文。"
)

_BRANCH_SYSTEM = (
    "你是內部作業流程分析器。給你一個使用者問題與數段規範內容，這些內容分屬不同的作業情境。\n"
    "請找出彼此互斥的情境分支，也就是「必須先知道使用者屬於哪一種，才能給出正確流程」的區辨維度。\n\n"
    "請「只」輸出一個 JSON 物件，格式如下：\n"
    "{\n"
    '  "dimension": "這些分支共同的區辨維度名稱，例如「業務類別」或「辦理通路」",\n'
    '  "branches": [\n'
    '    {"label": "分支名稱，例如「久未往來帳戶」", "key_terms": ["關鍵詞"], "chunks": ["C1"]}\n'
    "  ]\n"
    "}\n\n"
    "規則：\n"
    "1. 分支必須真的互斥，只是同一流程的不同步驟不算分支。\n"
    "2. label 要用使用者看得懂的說法，不要用檔名或章節編號。\n"
    "3. 最多列出 5 個分支。\n"
    "4. 使用繁體中文。"
)

_VERDICT_MAP = {
    "sufficient": Diagnosis.SUFFICIENT,
    "lexical_mismatch": Diagnosis.LEXICAL_MISMATCH,
    "multi_branch": Diagnosis.MULTI_BRANCH,
    "not_in_corpus": Diagnosis.OUT_OF_SCOPE,
}


def format_chunks_for_llm(chunks: Sequence[Chunk], limit: int = GRADE_CONTENT_LIMIT) -> Tuple[str, Dict[str, str]]:
    """把 chunks 編號成 C1, C2... 供 LLM 引用，並回傳編號到 chunk_id 的對照表。"""
    lines: List[str] = []
    ref_map: Dict[str, str] = {}
    for index, chunk in enumerate(chunks, start=1):
        ref = f"C{index}"
        ref_map[ref] = chunk.chunk_id
        content = chunk.content[:limit]
        if len(chunk.content) > limit:
            content += "…（後略）"
        page = f" 頁次：{chunk.page}" if chunk.page else ""
        lines.append(f"[{ref}] 來源：{chunk.filename}{page}\n{content}")
    return "\n\n".join(lines), ref_map


async def grade_chunks(
    query: str,
    chunks: Sequence[Chunk],
    llm: LLMClient,
    budget: Budget,
) -> GradeResult:
    """評估檢索結果是否足以回答問題。無法取得判斷時回傳 UNKNOWN，交由數值訊號決定。"""
    if not chunks:
        return GradeResult(verdict=Diagnosis.OUT_OF_SCOPE, reasoning="檢索結果為空")
    if not budget.can_afford_llm():
        return GradeResult(verdict=Diagnosis.UNKNOWN, reasoning="LLM 預算不足，略過 grading", parse_ok=False)

    formatted, ref_map = format_chunks_for_llm(chunks)
    user_prompt = f"【使用者問題】：\n{query}\n\n【檢索到的內容】：\n{formatted}"
    data = await llm.chat_json(_GRADE_SYSTEM, user_prompt, budget)
    if not data:
        return GradeResult(verdict=Diagnosis.UNKNOWN, reasoning="grader 輸出無法解析", parse_ok=False)

    verdict = _VERDICT_MAP.get(str(data.get("verdict", "")).strip().lower(), Diagnosis.UNKNOWN)
    relevant_ids = [
        ref_map[str(ref).strip().upper()]
        for ref in (data.get("relevant_chunks") or [])
        if str(ref).strip().upper() in ref_map
    ]
    return GradeResult(
        verdict=verdict,
        relevant_chunk_ids=relevant_ids,
        missing_information=str(data.get("missing_information") or "").strip()[:500],
        reasoning=str(data.get("reasoning") or "").strip()[:500],
        parse_ok=True,
    )


async def extract_branches(
    query: str,
    chunks: Sequence[Chunk],
    llm: LLMClient,
    budget: Budget,
) -> Tuple[str, List[Branch]]:
    """抽出互斥的情境分支。回傳 (區辨維度名稱, 分支清單)。"""
    if not chunks or not budget.can_afford_llm():
        return "", []

    formatted, ref_map = format_chunks_for_llm(chunks)
    user_prompt = f"【使用者問題】：\n{query}\n\n【檢索到的內容】：\n{formatted}"
    data = await llm.chat_json(_BRANCH_SYSTEM, user_prompt, budget)
    if not data:
        return "", []

    dimension = str(data.get("dimension") or "").strip()[:100]
    branches: List[Branch] = []
    for raw in (data.get("branches") or [])[:5]:
        if not isinstance(raw, dict):
            continue
        label = str(raw.get("label") or "").strip()[:100]
        if not label:
            continue
        key_terms = [str(t).strip()[:50] for t in (raw.get("key_terms") or []) if str(t).strip()][:5]
        chunk_ids = [
            ref_map[str(ref).strip().upper()]
            for ref in (raw.get("chunks") or [])
            if str(ref).strip().upper() in ref_map
        ]
        branches.append(Branch(label=label, key_terms=key_terms, chunk_ids=chunk_ids))

    # 只有一個分支代表其實沒有歧異，不該當作 multi_branch 處理
    return dimension, branches if len(branches) >= 2 else []
