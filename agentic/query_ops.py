"""Query 理解與改寫。

兩個重點：

1. 多輪對話不再用字串拼接。舊版把 `原問題 + "\\n補充說明：" + 補充` 直接
   送進 reranker，語意常常是壞的；這裡改成由 LLM 重構出一個獨立可檢索的
   standalone query。
2. 變體產生集中在一次 LLM 呼叫。同一次回傳 SOP 術語改寫、關鍵詞查詢、
   HyDE 假想條文與複合問題拆解，避免每種改寫各花一次呼叫。
"""

import logging
from typing import List, Optional

from pydantic import BaseModel, Field

from .budget import Budget
from .clients import LLMClient

logger = logging.getLogger("agentic_rag.query_ops")


class QueryAnalysis(BaseModel):
    """對一個問題的分析結果。"""

    is_compound: bool = False
    sub_questions: List[str] = Field(default_factory=list)
    rewritten_query: str = ""
    keyword_query: str = ""
    hyde_passage: str = ""
    parse_ok: bool = True

    def variants(self, original: str, include_hyde: bool = True) -> List[tuple]:
        """組出 [(變體名稱, query)] 清單，已去除空值與重複。"""
        candidates = [
            ("original", original),
            ("rewritten", self.rewritten_query),
            ("keyword", self.keyword_query),
        ]
        if include_hyde:
            candidates.append(("hyde", self.hyde_passage))

        seen = set()
        result = []
        for name, text in candidates:
            text = (text or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            result.append((name, text))
        return result


_STANDALONE_SYSTEM = (
    "你是一個查詢重構器。使用者正在與內部作業流程助手對話，助手上一輪反問了他一個問題。\n"
    "你的任務是把「原始問題」與「使用者的補充回答」合併成一個**獨立、完整、可直接用於文件檢索**的問句。\n\n"
    "規則：\n"
    "1. 只輸出重構後的問句本身，不要有任何前綴、解釋或標點以外的符號。\n"
    "2. 必須把補充回答提供的條件明確寫進問句中，不要用「上述」、「該」等指代詞。\n"
    "3. 不要新增使用者沒有提到的條件，也不要自行縮小範圍。\n"
    "4. 使用繁體中文。"
)

_ANALYZE_SYSTEM = (
    "你是內部作業流程知識庫的檢索前處理器。針對使用者的問題，產出有助於文件檢索的多種表述。\n\n"
    "請「只」輸出一個 JSON 物件，不要有任何其他文字，格式如下：\n"
    "{\n"
    '  "is_compound": true 或 false,\n'
    '  "sub_questions": ["若 is_compound 為 true，列出可獨立檢索的子問題；否則給空陣列"],\n'
    '  "rewritten_query": "把口語說法改寫成內部作業規範可能使用的正式術語後的問句",\n'
    '  "keyword_query": "只保留關鍵名詞與專有名詞的查詢字串，以空白分隔",\n'
    '  "hyde_passage": "假想這份作業規範中回答此問題的段落，用規範文件的語氣寫 2 至 4 句"\n'
    "}\n\n"
    "判斷 is_compound 的標準：問題是否包含兩個以上彼此獨立、需要分別查找不同章節才能回答的訴求。\n"
    "只是同一件事的條件描述不算複合問題。\n"
    "所有輸出使用繁體中文。"
)


async def rewrite_followup_to_standalone(
    original_query: str,
    clarification_question: str,
    user_reply: str,
    llm: LLMClient,
    budget: Budget,
) -> str:
    """把反問後的補充回答重構成獨立問句。失敗時退回舊版的字串拼接。"""
    fallback = f"{original_query}（補充條件：{user_reply}）"
    if not budget.can_afford_llm():
        return fallback

    user_prompt = (
        f"【原始問題】：{original_query}\n"
        f"【助手的反問】：{clarification_question}\n"
        f"【使用者的補充回答】：{user_reply}\n\n"
        "重構後的問句："
    )
    result = await llm.chat(_STANDALONE_SYSTEM, user_prompt, budget)
    if not result:
        return fallback
    rewritten = result.strip().split("\n")[0].strip()
    # 過短或明顯失控的輸出不採用，避免比拼接更糟
    if len(rewritten) < 4 or len(rewritten) > 300:
        return fallback
    return rewritten


async def analyze_query(
    query: str,
    llm: LLMClient,
    budget: Budget,
) -> QueryAnalysis:
    """一次呼叫取得改寫、關鍵詞、HyDE 與複合問題拆解。失敗時回傳只含原問題的保守結果。"""
    if not budget.can_afford_llm():
        return QueryAnalysis(parse_ok=False)

    data = await llm.chat_json(_ANALYZE_SYSTEM, f"使用者問題：{query}", budget)
    if not data:
        return QueryAnalysis(parse_ok=False)

    sub_questions = [
        str(q).strip()
        for q in (data.get("sub_questions") or [])
        if isinstance(q, (str, int, float)) and str(q).strip()
    ]
    is_compound = bool(data.get("is_compound")) and len(sub_questions) >= 2

    return QueryAnalysis(
        is_compound=is_compound,
        sub_questions=sub_questions[:4] if is_compound else [],
        rewritten_query=_as_text(data.get("rewritten_query")),
        keyword_query=_as_text(data.get("keyword_query")),
        hyde_passage=_as_text(data.get("hyde_passage")),
        parse_ok=True,
    )


def _as_text(value: Optional[object], limit: int = 800) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text[:limit]
