"""測試用的假 client。

外部端點是內網服務，測試不連線，改用可腳本化的假 client 驗證 agent
的決策路徑。假 LLM 依 system prompt 的特徵字串判斷這次是哪一種呼叫。
"""

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic.budget import Budget  # noqa: E402
from agentic.clients import parse_json_object  # noqa: E402

# system prompt 的特徵字串 -> 呼叫類型
_PROMPT_KINDS = [
    ("檢索品質評估器", "grade"),
    ("檢索前處理器", "analyze"),
    ("作業流程分析器", "branches"),
    ("查詢重構器", "standalone"),
    ("分情境作答", "branch_answer"),
    ("內部作業流程助手", "answer"),
]


def classify_prompt(system_prompt: str) -> str:
    for marker, kind in _PROMPT_KINDS:
        if marker in system_prompt:
            return kind
    return "unknown"


class FakeLLM:
    """依呼叫類型回傳腳本化內容。

    responses 的值可以是：
      - 字串 / dict（dict 會被序列化成 JSON）
      - list：依該類型的呼叫次數依序取用，用完後沿用最後一個
      - callable(user_prompt) -> 字串 / dict
      - None：模擬呼叫失敗
    """

    def __init__(self, responses: Dict[str, Any]) -> None:
        self.responses = responses
        self.calls: List[str] = []
        self.kind_counts: Dict[str, int] = {}

    async def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        budget: Budget,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> Optional[str]:
        budget.spend_llm()
        kind = classify_prompt(system_prompt)
        self.calls.append(kind)
        index = self.kind_counts.get(kind, 0)
        self.kind_counts[kind] = index + 1
        value = self.responses.get(kind)
        if isinstance(value, list):
            value = value[min(index, len(value) - 1)] if value else None
        if callable(value):
            value = value(user_prompt)
        if value is None:
            return None
        if isinstance(value, dict):
            import json

            return json.dumps(value, ensure_ascii=False)
        return str(value)

    async def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        budget: Budget,
        temperature: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        raw = await self.chat(system_prompt, user_prompt, budget)
        if raw is None:
            return None
        return parse_json_object(raw)


class FakeRetriever:
    """responder(query, call_index) -> 原始 item list；回傳 None 模擬呼叫失敗。"""

    def __init__(self, responder: Callable[[str, int], Optional[List[dict]]]) -> None:
        self.responder = responder
        self.queries: List[str] = []

    async def retrieve(self, query: str, kb_list, budget: Budget, top_n=6, top_k=66):
        budget.spend_retrieve()
        index = len(self.queries)
        self.queries.append(query)
        return self.responder(query, index)


def item(filename: str, score: float, content: str = "內容", page: str = "1") -> dict:
    """組出知識庫 API 的原始 item（分數放在 kwargs.relevance_score，與真實回應一致）。"""
    return {
        "filename": filename,
        "page": page,
        "content": content,
        "kwargs": {"relevance_score": score},
    }
