"""知識庫檢索與 LLM 的非同步用戶端。

改用 httpx.AsyncClient 而非 requests 的理由：agent 迴圈需要把多個
query 變體同時送去檢索（asyncio.gather），同步阻塞的 requests 會卡住
FastAPI 的 event loop，多變體 fan-out 也會退化成序列執行。
"""

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional

import httpx

from . import config
from .budget import Budget

logger = logging.getLogger("agentic_rag.clients")

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class RetrieveClient:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def retrieve(
        self,
        query: str,
        kb_list: List[str],
        budget: Budget,
        top_n: int = config.TOP_N,
        top_k: int = config.TOP_K,
    ) -> Optional[List[Dict[str, Any]]]:
        """向知識庫 API 檢索文件與重排。

        回傳 None 代表「呼叫失敗」，回傳 [] 代表「呼叫成功但沒有命中」。
        這兩者對診斷的意義完全不同（前者是服務問題，後者才是語料庫沒有），
        所以刻意分開，不要合併成空 list。
        """
        budget.spend_retrieve()
        payload = {
            "query": query,
            "kb_list": kb_list,
            "reranker_model": config.RERANKER_MODEL,
            "top_n": top_n,
            "kwargs": {"parent_vector": {"top_k": top_k}},
        }
        try:
            response = await self._client.post(
                config.RETRIEVE_URL,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=config.RETRIEVE_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # noqa: BLE001 - 外部服務任何失敗都只降級不中斷
            logger.warning("知識庫檢索 API 請求失敗: %s", exc)
            return None
        return data.get("data") or []


class LLMClient:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        budget: Budget,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> Optional[str]:
        """呼叫 LLM。失敗時回傳 None，由呼叫端決定如何降級。"""
        budget.spend_llm()
        payload: Dict[str, Any] = {
            "model": config.LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        # 不讓單一 LLM 呼叫吃掉整個請求的剩餘時間預算
        timeout = min(config.LLM_TIMEOUT, max(5.0, budget.remaining_seconds))
        try:
            response = await self._client.post(
                config.LLM_URL,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM API 請求失敗: %s", exc)
            return None

    async def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        budget: Budget,
        temperature: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """要求 LLM 回傳 JSON 並解析。解析失敗回傳 None（呼叫端須有保守 fallback）。"""
        raw = await self.chat(system_prompt, user_prompt, budget, temperature=temperature)
        if raw is None:
            return None
        return parse_json_object(raw)


def parse_json_object(raw: str) -> Optional[Dict[str, Any]]:
    """從 LLM 輸出中盡力抽出第一個 JSON object。

    小模型常在 JSON 外面包上說明文字或 ```json 圍欄，直接 json.loads 會失敗，
    所以先剝掉圍欄再退回正則抓取最外層大括號。
    """
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK.search(text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        logger.warning("無法解析 LLM 的 JSON 輸出: %s", text[:200])
        return None


async def gather_limited(coros, limit: int = 4):
    """有上限的併發執行，避免一次打爆後端服務。"""
    semaphore = asyncio.Semaphore(limit)

    async def _run(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*(_run(c) for c in coros), return_exceptions=True)
