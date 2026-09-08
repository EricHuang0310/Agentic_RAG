"""對話狀態儲存。

舊版是模組層級的 `CHAT_SESSIONS = {}`，有三個問題：沒有 TTL 會無限成長、
uvicorn 多 worker 時每個 worker 各有一份導致第二輪找不到 session、
`reload=True` 重啟就全部清空。

這裡先做成有 TTL 與鎖的記憶體實作，但介面刻意保持最小，
正式環境要換 Redis 只需要再寫一個同介面的 RedisSessionStore
（set / get / delete 三個方法）並在 app 啟動時替換。
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import config


@dataclass
class SessionState:
    """一次多輪對話的狀態。"""

    original_query: str = ""
    # 上一輪反問的問題本文，用來重構 standalone query
    clarification_question: str = ""
    # 已經問過的區辨維度，避免換句話問同一件事
    asked_dimensions: List[str] = field(default_factory=list)
    clarification_count: int = 0
    status: str = "active"
    updated_at: float = field(default_factory=time.time)

    @property
    def awaiting_clarification(self) -> bool:
        return self.status == "awaiting_clarification"


class InMemorySessionStore:
    def __init__(self, ttl_seconds: float = config.SESSION_TTL_SECONDS) -> None:
        self._ttl = ttl_seconds
        self._data: Dict[str, SessionState] = {}
        self._lock = asyncio.Lock()

    async def get(self, session_id: str) -> Optional[SessionState]:
        async with self._lock:
            self._sweep()
            state = self._data.get(session_id)
            if state is None:
                return None
            if self._expired(state):
                self._data.pop(session_id, None)
                return None
            return state

    async def set(self, session_id: str, state: SessionState) -> None:
        state.updated_at = time.time()
        async with self._lock:
            self._sweep()
            self._data[session_id] = state

    async def delete(self, session_id: str) -> None:
        async with self._lock:
            self._data.pop(session_id, None)

    async def size(self) -> int:
        async with self._lock:
            self._sweep()
            return len(self._data)

    def _expired(self, state: SessionState) -> bool:
        return time.time() - state.updated_at > self._ttl

    def _sweep(self) -> None:
        expired = [sid for sid, state in self._data.items() if self._expired(state)]
        for sid in expired:
            self._data.pop(sid, None)
