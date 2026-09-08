"""執行軌跡記錄。

沒有 trace 就沒有辦法回答「為什麼這一題反問了」，閾值也就永遠只能憑感覺調。
每一步都記下輸入摘要、輸出摘要與耗時，正式環境即使不回傳給前端，
也應該完整寫進 log 或觀測系統。
"""

import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, List

from .schemas import TraceStep

logger = logging.getLogger("agentic_rag.trace")


class Tracer:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.steps: List[TraceStep] = []

    def add(self, step: str, latency_ms: int = 0, **detail: Any) -> None:
        self.steps.append(TraceStep(step=step, latency_ms=latency_ms, detail=detail))
        logger.info(
            "[%s] %s (%dms) %s", self.session_id, step, latency_ms, _compact(detail)
        )

    @contextmanager
    def timed(self, step: str, **detail: Any):
        """量測一個步驟的耗時；區塊內可以往 extra 塞補充資訊。"""
        started = time.monotonic()
        extra: Dict[str, Any] = {}
        try:
            yield extra
        finally:
            latency_ms = int((time.monotonic() - started) * 1000)
            self.add(step, latency_ms=latency_ms, **{**detail, **extra})


def _compact(detail: Dict[str, Any], limit: int = 400) -> str:
    text = ", ".join(f"{k}={v!r}" for k, v in detail.items())
    return text if len(text) <= limit else text[:limit] + "..."
