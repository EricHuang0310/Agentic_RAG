"""預算控制。

agent 迴圈最危險的失敗模式不是答錯，而是無止境地重試導致 p95 延遲爆掉。
每一次外部呼叫都必須先跟 Budget 請額度，額度用完就強制收斂到
反問或拒答，絕不允許「再試一輪看看」。
"""

import time
from dataclasses import dataclass, field

from . import config


class BudgetExhausted(RuntimeError):
    """額度用盡。呼叫端應該收斂到 fallback，而不是重試。"""


@dataclass
class Budget:
    max_retrieve_calls: int = config.MAX_RETRIEVE_CALLS
    max_llm_calls: int = config.MAX_LLM_CALLS
    max_iterations: int = config.MAX_ITERATIONS
    deadline_seconds: float = config.DEADLINE_SECONDS

    retrieve_calls: int = 0
    llm_calls: int = 0
    iterations: int = 0
    started_at: float = field(default_factory=time.monotonic)
    exhausted: bool = False

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.deadline_seconds - self.elapsed)

    def spend_retrieve(self, n: int = 1) -> None:
        if self.retrieve_calls + n > self.max_retrieve_calls or self.remaining_seconds <= 0:
            self.exhausted = True
            raise BudgetExhausted("retrieve 額度或時間預算已用盡")
        self.retrieve_calls += n

    def spend_llm(self, n: int = 1) -> None:
        if self.llm_calls + n > self.max_llm_calls or self.remaining_seconds <= 0:
            self.exhausted = True
            raise BudgetExhausted("LLM 額度或時間預算已用盡")
        self.llm_calls += n

    def can_afford_retrieve(self, n: int = 1) -> bool:
        return self.retrieve_calls + n <= self.max_retrieve_calls and self.remaining_seconds > 0

    def can_afford_llm(self, n: int = 1) -> bool:
        return self.llm_calls + n <= self.max_llm_calls and self.remaining_seconds > 0

    def can_start_iteration(self) -> bool:
        """是否還能開新的一輪 retrieve -> grade -> diagnose。"""
        if self.iterations >= self.max_iterations:
            return False
        # 新一輪至少需要 1 次檢索 + 1 次 grading
        return self.can_afford_retrieve(1) and self.can_afford_llm(1)

    def start_iteration(self) -> None:
        self.iterations += 1

    def snapshot(self) -> dict:
        return {
            "iterations": self.iterations,
            "retrieve_calls": self.retrieve_calls,
            "llm_calls": self.llm_calls,
            "elapsed_ms": int(self.elapsed * 1000),
            "exhausted": self.exhausted,
        }
