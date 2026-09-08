"""預算與 session 狀態。"""

import asyncio
import time

import pytest

from agentic.budget import Budget, BudgetExhausted
from agentic.sessions import InMemorySessionStore, SessionState


def test_retrieve_budget_raises_when_exceeded():
    budget = Budget(max_retrieve_calls=2)
    budget.spend_retrieve()
    budget.spend_retrieve()
    with pytest.raises(BudgetExhausted):
        budget.spend_retrieve()
    assert budget.exhausted


def test_deadline_blocks_new_iterations():
    budget = Budget(deadline_seconds=0.0)
    assert not budget.can_start_iteration()
    assert not budget.can_afford_llm()


def test_iteration_cap():
    budget = Budget(max_iterations=2)
    budget.start_iteration()
    budget.start_iteration()
    assert not budget.can_start_iteration()


def test_session_roundtrip_and_delete():
    async def scenario():
        store = InMemorySessionStore()
        await store.set("s1", SessionState(original_query="Q", status="awaiting_clarification"))
        state = await store.get("s1")
        assert state is not None and state.awaiting_clarification
        await store.delete("s1")
        assert await store.get("s1") is None

    asyncio.run(scenario())


def test_session_expires_after_ttl():
    async def scenario():
        store = InMemorySessionStore(ttl_seconds=0.01)
        await store.set("s1", SessionState(original_query="Q"))
        time.sleep(0.02)
        assert await store.get("s1") is None
        assert await store.size() == 0

    asyncio.run(scenario())
