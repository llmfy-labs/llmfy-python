"""
Checkpointing example using InMemoryCheckpointer — session continuation,
reducer merge across resumed calls, checking state, and resetting a
session. See flowengine_sql_checkpointer_example.py /
flowengine_redis_checkpointer_example.py for persistent backends.
"""

import asyncio
from typing import Annotated, TypedDict

from llmfy import END, START, FlowEngine, InMemoryCheckpointer


def append_log(old: list[str] | None, new: list[str]) -> list[str]:
    return (old or []) + new


class AppState(TypedDict):
    log: Annotated[list[str], append_log]
    step: int


def build_flow(checkpointer: InMemoryCheckpointer) -> FlowEngine:
    async def step_node(state: AppState) -> dict:
        step = state.get("step", 0) + 1
        return {"log": [f"step {step}"], "step": step}

    flow = FlowEngine(AppState, checkpointer=checkpointer)
    flow.add_node("step", step_node)
    flow.add_edge(START, "step")
    flow.add_edge("step", END)
    return flow.build()


async def example_resume_continues_from_checkpoint():
    print("\n--- Resume continues from the last checkpoint ---")
    checkpointer = InMemoryCheckpointer()
    flow = build_flow(checkpointer)
    session_id = "user-123"

    result_1 = await flow.invoke(apply_state={"log": [], "step": 0}, session_id=session_id)
    print(f"Run 1: {result_1}")

    # A workflow that already reached END restarts from START on the next
    # call with the same session_id — apply_state merges via each field's
    # reducer (log appends, step replaces).
    result_2 = await flow.invoke(apply_state={"log": ["resumed"]}, session_id=session_id)
    print(f"Run 2: {result_2}")


async def example_get_state_and_reset():
    print("\n--- get_state() and reset_session() ---")
    checkpointer = InMemoryCheckpointer()
    flow = build_flow(checkpointer)
    session_id = "user-456"

    print(f"Before any run: {await flow.get_state(session_id)}")

    await flow.invoke(apply_state={"log": [], "step": 0}, session_id=session_id)
    print(f"After one run: {await flow.get_state(session_id)}")

    await flow.reset_session(session_id)
    print(f"After reset_session(): {await flow.get_state(session_id)}")


async def example_independent_sessions():
    print("\n--- Different session_ids never share state ---")
    checkpointer = InMemoryCheckpointer()
    flow = build_flow(checkpointer)

    result_a = await flow.invoke(apply_state={"log": [], "step": 0}, session_id="a")
    result_b = await flow.invoke(apply_state={"log": [], "step": 0}, session_id="b")
    print(f"session a: {result_a}")
    print(f"session b: {result_b}")


async def main():
    await example_resume_continues_from_checkpoint()
    await example_get_state_and_reset()
    await example_independent_sessions()


if __name__ == "__main__":
    asyncio.run(main())
