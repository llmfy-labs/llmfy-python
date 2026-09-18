"""
Basic FlowEngine example: a linear node, a conditional loop, and a reducer
on the state schema. No LLM/provider required — runs standalone.
"""

import asyncio
from typing import Annotated, TypedDict

from llmfy import END, START, FlowEngine


def append_log(old: list[str] | None, new: list[str]) -> list[str]:
    """Reducer: concatenate log entries instead of replacing them."""
    return (old or []) + new


class AppState(TypedDict):
    log: Annotated[list[str], append_log]
    counter: int


async def start_node(state: AppState) -> dict:
    return {"log": ["started"], "counter": 0}


async def increment_node(state: AppState) -> dict:
    counter = state["counter"] + 1
    return {"log": [f"increment -> {counter}"], "counter": counter}


def keep_looping(state: AppState) -> str:
    # Returns a semantic label rather than a node name directly — the dict
    # passed to add_conditional_edges below maps each label to its target
    # (LangGraph-style routing).
    return "continue" if state["counter"] < 3 else "stop"


async def main():
    flow = FlowEngine(AppState)

    flow.add_node("start", start_node)
    flow.add_node("increment", increment_node)

    flow.add_edge(START, "start")
    flow.add_edge("start", "increment")
    flow.add_conditional_edges(
        "increment", keep_looping, {"continue": "increment", "stop": END}
    )

    flow.build()

    print(flow.details())
    print(f"\nDiagram: {flow.visualize()}\n")

    result = await flow.invoke()
    print(f"Final counter: {result['counter']}")
    print("Log:")
    for entry in result["log"]:
        print(f"  - {entry}")


if __name__ == "__main__":
    asyncio.run(main())
