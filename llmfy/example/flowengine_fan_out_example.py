"""
Fan-out / fan-in (parallel branches + join) example.

`add_edge(source, [target_a, target_b])` (or two separate `add_edge` calls
from the same source) fans out — every target runs concurrently. A node
reached by more than one such edge becomes a join: it waits for every
branch to arrive before running, and per-field reducers merge whatever
each branch wrote to state, regardless of which branch finished first.
"""

import asyncio
from typing import Annotated, TypedDict

from llmfy import END, START, FlowEngine


def merge_sets(old: set[str] | None, new: set[str]) -> set[str]:
    """Order-independent reducer — required for fields written by parallel
    branches, since branches may finish in either order."""
    return (old or set()) | new


class ResearchState(TypedDict):
    topic: str
    findings: Annotated[set[str], merge_sets]


async def fetch(state: ResearchState) -> dict:
    print(f"[fetch] topic={state['topic']!r}")
    return {}


async def research_facts(state: ResearchState) -> dict:
    await asyncio.sleep(0.3)  # simulates a slower call, e.g. a web search
    print("[research_facts] done")
    return {"findings": {"facts"}}


async def research_opinions(state: ResearchState) -> dict:
    await asyncio.sleep(0.1)  # simulates a faster call, e.g. a cached lookup
    print("[research_opinions] done")
    return {"findings": {"opinions"}}


async def combine(state: ResearchState) -> dict:
    # Runs exactly once, only after BOTH branches above have completed.
    print(f"[combine] findings so far: {state['findings']}")
    return {}


async def main():
    flow = FlowEngine(ResearchState)

    flow.add_node("fetch", fetch)
    flow.add_node("research_facts", research_facts)
    flow.add_node("research_opinions", research_opinions)
    flow.add_node("combine", combine)

    flow.add_edge(START, "fetch")
    flow.add_edge("fetch", ["research_facts", "research_opinions"])  # fan-out
    flow.add_edge("research_facts", "combine")  # fan-in / join
    flow.add_edge("research_opinions", "combine")
    flow.add_edge("combine", END)

    flow.build()
    print(flow.details())
    print(flow.visualize())

    result = await flow.invoke(apply_state={"topic": "FlowEngine"})
    print(f"\nFinal findings: {result['findings']}")


if __name__ == "__main__":
    asyncio.run(main())
