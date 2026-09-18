"""
Dynamic fan-out (`Send`) example — LangGraph-style "map" over a
runtime-determined number of items.

Unlike `flowengine_fan_out_example.py`'s STATIC fan-out (branch targets
fixed via `add_edge(...)` before `build()`), a routing function registered
with `add_conditional_edges` can return a `Send` or `list[Send]` to spawn a
branch COUNT that's only known when the function runs — e.g. one branch per
item in a list whose length depends on upstream state. Every `Send` in one
routing call must target the same node; that node's own outgoing edge
determines what runs next, exactly once, after every branch finishes.

The dispatch commits ATOMICALLY: every branch's update lands in shared
state together, only once every branch succeeds — if any `tell_joke` call
raised, `finalize` would never run and NO joke would be recorded (not a
partial list). This example uses `stream()` (rather than `invoke()`) to
also show `branch_index`/`branch_total`/`dispatch_id` on each branch's
`FlowEngineStreamResponse` — the only way to tell "topic 2 of 3" apart from
another when every branch shares the same node name, `tell_joke`.

`max_steps` arithmetic for this example: `plan_topics` (1) + N `tell_joke`
branches (N, one per topic) + `finalize` (1) = N + 2 steps. `FlowEngine`'s
default `max_steps=100` easily covers the 3 topics here, but a map over,
say, 200 items would need `FlowEngine(AppState, max_steps=210)` or similar
— size it to (branch count) + a small fixed overhead, not just the branch
count alone.
"""

import asyncio
from typing import Annotated, TypedDict

from llmfy import END, START, FlowEngine, Send


def append_list(old: list[str] | None, new: list[str]) -> list[str]:
    """Order-independent enough for this example (order of jokes doesn't
    matter) — required for a field written by parallel Send branches."""
    return (old or []) + new


class JokeState(TypedDict):
    topics: list[str]
    jokes: Annotated[list[str], append_list]


async def plan_topics(state: JokeState) -> dict:
    # The topic COUNT is only known here, at runtime (e.g. from user input,
    # a DB query, or an upstream LLM call) — this is exactly the case
    # static fan-out (fixed targets declared before build()) can't express.
    topics = ["cats", "databases", "mondays"]
    print(f"[plan_topics] {len(topics)} topics: {topics}")
    return {"topics": topics}


def route_to_joke_per_topic(state: JokeState) -> list[Send]:
    # One Send per topic — all targeting "tell_joke", each with its own
    # isolated input (`{"topic": t}` fully REPLACES what "tell_joke" sees;
    # it does NOT also receive the parent's `topics` list or anything else
    # from shared state).
    return [Send("tell_joke", {"topic": t}) for t in state["topics"]]


async def tell_joke(state: dict) -> dict:
    # `state` here is exactly one Send's `{"topic": ...}` — not JokeState.
    topic = state["topic"]
    await asyncio.sleep(0.05)  # simulates a per-topic LLM call
    joke = f"Why did the {topic} cross the road? To fan out dynamically."
    print(f"[tell_joke] {topic} -> done")
    return {"jokes": [joke]}


async def finalize(state: JokeState) -> dict:
    # Runs exactly once, only after every tell_joke branch has completed.
    print(f"[finalize] collected {len(state['jokes'])} jokes")
    return {}


async def main():
    flow = FlowEngine(JokeState)

    flow.add_node("plan_topics", plan_topics)
    flow.add_node("tell_joke", tell_joke)
    flow.add_node("finalize", finalize)

    flow.add_edge(START, "plan_topics")
    flow.add_conditional_edges("plan_topics", route_to_joke_per_topic, ["tell_joke"])
    flow.add_edge("tell_joke", "finalize")
    flow.add_edge("finalize", END)

    flow.build()
    print(flow.details())
    print(flow.visualize())

    final_state: dict = {}
    async for response in flow.stream():
        if (
            response.node == "tell_joke"
            and response.type == "result"
            and response.branch_index is not None
            and response.branch_total is not None
            and response.dispatch_id is not None
        ):
            print(
                f"[stream] tell_joke branch {response.branch_index + 1}/"
                f"{response.branch_total} (dispatch={response.dispatch_id[:8]}...) "
                f"-> {response.content}"
            )
        if response.state is not None:
            final_state = response.state

    print("\nJokes:")
    for joke in final_state["jokes"]:
        print(f"  - {joke}")


if __name__ == "__main__":
    asyncio.run(main())
