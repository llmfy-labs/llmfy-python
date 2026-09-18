"""
Retry policy, per-node timeout, and lifecycle hooks example.

`RetryPolicy` retries a node on transient failures with exponential
backoff; `timeout` bounds how long a single attempt may take;
`FlowEngineHooks` gives observability into every node's start/end/error
without subclassing FlowEngine.
"""

import asyncio
from typing import TypedDict

from llmfy import END, START, FlowEngine, FlowEngineHooks, RetryPolicy
from llmfy.exception.llmfy_exception import NodeExecutionException, NodeTimeoutException


class AppState(TypedDict):
    result: str


class FlakyServiceError(Exception):
    """Simulates a transient failure from an external service."""


_attempts = {"flaky": 0}


async def flaky_node(state: AppState) -> dict:
    """Fails twice, then succeeds — RetryPolicy carries it through."""
    _attempts["flaky"] += 1
    print(f"[flaky_node] attempt {_attempts['flaky']}")
    if _attempts["flaky"] < 3:
        raise FlakyServiceError("simulated transient failure")
    return {"result": "flaky_node succeeded"}


async def slow_node(state: AppState) -> dict:
    """Always exceeds its configured timeout."""
    await asyncio.sleep(1.0)
    return {"result": "unreachable"}  # pragma: no cover


def build_flow_with_retry() -> FlowEngine:
    flow = FlowEngine(
        AppState,
        hooks=FlowEngineHooks(
            on_node_start=lambda name, state: print(f"  [hook] starting '{name}'"),
            on_node_end=lambda name, state, updates: print(f"  [hook] finished '{name}': {updates}"),
            on_error=lambda name, exc: print(f"  [hook] error in '{name}': {exc!r}"),
        ),
    )
    flow.add_node(
        "flaky",
        flaky_node,
        retry=RetryPolicy(max_attempts=3, backoff_seconds=0.05, retry_on=(FlakyServiceError,)),
    )
    flow.add_edge(START, "flaky")
    flow.add_edge("flaky", END)
    return flow.build()


def build_flow_with_timeout() -> FlowEngine:
    flow = FlowEngine(AppState)
    flow.add_node("slow", slow_node, timeout=0.05)
    flow.add_edge(START, "slow")
    flow.add_edge("slow", END)
    return flow.build()


async def main():
    print("--- RetryPolicy + hooks: recovers from transient failures ---")
    flow = build_flow_with_retry()
    result = await flow.invoke()
    print(f"Result: {result}\n")

    print("--- timeout: raises NodeTimeoutException when exceeded ---")
    flow = build_flow_with_timeout()
    try:
        await flow.invoke()
    except NodeTimeoutException as exc:
        print(f"Caught expected timeout: node={exc.node_name!r}, timeout={exc.timeout_seconds}s")
    except NodeExecutionException as exc:
        print(f"Caught node failure: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
