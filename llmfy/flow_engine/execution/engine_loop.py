"""The single execution loop shared by `FlowEngine.invoke()` and `.stream()`.

Everything that previously lived twice — once in `invoke()`, once in
`stream()` — lives here exactly once: the node-by-node walk, fan-out
(parallel branches) and fan-in (join/barrier), the step-limit guard,
per-node retry/timeout, lifecycle hooks, and checkpoint saving.
`invoke()` drains the event stream and returns the final state; `stream()`
maps each event to a `FlowEngineStreamResponse` and yields it.
"""

from __future__ import annotations

import asyncio
import inspect
import uuid
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from llmfy.exception.llmfy_exception import (
    GraphValidationException,
    NodeExecutionException,
    NodeTimeoutException,
    StepLimitExceededException,
)
from llmfy.flow_engine.checkpointer.base_checkpointer import (
    BaseCheckpointer,
    Checkpoint,
    CheckpointMetadata,
)
from llmfy.flow_engine.checkpointer.serde import TypeRegistry, serialize_state
from llmfy.flow_engine.execution.context import ExecutionContext
from llmfy.flow_engine.execution.hooks import FlowEngineHooks
from llmfy.flow_engine.execution.policy import RetryPolicy
from llmfy.flow_engine.execution.send import Send
from llmfy.flow_engine.graph.graph_builder import CompiledGraph
from llmfy.flow_engine.node.node import END, START, Node
from llmfy.flow_engine.stream.node_stream_response import (
    NodeStreamResponse,
    NodeStreamType,
)


def apply_reducers(
    state: dict[str, Any], updates: dict[str, Any], reducers: dict[str, Callable | None]
) -> None:
    """Merge `updates` into `state` in place, using each field's reducer
    (`old_value, new_value -> merged_value`) where one is registered,
    otherwise replacing the value outright. Shared by the engine loop's
    per-node update step and `FlowEngine`'s checkpoint-resume `apply_state`
    merge, so both go through the exact same merge semantics."""
    for key, new_value in updates.items():
        reducer = reducers.get(key)
        if reducer is not None:
            old_value = state.get(key)
            state[key] = reducer(old_value, new_value)
        else:
            state[key] = new_value


@dataclass
class InternalEvent:
    """One step of engine execution, translated by `FlowEngine` into a
    public `FlowEngineStreamResponse` for `stream()`, or simply drained
    (only the final state kept) by `invoke()`."""

    type: str  # "start" | "node_stream" | "node_result"
    node: str | None
    content: Any = None
    state: dict[str, Any] | None = None
    branch_index: int | None = None
    branch_total: int | None = None
    dispatch_id: str | None = None


@dataclass(frozen=True)
class SendBranch:
    """Identifies one branch within one dynamic (`Send`) fan-out dispatch.

    Internal to `engine_loop.py` — never exported. `dispatch_id` is a
    fresh uuid per call to a routing function that returns `Send`(s), so
    two dispatches to the same target node (e.g. from two concurrent
    static-fan-out branches each doing their own `Send`) never collide;
    `index`/`total` give a branch's position for correlation and for
    depositing its updates into `_advance_dynamic_fan_out`'s collector.
    """

    dispatch_id: str
    index: int
    total: int


async def _merge_streams(
    generators: list[AsyncGenerator[InternalEvent, None]],
) -> AsyncGenerator[InternalEvent, None]:
    """Run multiple async generators concurrently, yielding each item as it
    arrives from whichever generator produces it first — this is what
    drives fan-out branches in parallel while still producing one ordered
    event stream for `invoke()`/`stream()` to consume."""
    queue: asyncio.Queue = asyncio.Queue()
    sentinel = object()
    pending = len(generators)

    async def pump(gen: AsyncGenerator[InternalEvent, None]) -> None:
        try:
            async for item in gen:
                await queue.put(item)
        except Exception as exc:  # noqa: BLE001 - re-raised on the consumer side
            await queue.put(exc)
        finally:
            await queue.put(sentinel)

    tasks = [asyncio.create_task(pump(gen)) for gen in generators]
    try:
        while pending > 0:
            item = await queue.get()
            if item is sentinel:
                pending -= 1
                continue
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


class EngineLoop:
    """Owns the compiled graph, node registry, reducers, checkpointing, and
    hooks for one `FlowEngine`, and drives the node-by-node walk for a
    given `ExecutionContext`. Stateless across calls — everything mutable
    lives on the `ExecutionContext` passed into `run()`."""

    def __init__(
        self,
        nodes: dict[str, Node],
        graph: CompiledGraph,
        reducers: dict[str, Callable | None],
        checkpointer: BaseCheckpointer | None = None,
        type_registry: TypeRegistry | None = None,
        hooks: FlowEngineHooks | None = None,
    ):
        self.nodes = nodes
        self.graph = graph
        self.reducers = reducers
        self.checkpointer = checkpointer
        self.type_registry = type_registry or TypeRegistry()
        self.hooks = hooks or FlowEngineHooks()

    async def run(
        self, ctx: ExecutionContext, start_node: str
    ) -> AsyncGenerator[InternalEvent, None]:
        yield InternalEvent(
            type="start", node=START, content=None, state=dict(ctx.state)
        )
        async for event in self._execute_from(ctx, start_node, arrived_from=None):
            yield event

    async def _execute_from(
        self, ctx: ExecutionContext, node_name: str, arrived_from: str | None
    ) -> AsyncGenerator[InternalEvent, None]:
        if self.graph.is_join(node_name) and arrived_from is not None:
            proceed = await self._arrive_at_join(ctx, node_name, arrived_from)
            if not proceed:
                return

        async for event in self._run_node(ctx, node_name, arrived_from=arrived_from):
            yield event

        async for event in self._advance(ctx, node_name):
            yield event

    async def _run_node(
        self,
        ctx: ExecutionContext,
        node_name: str,
        state: dict[str, Any] | None = None,
        branch: SendBranch | None = None,
        collector: list[dict[str, Any]] | None = None,
        arrived_from: str | None = None,
    ) -> AsyncGenerator[InternalEvent, None]:
        """Execute one node: step-count, hooks, run-with-retry/timeout,
        apply updates into the shared `ctx.state`, checkpoint, yield its
        events. Does NOT advance to whatever comes next — callers decide
        that (a plain `_execute_from` advances once per call; a dynamic
        fan-out runs several `_run_node` calls concurrently and advances
        only once, after all of them finish).

        `state` is this invocation's input if given (a `Send`'s per-branch
        override — see `execution/send.py`), otherwise the shared
        `ctx.state` is used, identical to every pre-existing call site.

        `arrived_from` is whichever node's completion led here — `None`
        only for this run's very first node (fresh or resumed; see
        `run()`), in which case the saved checkpoint's `prev_node` records
        `START` instead.

        `branch` marks this call as one branch of a dynamic-fan-out
        dispatch (set only by `_advance_dynamic_fan_out`). When set, this
        method does NOT merge its own `updates` into `ctx.state` or save a
        checkpoint — it deposits `updates` into `collector[branch.index]`
        instead, and `_advance_dynamic_fan_out` commits every branch's
        updates atomically, in one shot, only once every branch succeeds.
        This is what makes a dynamic-fan-out dispatch all-or-nothing: a
        failing branch's exception propagates out of `_merge_streams`
        before any commit code runs, so `ctx.state` is never left holding
        some branches' updates but not others'.
        """
        effective_state = state if state is not None else ctx.state

        await self._increment_step(ctx, node_name)

        node = self.nodes[node_name]
        await self._call_hook(
            self.hooks.on_node_start, node_name, dict(effective_state)
        )
        if branch is not None:
            await self._call_hook(
                self.hooks.on_branch_start,
                node_name,
                branch.index,
                branch.total,
                dict(effective_state),
            )

        try:
            updates: dict[str, Any] = {}
            result_content: Any = None
            # Set by `_execute_node_with_policy`/`_execute_stream_node_with_policy`
            # to whichever attempt actually succeeded, once it does.
            attempt_holder: list[int] = [1]
            if node.stream:
                async for chunk in self._execute_stream_node_with_policy(
                    node, effective_state, attempt_holder
                ):
                    if chunk.type == NodeStreamType.RESULT:
                        updates = chunk.state or {}
                        result_content = chunk.content
                    else:
                        yield InternalEvent(
                            type="node_stream",
                            node=node_name,
                            content=chunk.content,
                            state=dict(effective_state),
                            branch_index=branch.index if branch else None,
                            branch_total=branch.total if branch else None,
                            dispatch_id=branch.dispatch_id if branch else None,
                        )
            else:
                updates = await self._execute_node_with_policy(
                    node, effective_state, attempt_holder
                )
                result_content = updates
        except Exception as exc:
            await self._call_hook(self.hooks.on_error, node_name, exc)
            raise

        if branch is not None:
            if collector is not None:
                collector[branch.index] = updates
            # Pre-commit snapshot: this dispatch's updates (this branch's
            # and every sibling's) are not yet in `ctx.state` — they only
            # land together, atomically, once `_advance_dynamic_fan_out`
            # confirms every branch succeeded. Documented explicitly on
            # `FlowEngineHooks.on_node_end`/`on_branch_end` and in
            # `execution/send.py` as a deliberate contract for Send
            # branches specifically; plain nodes and static fan-out are
            # unaffected and keep seeing post-update state as before.
            snapshot_state = dict(ctx.state)
            await self._call_hook(
                self.hooks.on_node_end, node_name, snapshot_state, updates
            )
            await self._call_hook(
                self.hooks.on_branch_end,
                node_name,
                branch.index,
                branch.total,
                snapshot_state,
                updates,
            )
            yield InternalEvent(
                type="node_result",
                node=node_name,
                content=result_content,
                state=snapshot_state,
                branch_index=branch.index,
                branch_total=branch.total,
                dispatch_id=branch.dispatch_id,
            )
            return

        if updates:
            await self._apply_updates(ctx, updates)

        await self._call_hook(
            self.hooks.on_node_end, node_name, dict(ctx.state), updates
        )
        await self._save_checkpoint(
            ctx,
            node_name,
            prev_node=arrived_from or START,
            updated_fields=list(updates.keys()),
            attempt=attempt_holder[0],
        )

        # Yielded after updates are applied and checkpointed, so the state
        # snapshot attached to this event reflects this node's own updates
        # — for a stream node, its STREAM chunks (above) intentionally see
        # pre-update state, but its final RESULT event must not.
        yield InternalEvent(
            type="node_result",
            node=node_name,
            content=result_content,
            state=dict(ctx.state),
        )

    async def _advance(
        self, ctx: ExecutionContext, node_name: str
    ) -> AsyncGenerator[InternalEvent, None]:
        if self.graph.is_conditional(node_name):
            result = await self._eval_condition(ctx, node_name)
            # Check `list` before any truthiness test — an empty list of
            # Sends ("map over zero items") is falsy but is NOT the same
            # case as `None` (END): both currently produce no further
            # execution, but for different reasons, and must stay
            # structurally distinct so a future distinct empty-fan-out
            # event doesn't have to be threaded through a truthiness check.
            if isinstance(result, list):
                async for event in self._advance_dynamic_fan_out(
                    ctx, result, dispatched_from=node_name
                ):
                    yield event
                return
            if result is None:
                return
            async for event in self._execute_from(
                ctx, result, arrived_from=node_name
            ):
                yield event
            return

        targets = [t for t in self.graph.targets_of(node_name) if t != END]
        if not targets:
            return
        if len(targets) == 1:
            async for event in self._execute_from(
                ctx, targets[0], arrived_from=node_name
            ):
                yield event
            return

        branches = [
            self._execute_from(ctx, target, arrived_from=node_name)
            for target in targets
        ]
        async for event in _merge_streams(branches):
            yield event

    async def _advance_dynamic_fan_out(
        self, ctx: ExecutionContext, sends: list[Send], dispatched_from: str
    ) -> AsyncGenerator[InternalEvent, None]:
        """Run every `Send` concurrently against its (single, shared)
        target node, commit their updates atomically once every branch
        succeeds, then advance downstream from that node exactly ONCE —
        not once per branch. This is what makes "map then reduce-once"
        work without any join/barrier bookkeeping: the branch count is
        already known (`len(sends)`) the instant the routing function
        returns, unlike a statically-declared join whose predecessor count
        is only known at graph-build time.

        Each branch calls `_run_node` directly (not `_execute_from`) so it
        skips the join-check and does NOT advance on its own — advancing
        once per branch here would run whatever comes after the mapped
        node once per item, which is never what "map" means.

        Commit is all-or-nothing: `_run_node` defers each branch's updates
        into `collector` instead of touching `ctx.state`, so if any branch
        raises, `_merge_streams` cancels the rest and re-raises *before*
        the merge loop below ever runs — `ctx.state` ends up completely
        untouched by a dispatch that didn't fully succeed. Only on full
        success are all branches' updates merged (in branch-index order,
        for reproducibility) and exactly one checkpoint saved for the
        whole dispatch — never one per branch, which is what previously
        made a checkpoint able to represent "some but not all branches
        done" (the resume hazard documented in `execution/send.py`).
        That one checkpoint's `prev_node` is `dispatched_from`, the node
        whose conditional routing produced these `Send`s; its `dispatch_id`
        is this dispatch's own id (correlates with the same id on this
        dispatch's stream events); its `updated_fields` is the union of
        every branch's update keys; its `attempt` is `None` — several
        branches, each with its own possibly-different attempt count, are
        committed together, so no single attempt number applies.
        """
        if not sends:
            return

        dispatch_id = str(uuid.uuid4())
        total = len(sends)
        collector: list[dict[str, Any]] = [{} for _ in range(total)]
        branches = [
            self._run_node(
                ctx,
                send.node,
                state=send.state,
                branch=SendBranch(dispatch_id=dispatch_id, index=i, total=total),
                collector=collector,
            )
            for i, send in enumerate(sends)
        ]
        async for event in _merge_streams(branches):
            yield event

        # Reached only if every branch succeeded (a branch's exception
        # propagates from `_merge_streams` above and skips everything
        # below) — safe to commit all collected updates in one shot.
        async with ctx.lock:
            for updates in collector:
                if updates:
                    apply_reducers(ctx.state, updates, self.reducers)

        target_node = sends[0].node
        updated_fields = sorted({key for updates in collector for key in updates})
        await self._save_checkpoint(
            ctx,
            target_node,
            prev_node=dispatched_from,
            updated_fields=updated_fields,
            dispatch_id=dispatch_id,
        )

        async for event in self._advance(ctx, target_node):
            yield event

    async def _arrive_at_join(
        self, ctx: ExecutionContext, node_name: str, arrived_from: str
    ) -> bool:
        async with ctx.lock:
            arrivals = ctx.join_arrivals.setdefault(node_name, set())
            arrivals.add(arrived_from)
            total_predecessors = len(self.graph.predecessors.get(node_name, set()))
            if len(arrivals) >= total_predecessors:
                ctx.join_arrivals[node_name] = set()
                return True
            return False

    async def _increment_step(self, ctx: ExecutionContext, node_name: str) -> None:
        async with ctx.lock:
            ctx.step += 1
            if ctx.step > ctx.max_steps:
                raise StepLimitExceededException(
                    f"FlowEngine exceeded max_steps={ctx.max_steps} at step {ctx.step} "
                    f"(about to execute node '{node_name}'). Increase max_steps if this "
                    "is a legitimate long-running workflow, or check for an unintended "
                    "infinite loop in a conditional edge.",
                    session_id=ctx.session_id,
                    step=ctx.step,
                    max_steps=ctx.max_steps,
                    node_name=node_name,
                )

    async def _apply_updates(
        self, ctx: ExecutionContext, updates: dict[str, Any]
    ) -> None:
        async with ctx.lock:
            apply_reducers(ctx.state, updates, self.reducers)

    async def _eval_condition(
        self, ctx: ExecutionContext, node_name: str
    ) -> str | list[Send] | None:
        edge = self.graph.outgoing[node_name]
        condition_func = edge.condition
        if condition_func is None:
            # Unreachable in practice — `_advance` only calls this when
            # `graph.is_conditional(node_name)` is True, which itself means
            # `edge.condition is not None`. Guarded explicitly so this fails
            # loudly instead of a bare "NoneType is not callable" if that
            # invariant is ever violated.
            raise GraphValidationException(
                f"Node '{node_name}' has no condition function to evaluate"
            )

        if inspect.iscoroutinefunction(condition_func):
            result = await condition_func(ctx.state)
        else:
            result = condition_func(ctx.state)

        # A Send/list[Send] names its own target(s) explicitly and must
        # never be run through target_map key-resolution or the plain
        # targets-membership check below (those are for a bare-string
        # return value only) — checked first, before target_map.
        if isinstance(result, Send) or isinstance(result, list):
            return self._validate_sends(node_name, edge.targets, result)

        if edge.target_map is not None:
            if result not in edge.target_map:
                raise GraphValidationException(
                    f"Condition function for node '{node_name}' returned '{result}' "
                    f"which is not a key in its target map: {list(edge.target_map)}"
                )
            next_node = edge.target_map[result]
        else:
            if result not in edge.targets:
                raise GraphValidationException(
                    f"Condition function for node '{node_name}' returned '{result}' "
                    f"which is not in its declared targets: {edge.targets}"
                )
            next_node = result

        return None if next_node == END else next_node

    def _validate_sends(
        self, node_name: str, declared_targets: list[str], result: Send | list[Send]
    ) -> list[Send]:
        sends = [result] if isinstance(result, Send) else result

        for item in sends:
            if not isinstance(item, Send):
                raise GraphValidationException(
                    f"Condition function for node '{node_name}' returned a list "
                    f"containing a non-Send item: {item!r}. A list return value "
                    "must contain only Send objects."
                )

        if not sends:
            return sends

        offenders = [s.node for s in sends if s.node not in declared_targets]
        if offenders:
            raise GraphValidationException(
                f"Send(s) from node '{node_name}' target {sorted(set(offenders))} "
                f"which {'is' if len(set(offenders)) == 1 else 'are'} not in its "
                f"declared targets: {declared_targets}"
            )

        distinct_targets = {s.node for s in sends}
        if len(distinct_targets) > 1:
            raise GraphValidationException(
                f"All Send objects returned from node '{node_name}' must target "
                f"the same node; got {sorted(distinct_targets)}. Fanning out to "
                "different nodes from one routing call is not supported."
            )

        return sends

    async def _call_hook(self, hook: Callable | None, *args: Any) -> None:
        if hook is None:
            return
        if inspect.iscoroutinefunction(hook):
            await hook(*args)
        else:
            hook(*args)

    async def _save_checkpoint(
        self,
        ctx: ExecutionContext,
        node_name: str,
        prev_node: str,
        updated_fields: list[str],
        attempt: int | None = None,
        dispatch_id: str | None = None,
    ) -> None:
        if self.checkpointer is None:
            return
        state_to_store = (
            serialize_state(ctx.state, self.type_registry)
            if self.checkpointer.requires_serialization
            else ctx.state
        )
        checkpoint = Checkpoint(
            metadata=CheckpointMetadata(
                checkpoint_id=str(uuid.uuid4()),
                session_id=ctx.session_id,
                run_id=ctx.run_id,
                created_at=datetime.now(UTC),
                node=node_name,
                prev_node=prev_node,
                step=ctx.step,
                updated_fields=updated_fields,
                attempt=attempt,
                dispatch_id=dispatch_id,
            ),
            state=state_to_store,
        )
        await self.checkpointer.save(checkpoint)

    # -- node execution with retry/timeout ---------------------------------

    async def _execute_node_with_policy(
        self, node: Node, state: dict[str, Any], attempt_holder: list[int]
    ) -> dict[str, Any]:
        policy = node.retry or RetryPolicy()
        for attempt in range(1, policy.max_attempts + 1):
            try:
                result = await self._call_node_once(node, state)
                attempt_holder[0] = attempt
                return result
            except GraphValidationException:
                # A contract/programming error (e.g. "no function defined"),
                # not a transient runtime failure — never retried or wrapped.
                raise
            except TimeoutError as exc:
                if attempt >= policy.max_attempts:
                    raise NodeTimeoutException(
                        f"Node '{node.name}' timed out after {attempt} attempt(s) "
                        f"(timeout={node.timeout}s)",
                        node_name=node.name,
                        attempt=attempt,
                        timeout_seconds=node.timeout,  # type: ignore[arg-type]
                    ) from exc
            except Exception as exc:
                if attempt >= policy.max_attempts or not policy.should_retry(exc):
                    raise NodeExecutionException(
                        f"Node '{node.name}' failed after {attempt} attempt(s): {exc}",
                        node_name=node.name,
                        attempt=attempt,
                    ) from exc

            delay = policy.delay_for_attempt(attempt)
            if delay > 0:
                await asyncio.sleep(delay)

        # Unreachable: the loop above always returns or raises.
        raise NodeExecutionException(
            f"Node '{node.name}' failed",
            node_name=node.name,
            attempt=policy.max_attempts,
        )

    async def _call_node_once(self, node: Node, state: dict[str, Any]) -> dict[str, Any]:
        func = node.func
        if func is None:
            raise GraphValidationException(
                f"Node '{node.name}' has no function defined"
            )

        if inspect.iscoroutinefunction(func):
            coro = func(state)
            result = (
                await asyncio.wait_for(coro, timeout=node.timeout)
                if node.timeout
                else await coro
            )
        else:
            result = func(state)

        return result or {}

    async def _execute_stream_node_with_policy(
        self, node: Node, state: dict[str, Any], attempt_holder: list[int]
    ) -> AsyncGenerator[NodeStreamResponse, None]:
        policy = node.retry or RetryPolicy()
        for attempt in range(1, policy.max_attempts + 1):
            try:
                async for chunk in self._execute_stream_node_once(node, state):
                    yield chunk
                attempt_holder[0] = attempt
                return
            except GraphValidationException:
                raise
            except TimeoutError as exc:
                if attempt >= policy.max_attempts:
                    raise NodeTimeoutException(
                        f"Node '{node.name}' timed out after {attempt} attempt(s) "
                        f"(timeout={node.timeout}s)",
                        node_name=node.name,
                        attempt=attempt,
                        timeout_seconds=node.timeout,  # type: ignore[arg-type]
                    ) from exc
            except Exception as exc:
                if attempt >= policy.max_attempts or not policy.should_retry(exc):
                    raise NodeExecutionException(
                        f"Node '{node.name}' failed after {attempt} attempt(s): {exc}",
                        node_name=node.name,
                        attempt=attempt,
                    ) from exc

            delay = policy.delay_for_attempt(attempt)
            if delay > 0:
                await asyncio.sleep(delay)

    async def _execute_stream_node_once(
        self, node: Node, state: dict[str, Any]
    ) -> AsyncGenerator[NodeStreamResponse, None]:
        func = node.func
        if func is None:
            raise GraphValidationException(
                f"Node '{node.name}' has no function defined"
            )

        if inspect.isasyncgenfunction(func):
            agen = func(state)
            while True:
                try:
                    chunk = (
                        await asyncio.wait_for(agen.__anext__(), timeout=node.timeout)
                        if node.timeout
                        else await agen.__anext__()
                    )
                except StopAsyncIteration:
                    break
                if not isinstance(chunk, NodeStreamResponse):
                    raise GraphValidationException(
                        f"Stream response in node '{node.name}' must use `NodeStreamResponse`"
                    )
                yield chunk

        elif inspect.isgeneratorfunction(func):
            for chunk in func(state):
                if not isinstance(chunk, NodeStreamResponse):
                    raise GraphValidationException(
                        f"Stream response in node '{node.name}' must use `NodeStreamResponse`"
                    )
                yield chunk

        else:
            raise GraphValidationException(
                f"Function in node '{node.name}' is not a stream generator. "
                "Please yield `NodeStreamResponse`."
            )
