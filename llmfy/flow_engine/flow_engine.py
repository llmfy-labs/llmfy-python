import inspect
import re
from collections.abc import Callable
from typing import (
    Annotated,
    Any,
    get_args,
    get_origin,
    get_type_hints,
)
from uuid import uuid4

from llmfy.exception.llmfy_exception import (
    GraphValidationException,
    InvalidSessionIdException,
)
from llmfy.flow_engine.checkpointer.base_checkpointer import (
    BaseCheckpointer,
    Checkpoint,
)
from llmfy.flow_engine.checkpointer.serde import TypeRegistry, deserialize_state
from llmfy.flow_engine.edge.edge import Edge
from llmfy.flow_engine.execution.context import ExecutionContext
from llmfy.flow_engine.execution.engine_loop import (
    EngineLoop,
    InternalEvent,
    apply_reducers,
)
from llmfy.flow_engine.execution.hooks import FlowEngineHooks
from llmfy.flow_engine.execution.policy import RetryPolicy
from llmfy.flow_engine.graph.graph_builder import build_graph
from llmfy.flow_engine.graph.validation import validate_workflow
from llmfy.flow_engine.node.node import END, START, Node, NodeType
from llmfy.flow_engine.stream.flow_engine_stream_response import (
    FlowEngineStreamResponse,
    FlowEngineStreamType,
)
from llmfy.flow_engine.visualizer.visualizer import WorkflowVisualizer

# Allow-list for caller-supplied `session_id`s: safe as a SQL primary key
# and as a Redis key-namespace segment. Engine-generated `checkpoint_id`s
# (always `uuid.uuid4()`) never pass through this check — only a `session_id`
# from `invoke()`/`stream()` does.
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,255}$")


class FlowEngine:
    """
    A workflow engine that manages state transitions through nodes and edges.

    Attributes:
        state_schema: TypedDict class defining the state structure
        nodes: Dictionary of node name to node function
        edges: List of edges connecting nodes
        checkpointer: Optional checkpointer for state persistence
        max_steps: Default step-limit guard for invoke()/stream() calls
        hooks: Optional lifecycle hooks (on_node_start/on_node_end/on_error)
    """

    def __init__(
        self,
        state_schema: type,
        checkpointer: BaseCheckpointer | None = None,
        max_steps: int = 100,
        types: list[type] | None = None,
        hooks: FlowEngineHooks | None = None,
    ):
        """
        Initialize the FlowEngine with a state schema.

        Args:
            state_schema: A TypedDict class defining the state structure
            checkpointer: Optional checkpointer for state persistence
            max_steps: Default max node executions per invoke()/stream() call
                before a `StepLimitExceededException` is raised — guards
                against an unintended infinite loop in a conditional edge.
                Overridable per call.
            types: Custom types (Pydantic `BaseModel` or `@dataclass`) that
                may appear in state and need to cross a checkpointer's
                serialization boundary (Redis/SQL). See `register_type`.
            hooks: Optional lifecycle hooks for observability/tracing.
        """
        self.is_built = False
        self.state_schema = state_schema
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.max_steps = max_steps
        self._reducers: dict[str, Callable | None] = {}
        self._type_hints: dict[str, Any] = {}

        self.checkpointer = checkpointer
        self.type_registry = TypeRegistry(types)
        self.hooks = hooks or FlowEngineHooks()

        self._engine_loop: EngineLoop | None = None

        # Add special START and END nodes
        self.nodes[START] = Node(name=START, node_type=NodeType.START)
        self.nodes[END] = Node(name=END, node_type=NodeType.END)

        # Extract annotations from the state schema and validate reducers
        self._extract_state_annotations()

        # Visualizer
        self.visualizer = WorkflowVisualizer()

    def _extract_state_annotations(self):
        """Extract and validate reducer functions from the state schema."""
        hints = get_type_hints(self.state_schema, include_extras=True)

        for f, hint in hints.items():
            origin = get_origin(hint)

            if origin is Annotated:
                args = get_args(hint)
                if len(args) >= 2:
                    self._type_hints[f] = args[0]
                    reducer = args[1]
                    self._validate_reducer(f, reducer)
                    self._reducers[f] = reducer
                else:
                    self._type_hints[f] = args[0] if args else hint
                    self._reducers[f] = None
            else:
                self._type_hints[f] = hint
                self._reducers[f] = None

    def _validate_reducer(self, field_name: str, reducer: Callable):
        """
        Validate that a reducer function has the correct signature.

        Args:
            field_name: Name of the field this reducer is for
            reducer: The reducer function to validate

        Raises:
            GraphValidationException: If the reducer doesn't have exactly 2 parameters
        """
        if not callable(reducer):
            raise GraphValidationException(
                f"Reducer for field '{field_name}' must be callable, got {type(reducer)}"
            )

        sig = inspect.signature(reducer)
        params = list(sig.parameters.values())

        if len(params) != 2:
            raise GraphValidationException(
                f"Reducer for field '{field_name}' must have exactly 2 parameters "
                f"(old_value, new_value), but has {len(params)} parameters"
            )

    def register_type(self, cls: type) -> None:
        """
        Register a custom type (Pydantic `BaseModel` or `@dataclass`) that may
        appear in state and needs to cross a checkpointer's serialization
        boundary (only Redis/SQL checkpointers cross one — `InMemoryCheckpointer`
        keeps live objects and never needs a registered type).

        Args:
            cls: The type to register.
        """
        self.type_registry.register(cls)

    def add_node(
        self,
        name: str,
        func: Callable,
        stream: bool = False,
        retry: RetryPolicy | None = None,
        timeout: float | None = None,
    ):
        """
        Add a node to the workflow.

        Args:
            name (str): Name of the node
            func (Callable): Function to execute (can be sync or async)
            stream (bool): Node is use stream or not, if node use streaming set to True. Defaults to False.
            retry (RetryPolicy | None): Retry behavior for transient failures. Defaults to no retry.
            timeout (float | None): Per-attempt timeout in seconds. Defaults to no timeout.
        """
        if name in (START, END):
            raise GraphValidationException(f"Cannot add node with reserved name: {name}")

        node = Node(
            name=name,
            node_type=NodeType.FUNCTION,
            func=func,
            stream=stream,
            retry=retry,
            timeout=timeout,
        )
        self.nodes[name] = node

    def add_edge(self, source: str, target: str | list[str]):
        """
        Add an edge connecting a node to one or more targets.

        Passing a list of targets (or calling `add_edge` more than once from
        the same source) fans out — every target executes concurrently, and
        a downstream node with more than one such predecessor becomes a
        join: it waits for every predecessor branch before running.

        Args:
            source: Source node name (can be START)
            target: Target node name, or a list of target node names to fan
                out to (can include END)
        """
        if target == START or (isinstance(target, list) and START in target):
            raise GraphValidationException("START cannot be a target node")

        if source == END:
            raise GraphValidationException("END cannot be a source node")

        targets = [target] if isinstance(target, str) else target
        if source in targets:
            raise GraphValidationException(
                "Source same as target, edge cannot target itself"
            )

        edge = Edge(source=source, targets=target)
        self.edges.append(edge)

        if source in self.nodes:
            self.nodes[source].targets.extend(edge.targets)
        for to_node in edge.targets:
            if to_node in self.nodes:
                self.nodes[to_node].sources.append(source)

    def add_conditional_edges(
        self,
        source: str,
        condition_func: Callable,
        targets: list[str] | dict[str, str],
    ):
        """
        Add a conditional edge that routes to different nodes based on a condition.

        Args:
            source: Source node name
            condition_func: Function that takes state and returns either a
                target node name (if `targets` is a list) or a key into
                `targets` (if `targets` is a dict, e.g.
                `{"Pass": END, "Fail": "improve_joke"}`).
            targets: List of possible target nodes (can include END), or a
                dict mapping `condition_func`'s return value to the actual
                target node name.
        """
        if isinstance(targets, dict):
            target_map: dict[str, str] | None = targets
            target_names: list[str] = list(targets.values())
        else:
            target_map = None
            target_names = targets

        if START in target_names:
            raise GraphValidationException(
                "START cannot be a target in conditional edges"
            )

        if source == END:
            raise GraphValidationException("END cannot be a source node")

        edge = Edge(
            source=source, targets=target_names, condition=condition_func, target_map=target_map
        )
        self.edges.append(edge)

        if source in self.nodes and source not in (START, END):
            self.nodes[source].node_type = NodeType.CONDITIONAL

        if source in self.nodes:
            self.nodes[source].targets.extend(target_names)
        for to_node in target_names:
            if to_node in self.nodes:
                self.nodes[to_node].sources.append(source)

    def build(self):
        """
        Build FlowEngine workflow.

        Returns:
            FlowEngine: Workflow built.
        """
        graph = build_graph(self.nodes, self.edges)
        validate_workflow(self.nodes, self.edges, graph)

        self._engine_loop = EngineLoop(
            nodes=self.nodes,
            graph=graph,
            reducers=self._reducers,
            checkpointer=self.checkpointer,
            type_registry=self.type_registry,
            hooks=self.hooks,
        )

        self.is_built = True
        return self

    def _require_built(self) -> EngineLoop:
        if not self.is_built or self._engine_loop is None:
            raise GraphValidationException("Build first. Use `your_flow.build()`")
        return self._engine_loop

    async def _new_context(
        self,
        apply_state: dict[str, Any] | None,
        session_id: str | None,
        max_steps: int | None,
    ) -> tuple[ExecutionContext, str]:
        """Resolve the starting node and initial state for a call — either
        fresh, or resumed from the last checkpoint for `session_id`.

        Every call gets its own `run_id` and starts `step` at 0, whether
        fresh or resumed: `max_steps` guards this one `invoke()`/`stream()`
        call against a runaway loop, not the cumulative lifetime of a
        `session_id` — a resumed run picks up the graph position (which
        node to start at) from the last checkpoint, but gets its own fresh
        step budget and its own `run_id` grouping the checkpoints it saves.
        """
        engine_loop = self._require_built()
        session_id = session_id or str(uuid4())
        if not _SESSION_ID_RE.match(session_id):
            raise InvalidSessionIdException(
                f"Invalid session_id {session_id!r}: must match "
                f"{_SESSION_ID_RE.pattern!r}.",
                session_id=session_id,
            )
        apply_state = apply_state or {}
        effective_max_steps = max_steps if max_steps is not None else self.max_steps
        run_id = str(uuid4())

        loaded: Checkpoint | None = None
        if session_id and self.checkpointer:
            loaded = await self.checkpointer.load(session_id)

        if loaded is not None:
            raw_state = (
                deserialize_state(loaded.state, self.type_registry)
                if self.checkpointer.requires_serialization  # type: ignore[union-attr]
                else loaded.state
            )
            ctx = ExecutionContext(
                session_id=session_id,
                state=raw_state,
                run_id=run_id,
                max_steps=effective_max_steps,
            )
            if apply_state:
                apply_reducers(ctx.state, apply_state, self._reducers)

            start_node = self._resolve_resume_node(engine_loop, loaded.metadata.node)
            if start_node is None:
                start_node = self._start_target(engine_loop)
            return ctx, start_node

        ctx = ExecutionContext(
            session_id=session_id,
            state=dict(apply_state),
            run_id=run_id,
            max_steps=effective_max_steps,
        )
        return ctx, self._start_target(engine_loop)

    def _start_target(self, engine_loop: EngineLoop) -> str:
        targets = engine_loop.graph.targets_of(START)
        if not targets:
            raise GraphValidationException(
                "No edge from START node. Use flow.add_edge(START, 'node_name')"
            )
        return targets[0]

    def _resolve_resume_node(self, engine_loop: EngineLoop, completed_node: str) -> str | None:
        """Determine the node to resume at, right after `completed_node`.

        Resuming mid-fan-out (a checkpoint saved while multiple branches
        were in flight) is not supported — this always resumes via a
        single next node, taking the first target when `completed_node`'s
        outgoing edge fans out to several. Fan-out resume across a process
        restart isn't well-defined without knowing which branches had
        already completed, so this is a known, documented limitation.
        """
        graph = engine_loop.graph
        if graph.is_conditional(completed_node):
            # Condition functions are evaluated live during a run, not
            # re-derivable from a checkpoint alone — resuming right after a
            # conditional node re-enters via its regular successor set is
            # not attempted; instead, treat it the same as "workflow
            # completed" and let the caller start over from START.
            return None

        targets = [t for t in graph.targets_of(completed_node) if t != END]
        return targets[0] if targets else None

    async def invoke(
        self,
        apply_state: dict[str, Any] | None = None,
        session_id: str | None = None,
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        """
        Execute the workflow starting from START node or continue from last checkpoint.

        Args:
            apply_state: Optional state updates to apply. If continuing from checkpoint,
                these are merged with the checkpoint state using reducers.
            session_id: Session ID for checkpoint management. If provided and a checkpoint
                exists, continues from last checkpoint. If None, always starts fresh.
            max_steps: Overrides the instance's default `max_steps` for this call.

        Returns:
            Final state after workflow execution
        """
        engine_loop = self._require_built()
        ctx, start_node = await self._new_context(apply_state, session_id, max_steps)

        async for _ in engine_loop.run(ctx, start_node):
            pass

        return ctx.state

    async def stream(
        self,
        apply_state: dict[str, Any] | None = None,
        session_id: str | None = None,
        max_steps: int | None = None,
    ):
        """
        Execute the workflow in streaming mode, starting from START node or continue from last checkpoint.

        Args:
            apply_state: Optional state updates to apply. If continuing from checkpoint,
                these are merged with the checkpoint state using reducers.
            session_id: Session ID for checkpoint management. If provided and a checkpoint
                exists, continues from last checkpoint. If None, always starts fresh.
            max_steps: Overrides the instance's default `max_steps` for this call.

        Yields:
            FlowEngineStreamResponse: One response per engine event.
        """
        engine_loop = self._require_built()
        ctx, start_node = await self._new_context(apply_state, session_id, max_steps)

        async for event in engine_loop.run(ctx, start_node):
            yield self._to_stream_response(event)

    def _to_stream_response(self, event: InternalEvent) -> FlowEngineStreamResponse:
        response = FlowEngineStreamResponse()
        response.type = {
            "start": FlowEngineStreamType.START,
            "node_stream": FlowEngineStreamType.STREAM,
            "node_result": FlowEngineStreamType.RESULT,
        }[event.type]
        response.node = event.node
        response.content = event.content
        response.state = event.state
        response.branch_index = event.branch_index
        response.branch_total = event.branch_total
        response.dispatch_id = event.dispatch_id
        return response

    async def get_state(self, session_id: str) -> dict[str, Any] | None:
        """
        Get the current state for a thread from the last checkpoint.

        Args:
            session_id: The session ID

        Returns:
            The state if checkpoint exists, None otherwise
        """
        if not self.checkpointer:
            raise GraphValidationException("No checkpointer configured")

        checkpoint = await self.checkpointer.load(session_id)
        if checkpoint is None:
            return None
        if self.checkpointer.requires_serialization:
            return deserialize_state(checkpoint.state, self.type_registry)
        return checkpoint.state

    async def list_checkpoints(
        self,
        session_id: str,
        limit: int = 10,
    ) -> list[Checkpoint]:
        """
        List checkpoints for a specific thread.

        Args:
            session_id: The session ID
            limit: Maximum number of checkpoints to return

        Returns:
            List of checkpoints, newest first
        """
        if not self.checkpointer:
            raise GraphValidationException("No checkpointer configured")

        return await self.checkpointer.list(session_id, limit)

    async def get_checkpoint(
        self,
        session_id: str,
        checkpoint_id: str | None = None,
    ) -> Checkpoint | None:
        """
        Get a specific checkpoint or the latest checkpoint for a thread.

        Args:
            session_id: The session ID
            checkpoint_id: Optional checkpoint ID, or None for latest

        Returns:
            The checkpoint if found, None otherwise
        """
        if not self.checkpointer:
            raise GraphValidationException("No checkpointer configured")

        return await self.checkpointer.load(session_id, checkpoint_id)

    async def delete_checkpoints(
        self,
        session_id: str,
        checkpoint_id: str | None = None,
    ):
        """
        Delete checkpoint(s) for a thread.

        Args:
            session_id: The session ID
            checkpoint_id: Optional checkpoint ID to delete, or None to delete all
        """
        if not self.checkpointer:
            raise GraphValidationException("No checkpointer configured")

        await self.checkpointer.delete(session_id, checkpoint_id)

    async def reset_session(self, session_id: str):
        """
        Reset a session by deleting all its checkpoints.
        This allows starting fresh with the same session_id.

        Args:
            session_id: The session ID to reset
        """
        if not self.checkpointer:
            raise GraphValidationException("No checkpointer configured")

        await self.checkpointer.delete(session_id)

    def details(self) -> str:
        """
        Generate a simple details text visualization of the workflow.

        Returns:
            String representation of the workflow graph
        """
        self._require_built()

        lines = ["Workflow Graph:", "=" * 50]

        start_edges = [e for e in self.edges if e.source == START]
        for edge in start_edges:
            for target in edge.targets:
                lines.append(f"START -> {target}")

        function_nodes = [
            n for n in self.nodes.values() if n.node_type == NodeType.FUNCTION
        ]
        if function_nodes:
            lines.append("\nFunction Nodes:")
            for node in function_nodes:
                lines.append(f"  - {node.name}")

        conditional_nodes = [
            n for n in self.nodes.values() if n.node_type == NodeType.CONDITIONAL
        ]
        if conditional_nodes:
            lines.append("\nConditional Nodes:")
            for node in conditional_nodes:
                lines.append(f"  - {node.name}")

        regular_edges = [
            e for e in self.edges if e.condition is None and e.source != START
        ]
        if regular_edges:
            lines.append("\nRegular Edges:")
            for edge in regular_edges:
                for target in edge.targets:
                    lines.append(f"  {edge.source} -> {target}")

        conditional_edges = [e for e in self.edges if e.condition is not None]
        if conditional_edges:
            lines.append("\nConditional Edges:")
            for edge in conditional_edges:
                targets = ", ".join(edge.targets)
                lines.append(f"  {edge.source} ->? [{targets}]")

        return "\n".join(lines)

    def visualize(self) -> str:
        """
        Visualize workflow diagram.
        Generate Mermaid diagram url.

        Returns:
            str: Mermaid URL.
        """
        self._require_built()

        mermaid_code = self.visualizer.create_mermaid_diagram(self)
        return self.visualizer.generate_diagram_url(mermaid_code)
