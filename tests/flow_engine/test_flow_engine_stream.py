"""Unit tests for FlowEngine.stream()."""

from typing import TypedDict

from llmfy.flow_engine.flow_engine import FlowEngine
from llmfy.flow_engine.node.node import END, START
from llmfy.flow_engine.stream.flow_engine_stream_response import FlowEngineStreamType
from llmfy.flow_engine.stream.node_stream_response import (
    NodeStreamResponse,
    NodeStreamType,
)


class AppState(TypedDict):
    text: str


class CountState(TypedDict):
    count: int


class TestStreamNonStreamingNodes:
    async def test_yields_start_then_result_per_node(self):
        async def a(state):
            return {"text": "a"}

        async def b(state):
            return {"text": "ab"}

        flow = FlowEngine(AppState)
        flow.add_node("a", a)
        flow.add_node("b", b)
        flow.add_edge(START, "a")
        flow.add_edge("a", "b")
        flow.add_edge("b", END)
        flow.build()

        responses = [r async for r in flow.stream()]

        assert responses[0].type == FlowEngineStreamType.START
        result_responses = [
            r for r in responses if r.type == FlowEngineStreamType.RESULT
        ]
        assert [r.node for r in result_responses] == ["a", "b"]
        assert result_responses[-1].state == {"text": "ab"}


class TestStreamStreamingNode:
    async def test_yields_stream_chunks_then_result(self):
        async def streamer(state):
            yield NodeStreamResponse(
                type=NodeStreamType.STREAM, content="he", state=None
            )
            yield NodeStreamResponse(
                type=NodeStreamType.STREAM, content="llo", state=None
            )
            yield NodeStreamResponse(
                type=NodeStreamType.RESULT, content="hello", state={"text": "hello"}
            )

        flow = FlowEngine(AppState)
        flow.add_node("streamer", streamer, stream=True)
        flow.add_edge(START, "streamer")
        flow.add_edge("streamer", END)
        flow.build()

        responses = [r async for r in flow.stream()]

        stream_chunks = [r for r in responses if r.type == FlowEngineStreamType.STREAM]
        result_chunks = [r for r in responses if r.type == FlowEngineStreamType.RESULT]
        assert [r.content for r in stream_chunks] == ["he", "llo"]
        assert len(result_chunks) == 1
        assert result_chunks[0].content == "hello"
        assert result_chunks[0].state == {"text": "hello"}


class TestStreamLoopBackToEntryNode:
    async def test_loop_back_edge_to_entry_node_keeps_looping(self):
        """Same regression as
        test_flow_engine_execution.TestInvokeConditional
        .test_loop_back_edge_to_entry_node_keeps_looping, exercised through
        stream() with both nodes as stream=True — matches the shape of
        flowengine_agent_stream_example.py (`main`/`tools`). `main` is both
        the START target and the target of a regular loop-back edge from
        `tools`, which previously made `is_join("main")` true (START and
        `tools` both counted as predecessors) and silently stopped the
        stream after one tool call instead of looping back into `main`."""

        async def main(state):
            count = state.get("count", 0) + 1
            yield NodeStreamResponse(
                type=NodeStreamType.STREAM, content=str(count), state=None
            )
            yield NodeStreamResponse(
                type=NodeStreamType.RESULT, content=count, state={"count": count}
            )

        async def tools(state):
            yield NodeStreamResponse(
                type=NodeStreamType.STREAM, content="tool", state=None
            )
            yield NodeStreamResponse(type=NodeStreamType.RESULT, content=None, state={})

        def route(state):
            return "tools" if state["count"] < 2 else END

        flow = FlowEngine(CountState)
        flow.add_node("main", main, stream=True)
        flow.add_node("tools", tools, stream=True)
        flow.add_edge(START, "main")
        flow.add_edge("tools", "main")
        flow.add_conditional_edges("main", route, ["tools", END])
        flow.build()

        responses = [r async for r in flow.stream()]

        result_responses = [
            r for r in responses if r.type == FlowEngineStreamType.RESULT
        ]
        assert [r.node for r in result_responses] == ["main", "tools", "main"]
        assert result_responses[-1].state == {"count": 2}
