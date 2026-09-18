"""Unit tests for llmfy/flow_engine/execution/hooks.py (FlowEngineHooks)."""

from llmfy.flow_engine.execution.hooks import FlowEngineHooks


class TestDefaults:
    def test_all_hooks_default_to_none(self):
        hooks = FlowEngineHooks()
        assert hooks.on_node_start is None
        assert hooks.on_node_end is None
        assert hooks.on_error is None


class TestConstruction:
    def test_stores_sync_callables(self):
        calls = []

        def on_start(node_name, state):
            calls.append(("start", node_name, state))

        hooks = FlowEngineHooks(on_node_start=on_start)
        hooks.on_node_start("n1", {"a": 1})  # type: ignore
        assert calls == [("start", "n1", {"a": 1})]

    async def test_stores_async_callables(self):
        calls = []

        async def on_end(node_name, state, updates):
            calls.append(("end", node_name, state, updates))

        hooks = FlowEngineHooks(on_node_end=on_end)
        await hooks.on_node_end("n1", {"a": 1}, {"b": 2})  # type: ignore
        assert calls == [("end", "n1", {"a": 1}, {"b": 2})]

    def test_on_error_receives_node_name_and_exception(self):
        received = {}

        def on_error(node_name, exc):
            received["node_name"] = node_name
            received["exc"] = exc

        hooks = FlowEngineHooks(on_error=on_error)
        exc = ValueError("boom")
        hooks.on_error("n1", exc)  # type: ignore
        assert received == {"node_name": "n1", "exc": exc}
