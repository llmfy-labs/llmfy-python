"""Unit tests for llmfy/flow_engine/visualizer/visualizer.py."""

import base64
from types import SimpleNamespace

from llmfy.flow_engine.edge.edge import Edge
from llmfy.flow_engine.node.node import END, START
from llmfy.flow_engine.visualizer.visualizer import WorkflowVisualizer


def make_workflow(nodes, edges):
    return SimpleNamespace(nodes=nodes, edges=edges)


class TestCreateMermaidDiagram:
    def test_includes_all_node_names(self):
        workflow = make_workflow(
            nodes=[START, "a", "b", END],
            edges=[Edge(START, "a"), Edge("a", "b"), Edge("b", END)],
        )
        mermaid = WorkflowVisualizer.create_mermaid_diagram(workflow)

        assert "graph TD" in mermaid
        assert "a(a)" in mermaid
        assert "b(b)" in mermaid
        assert f"{START}([{START}])" in mermaid
        assert f"{END}([{END}])" in mermaid

    def test_regular_edge_uses_solid_arrow(self):
        workflow = make_workflow(nodes=["a", "b"], edges=[Edge("a", "b")])
        mermaid = WorkflowVisualizer.create_mermaid_diagram(workflow)
        assert "a --> b" in mermaid

    def test_conditional_edge_uses_dashed_arrow_with_label(self):
        def cond(_state):
            return "b"

        workflow = make_workflow(
            nodes=["a", "b", "c"], edges=[Edge("a", ["b", "c"], condition=cond)]
        )
        mermaid = WorkflowVisualizer.create_mermaid_diagram(workflow)
        assert "a -.->|condition| b" in mermaid
        assert "a -.->|condition| c" in mermaid

    def test_fan_out_edge_produces_one_arrow_per_target(self):
        workflow = make_workflow(
            nodes=["a", "b", "c"], edges=[Edge("a", ["b", "c"])]
        )
        mermaid = WorkflowVisualizer.create_mermaid_diagram(workflow)
        assert "a --> b" in mermaid
        assert "a --> c" in mermaid

    def test_target_map_conditional_edge_labels_arrows_with_dict_keys(self):
        def cond(_state):
            return "Pass"

        workflow = make_workflow(
            nodes=["a", "b", "c"],
            edges=[Edge("a", ["b", "c"], condition=cond, target_map={"Pass": "b", "Fail": "c"})],
        )
        mermaid = WorkflowVisualizer.create_mermaid_diagram(workflow)
        assert "a -.->|Pass| b" in mermaid
        assert "a -.->|Fail| c" in mermaid
        assert "|condition|" not in mermaid


class TestGenerateDiagramUrl:
    def test_produces_mermaid_ink_url_with_base64_payload(self):
        code = "graph TD\n    a --> b"
        url = WorkflowVisualizer.generate_diagram_url(code)

        assert url.startswith("https://mermaid.ink/img/")
        encoded = url.removeprefix("https://mermaid.ink/img/")
        assert base64.b64decode(encoded).decode() == code
