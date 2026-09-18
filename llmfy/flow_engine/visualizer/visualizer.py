import base64

from llmfy.flow_engine.node.node import END, START


class WorkflowVisualizer:
    @staticmethod
    def create_mermaid_diagram(workflow) -> str:
        """Generate Mermaid diagram markdown from workflow."""
        nodes = workflow.nodes
        edges = workflow.edges
        
        # Start building the diagram
        mermaid = [
            "graph TD",
            "    %% Node Styles",
            "    %% Nodes",
        ]
        
        # Add all nodes
        for name in nodes:
            # Style nodes based on their type
            if name == START:
                mermaid.append(f"    {name}([{name}])")
                # mermaid.append(f"    class {name} start")
            elif name == END:
                mermaid.append(f"    {name}([{name}])")
                # mermaid.append(f"    class {name} end")
            else:
                mermaid.append(f"    {name}({name})")
        
        mermaid.append("    %% Edges")
        
        # Add all edges
        for edge in edges:
            source = edge.source
            
            if edge.condition is None:
                # Simple edges
                for target in edge.targets:
                    mermaid.append(f"    {source} --> {target}")
            elif edge.target_map is not None:
                # Conditional edges with a {return_value: target} map — label
                # each arrow with the key that routes to it.
                for label, target in edge.target_map.items():
                    mermaid.append(f"    {source} -.->|{label}| {target}")
            else:
                # Conditional edges
                for target in edge.targets:
                    mermaid.append(f"    {source} -.->|condition| {target}")
        
        mermaid.append("style __start__ fill:#189C82,stroke:#066F62,color:white")
        mermaid.append("style __end__ fill:#189C82,stroke:#066F62,color:white")

        return "\n".join(mermaid)

    @staticmethod
    def generate_diagram_url(mermaid_code: str) -> str:
        """Generate URL for Mermaid diagram image."""
        encoded = base64.b64encode(mermaid_code.encode()).decode()
        return f"https://mermaid.ink/img/{encoded}"