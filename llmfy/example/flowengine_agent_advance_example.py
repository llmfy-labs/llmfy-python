"""
Tool-calling LLM agent built on FlowEngine, driven with `stream()`, also
demonstrating both fan-out styles on the tool-execution path:

- DYNAMIC fan-out (`Send`): when the model requests N tool calls in one
  turn, `route_after_main` returns one `Send("execute_tool", ...)` per
  call — they run CONCURRENTLY (not the sequential loop a single node
  would do), and their results commit atomically: if any tool call raises,
  none of that turn's tool results are merged (see `Send`'s docstring).
  Branch count is only known at runtime, from however many tool calls the
  model happened to request — the textbook case for dynamic over static
  fan-out. See flowengine_dynamic_fan_out_example.py for a simpler,
  narrated version of this same mechanism.
- STATIC fan-out: `execute_tool`'s own outgoing edge is fixed at
  build-time to two targets, `audit_log` and `safety_check` — they always
  both run, once, after every tool-calling round (whether that round
  dispatched one tool call or five), joining back into `after_tools`
  before looping to `main`. See flowengine_fan_out_example.py for a
  simpler, narrated version of this same mechanism.

`main_orchestrator` is a `stream=True` async generator yielding
`NodeStreamResponse` — partial content streams to the terminal as it
arrives, and the final `NodeStreamResponse` (type=RESULT) carries the
state update. When the model's `thinking`/`reasoning` config is enabled
(see docs/documentation/features/thinking-config.md), its reasoning trace
streams too, wrapped in a `ThinkingChunk` so the terminal can tell it apart
from the model's actual answer — same `[thinking]...[/thinking]` framing as
thinking_example.py, just routed through a FlowEngine node instead of
`LLMfy` directly. `execute_tool`/`audit_log`/`safety_check`/`after_tools`
are plain (non-streaming) nodes — each produces exactly one RESULT event,
no incremental STREAM chunks.

Requires a Google AI API key (`GOOGLE_API_KEY` in the environment / a
`.env` file) — see flowengine_agent_example.py for the non-streaming
equivalent.
"""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, TypedDict

# from urllib.parse import quote_plus
from dotenv import load_dotenv

from llmfy import (
    END,
    START,
    # AnthropicMessagesConfig,
    # AnthropicMessagesModel,
    # AnthropicMessagesThinkingConfig,
    FlowEngine,
    GenerationResponse,
    # GoogleAIGenerateConfig,
    # GoogleAIGenerateModel,
    LLMfy,
    Message,
    NodeStreamResponse,
    NodeStreamType,
    OpenAIChatConfig,
    OpenAIChatModel,
    RedisCheckpointer,
    # OpenAIResponsesConfig,
    # OpenAIResponsesModel,
    # OpenAIResponsesReasoningConfig,
    Role,
    # SQLCheckpointer,
    Send,
    Tool,
    ToolRegistry,
    tool_trim_messages,
)
from llmfy.flow_engine.stream.flow_engine_stream_response import FlowEngineStreamType

load_dotenv()

OUTPUT_DIR = Path(__file__).parent / "output"


def _dump_messages(messages: list[Message], path: Path) -> None:
    path.write_text(
        json.dumps([m.model_dump(mode="json") for m in messages], indent=2)
    )


def append_messages(old: list[Message] | None, new: list[Message]) -> list[Message]:
    return (old or []) + new


@dataclass
class ThinkingChunk:
    """Marks a streamed `NodeStreamResponse.content` as reasoning/thinking
    text rather than the model's final answer, so `main()`'s terminal
    consumer can tell the two apart via `isinstance` and print each
    differently."""

    content: str


class AppState(TypedDict):
    messages: Annotated[list[Message], append_messages]


@Tool()
def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather for a location."""
    return f"The weather in {location} is 22 degrees {unit}"


@Tool()
def get_current_time(location: str) -> str:
    """Get the current time for a location."""
    return f"The time in {location} is 09:00 AM"


def build_agent() -> FlowEngine:
    # model = GoogleAIGenerateModel(
    #     model="gemini-2.5-flash-lite",
    #     config=GoogleAIGenerateConfig(temperature=0.7),
    # )
    # model = OpenAIResponsesModel(
    #     model="gpt-5",
    #     config=OpenAIResponsesConfig(
    #         temperature=None,
    #         reasoning=OpenAIResponsesReasoningConfig(enabled=True, effort="medium", summary="auto")
    #     ),
    # )
    # model = AnthropicMessagesModel(
    #     model="claude-haiku-4-5",
    #     config=AnthropicMessagesConfig(
    #         thinking=AnthropicMessagesThinkingConfig(enabled=True, budget_tokens=1024)
    #     ),
    # )
    model = OpenAIChatModel(
        model="gemma4:e4b",
        # model="qwen3.5:9b",
        config=OpenAIChatConfig(temperature=0.7),
        api_key="ollama",  # unused by Ollama, but required by the openai SDK client
        base_url="http://localhost:11434/v1",  # Ollama proxy
    )
    llm = LLMfy(model, system_message="You are a helpful assistant.")

    tools = [get_current_weather, get_current_time]
    llm.register_tool(tools)
    tool_registry = ToolRegistry(tools, model)

    turn_counter = 0

    async def main_orchestrator(state: AppState):
        nonlocal turn_counter
        turn_counter += 1

        full_content = ""
        last_message: Message | None = None

        msgs = tool_trim_messages(state["messages"])

        # DEBUG INPUT REQUEST
        OUTPUT_DIR.mkdir(exist_ok=True)
        # Safe (untrimmed) vs tool-trimmed messages, dumped per turn for comparison.
        _dump_messages(
            state["messages"], OUTPUT_DIR / f"messages_{turn_counter}.json"
        )
        _dump_messages(msgs, OUTPUT_DIR / f"tool_trim_messages_{turn_counter}.json")

        async for chunk in llm.achat_stream(msgs):
            if not isinstance(chunk, GenerationResponse):
                continue
            if chunk.messages:
                last_message = chunk.messages[-1]
            if chunk.result.thinking:
                yield NodeStreamResponse(
                    type=NodeStreamType.STREAM,
                    content=ThinkingChunk(content=chunk.result.thinking),
                )
            if chunk.result.content:
                full_content += chunk.result.content
                yield NodeStreamResponse(
                    type=NodeStreamType.STREAM, content=chunk.result.content
                )

        yield NodeStreamResponse(
            type=NodeStreamType.RESULT,
            content=full_content,
            state={"messages": [last_message]},
        )

    async def execute_tool(state: dict) -> dict:
        # DYNAMIC fan-out target: `state` here is exactly one Send's own
        # `{"tool_call": ...}` — not AppState, and not the other tool
        # calls dispatched alongside it in this same turn (see `Send`'s
        # docstring). Runs concurrently with any sibling tool calls.
        tool_call = state["tool_call"]
        result = tool_registry.execute_tool(
            name=tool_call.name, arguments=tool_call.arguments
        )
        tool_message = Message(
            role=Role.TOOL,
            request_call_id=tool_call.request_call_id,
            tool_call_id=tool_call.tool_call_id,
            tool_results=[str(result)],
            name=tool_call.name,
        )
        return {"messages": [tool_message]}

    async def audit_log(state: AppState) -> dict:
        # STATIC fan-out branch #1 — always runs once per tool-calling
        # round, after every dispatched tool call has committed (however
        # many there were), never once per tool call.
        tool_messages = [m for m in state["messages"] if m.role == Role.TOOL]
        print(f"\n[audit_log] {len(tool_messages)} tool call(s) recorded so far\n")
        return {}

    async def safety_check(state: AppState) -> dict:
        # STATIC fan-out branch #2 — runs concurrently with audit_log.
        print("\n[safety_check] no policy violations detected\n")
        return {}

    async def after_tools(state: AppState) -> dict:
        # Join: waits for BOTH audit_log and safety_check before running.
        return {}

    def route_after_main(state: AppState) -> list[Send] | str:
        last_message = state["messages"][-1]
        if not last_message.tool_calls:
            return END
        # One Send per requested tool call — the count is only known here,
        # at runtime, from however many calls the model happened to
        # request this turn.
        return [
            Send("execute_tool", {"tool_call": tc})
            for tc in last_message.tool_calls
        ]

    checkpointer = RedisCheckpointer(
        redis_url="redis://localhost:6381/0",
        prefix="agent-x:",
        ttl=3600,  # checkpoints expire after 1 hour
    )
    # Password is percent-encoded since it contains `@`, which would
    # otherwise be parsed as the userinfo/host separator in the URL.
    # mysql_password = quote_plus("Your_password")
    # checkpointer = SQLCheckpointer(
    #     f"mysql+pymysql://root:{mysql_password}@localhost:3306/agent"
    # )

    # max_steps arithmetic per tool-calling round: N `execute_tool`
    # branches (N = tool calls requested this turn) + audit_log (1) +
    # safety_check (1) + after_tools (1) + the next "main" (1) = N + 4.
    # Default max_steps=100 easily covers many rounds of a few tools each;
    # widen it if a single turn can request very many tool calls.
    flow = FlowEngine(AppState, checkpointer=checkpointer, types=[Message])
    flow.add_node("main", main_orchestrator, stream=True)
    flow.add_node("execute_tool", execute_tool)
    flow.add_node("audit_log", audit_log)
    flow.add_node("safety_check", safety_check)
    flow.add_node("after_tools", after_tools)
    flow.add_edge(START, "main")
    flow.add_conditional_edges("main", route_after_main, ["execute_tool", END])
    flow.add_edge("execute_tool", ["audit_log", "safety_check"])  # static fan-out
    flow.add_edge("audit_log", "after_tools")  # join
    flow.add_edge("safety_check", "after_tools")
    flow.add_edge("after_tools", "main")  # loop back

    return flow.build()


async def main():
    agent = build_agent()
    session_id = "agent-x-session"

    print(agent.visualize())
    print("=== FlowEngine agent (stream) — type 'exit' to quit ===\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        print("Assistant: ", end="", flush=True)
        is_thinking = False
        async for response in agent.stream(
            apply_state={"messages": [Message(role=Role.USER, content=user_input)]},
            session_id=session_id,
        ):
            if response.type == FlowEngineStreamType.STREAM:
                if isinstance(response.content, ThinkingChunk):
                    if not is_thinking:
                        print("\n-----[thinking]-----\n", end="", flush=True)
                        is_thinking = True
                    print(response.content.content, end="", flush=True)
                    continue
                if is_thinking:
                    print("\n-----[/thinking]-----\n\n", end="", flush=True)
                    is_thinking = False
                if isinstance(response.content, str):
                    print(response.content, end="", flush=True)
            elif response.type == FlowEngineStreamType.RESULT:
                if is_thinking:
                    # A tool-calling round can end in RESULT (tool_calls,
                    # no plain content) without ever hitting the STREAM/str
                    # branch above, so the close tag must also be forced
                    # here — otherwise execute_tool/audit_log/safety_check
                    # output (printed while this round's tools run) would
                    # visually land inside an unclosed [thinking] block.
                    print("\n-----[/thinking]-----\n\n", end="", flush=True)
                    is_thinking = False
                if response.node == "execute_tool":
                    # DYNAMIC fan-out: one RESULT event per Send branch —
                    # branch_index/branch_total/dispatch_id are the only
                    # way to tell "tool call 2 of 3" apart from another
                    # when every branch shares the same node name.
                    tool_message = response.content["messages"][0]  # type: ignore
                    branch = ""
                    if response.branch_index is not None:
                        branch = f" ({response.branch_index + 1}/{response.branch_total})"
                    print(
                        f"\n[tool result{branch}: {tool_message.name} -> "
                        f"{tool_message.tool_results}]\n",
                        flush=True,
                    )
        print("\n")


if __name__ == "__main__":
    asyncio.run(main())
