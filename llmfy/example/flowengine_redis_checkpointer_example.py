"""
Redis checkpointer example — persists checkpoints to Redis with a TTL.

Requires a local Redis server (e.g. `docker run -p 6379:6379 redis`) and
the `redis` optional extra (`pip install "llmfy[redis]"`).

Like SQLCheckpointer, RedisCheckpointer persists outside the current
process, so custom types in state must be registered via
`FlowEngine(types=[...])` before they can be saved.
"""

import asyncio
from typing import Annotated, TypedDict

from llmfy import END, START, FlowEngine, Message, Role
from llmfy.flow_engine.checkpointer.redis_checkpointer import RedisCheckpointer


def append_messages(old: list[Message] | None, new: list[Message]) -> list[Message]:
    return (old or []) + new


class ChatState(TypedDict):
    messages: Annotated[list[Message], append_messages]


async def echo_node(state: ChatState) -> dict:
    last = state["messages"][-1]
    reply = Message(role=Role.ASSISTANT, content=f"echo: {last.content}")
    return {"messages": [reply]}


async def main():
    checkpointer = RedisCheckpointer(
        redis_url="redis://localhost:6381/0",
        prefix="flowengine:example:",
        ttl=3600,  # checkpoints expire after 1 hour
    )

    flow = FlowEngine(ChatState, checkpointer=checkpointer, types=[Message])
    flow.add_node("echo", echo_node)
    flow.add_edge(START, "echo")
    flow.add_edge("echo", END)
    flow.build()

    session_id = "redis-demo"
    user_message = Message(role=Role.USER, content="hello from RedisCheckpointer")

    result = await flow.invoke(
        apply_state={"messages": [user_message]}, session_id=session_id
    )
    for msg in result["messages"]:
        print(f"{msg.role}: {msg.content}")

    persisted = await flow.get_state(session_id)
    print(f"\nPersisted message count: {len(persisted['messages'])}")  # type: ignore

    await checkpointer.close()


if __name__ == "__main__":
    asyncio.run(main())
