"""
SQL checkpointer example — persists checkpoints to a real database.

Uses SQLite (`sqlite:///...`, built into Python/SQLAlchemy, no extra driver
needed) so this runs standalone. Swap the connection string for Postgres/
MySQL in production — see `SQLCheckpointer`'s docstring for connection
string examples per database/driver.

SQLCheckpointer persists outside the current process, so any custom type
in state (here, `Message`) must be registered via `FlowEngine(types=[...])`
before it can be saved — see `checkpointer/serde.py` for why (an
allow-list, not blind deserialization off whatever a stored checkpoint
happens to name).
"""

import asyncio
from pathlib import Path
from typing import Annotated, TypedDict

from llmfy import END, START, FlowEngine, Message, Role
from llmfy.flow_engine.checkpointer.sql_checkpointer import SQLCheckpointer


def append_messages(old: list[Message] | None, new: list[Message]) -> list[Message]:
    return (old or []) + new


class ChatState(TypedDict):
    messages: Annotated[list[Message], append_messages]


async def echo_node(state: ChatState) -> dict:
    last = state["messages"][-1]
    reply = Message(role=Role.ASSISTANT, content=f"echo: {last.content}")
    return {"messages": [reply]}


async def main():
    db_path = Path(__file__).parent / "flowengine_sql_checkpointer_example.db"
    checkpointer = SQLCheckpointer(f"sqlite:///{db_path}")

    flow = FlowEngine(ChatState, checkpointer=checkpointer, types=[Message])
    flow.add_node("echo", echo_node)
    flow.add_edge(START, "echo")
    flow.add_edge("echo", END)
    flow.build()

    session_id = "sql-demo"
    user_message = Message(role=Role.USER, content="hello from SQLCheckpointer")

    result = await flow.invoke(
        apply_state={"messages": [user_message]}, session_id=session_id
    )
    for msg in result["messages"]:
        print(f"{msg.role}: {msg.content}")

    # Reload straight from the database to prove it persisted.
    persisted = await flow.get_state(session_id)
    print(f"\nPersisted message count: {len(persisted['messages'])}")  # type: ignore

    await checkpointer.close()


if __name__ == "__main__":
    asyncio.run(main())
