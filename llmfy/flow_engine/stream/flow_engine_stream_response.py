from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class FlowEngineStreamType(StrEnum):
    """FlowEngineStreamType"""
    START = "start"
    STREAM = "stream"
    RESULT = "result"
    ERROR = "error"


class FlowEngineStreamResponse(BaseModel):
    """FlowEngineStreamResponse"""
    type: str | None = Field(default=None)
    node: str | None = Field(default=None)
    content: Any | None = Field(default=None)
    state: dict | None = Field(default=None)
    error: Any | None = Field(default=None)
    branch_index: int | None = Field(
        default=None,
        description="Position of this event's branch within its dynamic "
        "(`Send`) fan-out dispatch (0-based). `None` outside a dynamic "
        "fan-out — a plain node or a static fan-out branch has a distinct "
        "`node` name already, so it doesn't need this.",
    )
    branch_total: int | None = Field(
        default=None,
        description="Number of branches in this event's dynamic fan-out "
        "dispatch. `None` outside a dynamic fan-out.",
    )
    dispatch_id: str | None = Field(
        default=None,
        description="Identifies one dynamic-fan-out dispatch (one routing "
        "call that returned Send(s)) — shared by every branch of that "
        "dispatch, distinct across separate dispatches even to the same "
        "node. `None` outside a dynamic fan-out.",
    )
