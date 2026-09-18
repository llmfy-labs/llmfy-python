from llmfy.llmfy_core.messages.message import Message
from llmfy.llmfy_core.messages.role import Role


def tool_trim_messages(messages: list[Message]):
    """Trim messages but ALWAYS preserve active tool context.

    Mechanism:
        Step 1 - Forward scan:
            Iterates all messages to collect:
            - tool_call_ids       : all tool calls made by the assistant
            - resolved_tool_ids   : tool calls that already have a TOOL result
            - last_tool_call_idx  : index of the last assistant message with tool calls
            - last_message_is_tool_result : whether the last message is a TOOL result

        Step 2 - Detect pending tools:
            pending_tool_ids = tool_call_ids - resolved_tool_ids
            Any tool call without a matching result is "pending."

        Step 3 - Two paths:
            - Active tool cycle (pending tools OR last msg is tool result):
                Split at last_tool_call_idx, protect everything from there onward,
                trim earlier messages down to the last non-TOOL anchor to avoid
                starting with an orphaned TOOL result.
            - No active tool cycle:
                Aggressively trim — keep only messages[-1:].

    Example:
        messages = [
            0: USER   "search for X"
            1: ASSIST  [tool_call: search, id="tc1"]
            2: TOOL    [tool_call_id="tc1", result="..."]
            3: ASSIST  "Based on results..." [tool_call: lookup, id="tc2"]  <- last_tool_call_idx
            4: TOOL    [tool_call_id="tc2", result="..."]  <- last_message_is_tool_result=True
        ]

        tool_call_ids     = {tc1, tc2}
        resolved_tool_ids = {tc1, tc2}
        pending_tool_ids  = {}  (empty, but last_message_is_tool_result=True -> protect context)

        protected = messages[3:] -> [ASSIST tc2, TOOL tc2]
        trimmable = messages[:3] -> [USER, ASSIST tc1, TOOL tc1]
        anchor    = index 1 (last non-TOOL in trimmable) -> ASSIST tc1

        result = [ASSIST tc1, TOOL tc1] + [ASSIST tc2, TOOL tc2]
        # USER "search for X" is dropped; active tool context is fully preserved.

    Example 2 (parallel tool calls in a single ASSIST message):
        messages = [
            0: USER   "search for X"
            1: ASSIST  [tool_call: search, id="tc1", tool_call: search, id="tc2"]  <- last_tool_call_idx
            2: TOOL    [tool_call_id="tc1", result="..."]
            3: TOOL    [tool_call_id="tc2", result="..."]  <- last_message_is_tool_result=True
        ]

        tool_call_ids     = {tc1, tc2}
        resolved_tool_ids = {tc1, tc2}
        pending_tool_ids  = {}  (empty, but last_message_is_tool_result=True -> protect context)

        Condition: (False) or True -> True  -> enters protect-context path

        protected = messages[1:] -> [ASSIST(tc1,tc2), TOOL(tc1), TOOL(tc2)]
        trimmable = messages[:1] -> [USER]
        anchor    = index 0 (USER is not TOOL, no skip needed)

        result = [USER] + [ASSIST(tc1,tc2), TOOL(tc1), TOOL(tc2)]
        # All 4 messages preserved; parallel tool results stay intact.
    """
    if len(messages) == 1:
        return messages

    # Step 1: FORWARD pass - collect all tool_call_ids and their results
    tool_call_ids = set()  # All tool calls made
    resolved_tool_ids = set()  # Tool calls that have results
    last_tool_call_idx = None  # Last index tool calling in messages
    last_message_is_tool_result = False

    for i, msg in enumerate(messages):
        # Track tool calls (assistant messages)
        if msg.role == Role.ASSISTANT and msg.tool_calls:
            last_tool_call_idx = i
            for tc in msg.tool_calls:
                tool_call_ids.add(tc.tool_call_id)

        # Track tool results
        if msg.role == Role.TOOL:
            if msg.tool_call_id:
                resolved_tool_ids.add(msg.tool_call_id)

    # Check if last message is a tool result
    if messages and messages[-1].role == Role.TOOL:
        last_message_is_tool_result = True

    # Step 2: Calculate pending tools
    pending_tool_ids = tool_call_ids - resolved_tool_ids

    # Step 3: If there are pending tools OR last message is tool result, protect the tool context
    # This is the KEY FIX: Even if tools are "resolved", if we just got a tool result,
    # we need to preserve the context for the orchestrator to process
    if (
        pending_tool_ids and last_tool_call_idx is not None
    ) or last_message_is_tool_result:
        # There are active tool cycle or just completed tool execution

        # If last message is tool result, we must preserve from the assistant message that called it
        if last_message_is_tool_result and last_tool_call_idx is not None:
            protected_messages = messages[last_tool_call_idx:]
            trimmable_messages = messages[:last_tool_call_idx]
        else:
            # Pending tools case
            protected_messages = messages[last_tool_call_idx:]
            trimmable_messages = messages[:last_tool_call_idx]

        if not trimmable_messages:
            return protected_messages

        try:
            # Find the last USER message as the anchor so we never start with an
            # orphaned tool_result (which happens in multi-step tool chains when
            # trimmable_messages[-1] is a Role.TOOL message from a prior tool call).
            anchor_idx = len(trimmable_messages) - 1
            while anchor_idx >= 0 and trimmable_messages[anchor_idx].role == Role.TOOL:
                anchor_idx -= 1

            if anchor_idx < 0:
                return protected_messages

            return trimmable_messages[anchor_idx:] + protected_messages

        except Exception:
            return protected_messages

    else:
        return messages[-1:]
