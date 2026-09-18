"""Contract test for llmfy/__init__.py — the sole place that defines the
package's public API surface. A regression here (a name in
`__all__` that fails to import, or silently vanishing) breaks every
downstream `from llmfy import X` user has, so this is cheap insurance with
outsized value — and it's exactly the kind of "stable public contract" a
future port to another language needs to replicate faithfully (see
PORTABILITY.md).
"""

import llmfy


def test_every_name_in_all_is_actually_importable():
    for name in llmfy.__all__:
        assert hasattr(llmfy, name), f"'{name}' is listed in __all__ but not importable from llmfy"


def test_all_has_no_duplicate_entries():
    assert len(llmfy.__all__) == len(set(llmfy.__all__))


def test_core_chat_api_is_exported():
    for name in ["LLMfy", "Message", "Role", "Tool", "ToolRegistry", "AIResponse", "GenerationResponse"]:
        assert name in llmfy.__all__


def test_every_provider_model_and_config_is_exported():
    for name in [
        "OpenAIChatModel",
        "OpenAIChatConfig",
        "OpenAIResponsesModel",
        "OpenAIResponsesConfig",
        "AnthropicMessagesModel",
        "AnthropicMessagesConfig",
        "BedrockConverseModel",
        "BedrockConverseConfig",
        "GoogleAIGenerateModel",
        "GoogleAIGenerateConfig",
    ]:
        assert name in llmfy.__all__


def test_full_exception_hierarchy_is_exported():
    for name in [
        "LLMfyException",
        "AuthenticationException",
        "ContentFilterException",
        "InvalidRequestException",
        "ModelErrorException",
        "ModelNotFoundException",
        "PermissionDeniedException",
        "QuotaExceededException",
        "RateLimitException",
        "ServiceUnavailableException",
        "TimeoutException",
        "GraphValidationException",
        "NodeExecutionException",
        "NodeTimeoutException",
        "StepLimitExceededException",
        "CheckpointDeserializationException",
        "CheckpointPayloadTooLargeException",
        "InvalidSessionIdException",
    ]:
        assert name in llmfy.__all__
        assert issubclass(getattr(llmfy, name), llmfy.LLMfyException)


def test_chunking_and_text_preprocessing_utilities_exported():
    for name in ["chunk_text", "chunk_markdown_by_header", "clean_text_for_embedding"]:
        assert name in llmfy.__all__


def test_version_is_exported_and_is_a_string():
    assert "__version__" in llmfy.__all__
    assert isinstance(llmfy.__version__, str)


def test_flow_engine_core_names_are_exported_and_deprecated_state_helpers_are_gone():
    # flow_engine's rewrite has its own full test suite under
    # tests/flow_engine/ — this just guards the top-level public contract.
    for name in [
        "FlowEngine",
        "Edge",
        "Node",
        "START",
        "END",
        "RetryPolicy",
        "FlowEngineHooks",
        "Send",
        "BaseCheckpointer",
        "InMemoryCheckpointer",
        "RedisCheckpointer",
        "SQLCheckpointer",
    ]:
        assert name in llmfy.__all__

    # WorkflowState/MemoryManager were deprecated, dead legacy code, removed
    # entirely as part of the flow_engine rewrite — regression guard against
    # them being reintroduced.
    for name in ["WorkflowState", "MemoryManager"]:
        assert name not in llmfy.__all__
        assert not hasattr(llmfy, name)
