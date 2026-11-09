from dotenv import load_dotenv

from llmfy import (
    BedrockConverseModel,
    LLMfy,
    LLMfyException,
    LLMRetryPolicy,
    RetryEvent,
    SDKRetryConfig,
)

load_dotenv()


def log_retry(event: RetryEvent) -> None:
    print(
        f"retry ({event.reason}) on {event.model}: attempt {event.attempt}, "
        f"sleeping {event.delay:.1f}s"
    )


def retry_example():
    llm = BedrockConverseModel(
        model="amazon.nova-lite-v1:0",
        # SDK layer: a few quick attempts for throttling/5xx/network errors.
        sdk_retry=SDKRetryConfig(max_attempts=3, read_timeout=60),
    )

    # LLMfy layer: retry model errors and empty replies, and keep retrying
    # throttling for up to 30 seconds after the SDK gives up.
    chat = LLMfy(
        llm,
        system_message="You are a helpful assistant.",
        retry=LLMRetryPolicy(max_attempts=3, throttle_wait=30, on_retry=log_retry),
    )

    try:
        response = chat.invoke("What is the capital of Indonesia?")
        print(response.result.content)
    except LLMfyException as e:
        print(f"Failed after retries: {e!r}")


if __name__ == "__main__":
    retry_example()
