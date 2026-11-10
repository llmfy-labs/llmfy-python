from llmfy.exception._httpx_compat import (
    resolve_timeout_type,
    timeout_exception_classes,
)
from llmfy.exception.exception_mapper import (
    ANTHROPIC_ERROR_MAP,
    BEDROCK_ERROR_MAP,
    GOOGLE_ERROR_MAP,
    OPENAI_ERROR_MAP,
)
from llmfy.exception.llmfy_exception import (
    LLMfyException,
    TimeoutException,
    TimeoutType,
)
from llmfy.llmfy_core.service_provider import ServiceProvider


def handle_bedrock_error(e) -> LLMfyException:
    """
    Handle AWS Bedrock ClientError exceptions.

    Docs: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
    """
    from botocore.exceptions import ClientError, ConnectTimeoutError, ReadTimeoutError

    if isinstance(e, ReadTimeoutError):
        return TimeoutException(
            message=str(e),
            raw_error=e,
            provider=ServiceProvider.BEDROCK,
            timeout_type=TimeoutType.READ,
        )

    if isinstance(e, ConnectTimeoutError):
        return TimeoutException(
            message=str(e),
            raw_error=e,
            provider=ServiceProvider.BEDROCK,
            timeout_type=TimeoutType.CONNECT,
        )

    if not isinstance(e, ClientError):
        return LLMfyException(
            message=str(e),
            raw_error=e,
            provider=ServiceProvider.BEDROCK,
        )

    error_code = e.response["Error"]["Code"]
    message = e.response["Error"].get("Message", str(e))
    http_status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")

    if error_code in BEDROCK_ERROR_MAP:
        exception_class, default_status = BEDROCK_ERROR_MAP[error_code]
        status_code = http_status or default_status
    else:
        exception_class = LLMfyException
        status_code = http_status

    if exception_class is TimeoutException:
        return TimeoutException(
            message=message,
            status_code=status_code,
            raw_error=e.response,
            provider=ServiceProvider.BEDROCK,
            timeout_type=TimeoutType.MODEL,
        )

    return exception_class(
        message=message,
        status_code=status_code,
        raw_error=e.response,
        provider=ServiceProvider.BEDROCK,
    )


def handle_openai_error(e) -> LLMfyException:
    """
    Handle OpenAI API exceptions.

    Docs: https://github.com/openai/openai-python#handling-errors
    """
    # import openai

    error_type = type(e).__name__

    # Extract status code from APIStatusError subclasses
    status_code = getattr(e, "status_code", None)

    # Build raw error dict
    raw_error = {
        "type": error_type,
        "message": str(e),
    }

    # Add response if available
    if hasattr(e, "response"):
        raw_error["response"] = e.response

    # Add request_id if available
    if hasattr(e, "request_id"):
        raw_error["request_id"] = e.request_id

    # Add body if available
    if hasattr(e, "body"):
        raw_error["body"] = e.body

    if error_type in OPENAI_ERROR_MAP:
        exception_class, default_status = OPENAI_ERROR_MAP[error_type]
        status_code = status_code or default_status
    else:
        exception_class = LLMfyException

    if exception_class is TimeoutException:
        return TimeoutException(
            message=str(e),
            status_code=status_code,
            raw_error=raw_error,
            provider=ServiceProvider.OPENAI,
            timeout_type=resolve_timeout_type(getattr(e, "__cause__", None)),
        )

    return exception_class(
        message=str(e),
        status_code=status_code,
        raw_error=raw_error,
        provider=ServiceProvider.OPENAI,
    )


def handle_anthropic_error(e) -> LLMfyException:
    """
    Handle Anthropic API exceptions.

    Docs: https://github.com/anthropics/anthropic-sdk-python#handling-errors
    """
    error_type = type(e).__name__

    # Extract status code from APIStatusError subclasses
    status_code = getattr(e, "status_code", None)

    # Build raw error dict
    raw_error = {
        "type": error_type,
        "message": str(e),
    }

    # Add response if available
    if hasattr(e, "response"):
        raw_error["response"] = e.response

    # Add request_id if available
    if hasattr(e, "request_id"):
        raw_error["request_id"] = e.request_id

    # Add body if available
    if hasattr(e, "body"):
        raw_error["body"] = e.body

    if error_type in ANTHROPIC_ERROR_MAP:
        exception_class, default_status = ANTHROPIC_ERROR_MAP[error_type]
        status_code = status_code or default_status
    else:
        exception_class = LLMfyException

    if exception_class is TimeoutException:
        return TimeoutException(
            message=str(e),
            status_code=status_code,
            raw_error=raw_error,
            provider=ServiceProvider.ANTHROPIC,
            timeout_type=resolve_timeout_type(getattr(e, "__cause__", None)),
        )

    return exception_class(
        message=str(e),
        status_code=status_code,
        raw_error=raw_error,
        provider=ServiceProvider.ANTHROPIC,
    )


def handle_google_error(e) -> LLMfyException:
    """
    Handle Google Gen AI API exceptions.

    Docs: https://github.com/googleapis/python-genai#error-handling
    """
    from google.genai import errors

    if isinstance(e, timeout_exception_classes()):
        return TimeoutException(
            message=str(e),
            raw_error=e,
            provider=ServiceProvider.GOOGLE,
            timeout_type=resolve_timeout_type(e),
        )

    if isinstance(e, errors.APIError):
        status_code = e.code
        message = e.message

        raw_error = {
            "code": e.code,
            "message": e.message,
        }

        if hasattr(e, "details"):
            raw_error["details"] = e.details

        exception_class = GOOGLE_ERROR_MAP.get(status_code, LLMfyException)

        return exception_class(
            message=message,
            status_code=status_code,
            raw_error=raw_error,
            provider=ServiceProvider.GOOGLE,
        )

    return LLMfyException(
        message=str(e),
        raw_error={"error": str(e)},
        provider=ServiceProvider.GOOGLE,
    )
