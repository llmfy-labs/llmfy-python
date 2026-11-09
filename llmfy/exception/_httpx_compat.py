"""Resolve httpx-family timeout exceptions across `httpx` and `httpx2`.

Recent provider SDK releases (openai 3.x, anthropic 2.x, google-genai 3.x)
moved from `httpx` to `httpx2`, a drop-in fork published under a separate
import name. Its classes are *not* subclasses of the `httpx` ones, so code
that checks for `httpx.ReadTimeout` never matches an `httpx2.ReadTimeout`.
Older SDK releases still raise `httpx` exceptions, and google-genai works with
either package, so both can be installed side by side.

This module imports whichever of the two are installed and exposes their
timeout classes together. Nothing here imports at module load time, and
neither package is a hard dependency of llmfy.
"""

from __future__ import annotations

import importlib
from functools import cache
from types import ModuleType

from llmfy.exception.llmfy_exception import TimeoutType

# `httpx2` first: it is what current provider SDK releases use.
HTTPX_MODULE_NAMES = ("httpx2", "httpx")

_TIMEOUT_TYPE_BY_CLASS_NAME = {
    "ConnectTimeout": TimeoutType.CONNECT,
    "ReadTimeout": TimeoutType.READ,
    "WriteTimeout": TimeoutType.WRITE,
    "PoolTimeout": TimeoutType.POOL,
}


@cache
def httpx_modules() -> tuple[ModuleType, ...]:
    """Return every installed httpx-family module, in `HTTPX_MODULE_NAMES` order.

    Cached, so a missing package is looked up once rather than on every
    request. Empty when neither `httpx2` nor `httpx` is installed.
    """
    modules = []
    for name in HTTPX_MODULE_NAMES:
        try:
            modules.append(importlib.import_module(name))
        except ImportError:
            continue
    return tuple(modules)


def timeout_exception_classes() -> tuple[type[BaseException], ...]:
    """Return the `TimeoutException` base class of every installed httpx-family module.

    Usable directly in `except` clauses and `isinstance` checks. An empty
    tuple matches nothing.
    """
    return tuple(module.TimeoutException for module in httpx_modules())


def resolve_timeout_type(exc: BaseException | None) -> TimeoutType | None:
    """Map an httpx/httpx2 timeout exception to its `TimeoutType`.

    Matches on the exception's exact class (`ConnectTimeout`, `ReadTimeout`,
    `WriteTimeout`, `PoolTimeout`). Returns `None` for anything else,
    including `None`.
    """
    if exc is None:
        return None
    for module in httpx_modules():
        for class_name, timeout_type in _TIMEOUT_TYPE_BY_CLASS_NAME.items():
            if type(exc) is getattr(module, class_name):
                return timeout_type
    return None
