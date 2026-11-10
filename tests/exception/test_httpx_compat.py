"""Unit tests for llmfy/exception/_httpx_compat.py.

The real-package tests run against whatever httpx-family packages the dev/test
environment has installed (`httpx2` with current provider SDKs). The fallback
tests swap fake `httpx`/`httpx2` modules into `sys.modules`, so the "only old
`httpx` installed", "both installed" and "neither installed" cases are covered
without changing the environment.
"""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from llmfy.exception import _httpx_compat
from llmfy.exception._httpx_compat import (
    httpx_modules,
    resolve_timeout_type,
    timeout_exception_classes,
)
from llmfy.exception.llmfy_exception import TimeoutType


def _fake_httpx_module(name: str) -> ModuleType:
    """A stand-in for `httpx`/`httpx2` with the same timeout class hierarchy."""
    module = ModuleType(name)
    module.TimeoutException = type("TimeoutException", (Exception,), {})  # type: ignore[attr-defined]
    for class_name in ("ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout"):
        setattr(module, class_name, type(class_name, (module.TimeoutException,), {}))  # type: ignore[attr-defined]
    return module


@pytest.fixture
def fresh_cache():
    """Clear the module lookup cache before and after a test that fakes imports."""
    httpx_modules.cache_clear()
    yield
    httpx_modules.cache_clear()


def _install(monkeypatch: pytest.MonkeyPatch, **modules: ModuleType | None) -> None:
    # A `None` entry in sys.modules makes `import name` raise ImportError.
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


class TestInstalledPackages:
    def test_at_least_one_httpx_family_package_is_installed(self):
        # Every supported provider SDK depends on httpx2 or httpx.
        assert httpx_modules()
        assert all(m.__name__ in _httpx_compat.HTTPX_MODULE_NAMES for m in httpx_modules())

    @pytest.mark.parametrize(
        ("class_name", "expected"),
        [
            ("ConnectTimeout", TimeoutType.CONNECT),
            ("ReadTimeout", TimeoutType.READ),
            ("WriteTimeout", TimeoutType.WRITE),
            ("PoolTimeout", TimeoutType.POOL),
        ],
    )
    def test_resolves_each_timeout_class(self, httpx_mod, class_name, expected):
        exc = getattr(httpx_mod, class_name)("timed out")
        assert resolve_timeout_type(exc) == expected
        assert isinstance(exc, timeout_exception_classes())

    def test_base_timeout_exception_has_no_specific_type(self, httpx_mod):
        # Exact-type match: the generic base class is a timeout, but of no known kind.
        exc = httpx_mod.TimeoutException("timed out")
        assert isinstance(exc, timeout_exception_classes())
        assert resolve_timeout_type(exc) is None

    def test_none_and_unrelated_exceptions_resolve_to_none(self):
        assert resolve_timeout_type(None) is None
        assert resolve_timeout_type(ValueError("nope")) is None
        assert not isinstance(ValueError("nope"), timeout_exception_classes())


class TestPackageResolution:
    def test_falls_back_to_httpx_when_httpx2_is_missing(self, monkeypatch, fresh_cache):
        old_httpx = _fake_httpx_module("httpx")
        _install(monkeypatch, httpx2=None, httpx=old_httpx)

        assert httpx_modules() == (old_httpx,)
        assert timeout_exception_classes() == (old_httpx.TimeoutException,)
        assert resolve_timeout_type(old_httpx.WriteTimeout("w")) == TimeoutType.WRITE

    def test_both_packages_are_recognised_when_both_are_installed(
        self, monkeypatch, fresh_cache
    ):
        new, old = _fake_httpx_module("httpx2"), _fake_httpx_module("httpx")
        _install(monkeypatch, httpx2=new, httpx=old)

        assert httpx_modules() == (new, old)
        assert timeout_exception_classes() == (new.TimeoutException, old.TimeoutException)
        assert resolve_timeout_type(new.ReadTimeout("r")) == TimeoutType.READ
        assert resolve_timeout_type(old.PoolTimeout("p")) == TimeoutType.POOL

    def test_neither_package_installed_matches_nothing(self, monkeypatch, fresh_cache):
        _install(monkeypatch, httpx2=None, httpx=None)

        assert httpx_modules() == ()
        assert timeout_exception_classes() == ()
        assert not isinstance(TimeoutError("t"), timeout_exception_classes())
        assert resolve_timeout_type(TimeoutError("t")) is None

    def test_lookup_is_cached(self, monkeypatch, fresh_cache):
        first = _fake_httpx_module("httpx2")
        _install(monkeypatch, httpx2=first, httpx=None)
        assert httpx_modules() == (first,)

        # A later change to what's importable is not picked up until the cache is cleared.
        _install(monkeypatch, httpx2=_fake_httpx_module("httpx2"))
        assert httpx_modules() == (first,)
