"""Tests for the DuckDuckGo (ddgs) web search provider.

Covers:
- DDGSWebSearchProvider.is_available() — reflects package importability
- DDGSWebSearchProvider.search() — happy path, missing package, runtime error
- Result normalization (title, url, description, position)
- Process-isolated timeout / interrupt / GIL-hold / reap (#68096)
- _is_backend_available("ddgs") / _get_backend() integration
- web_extract returns a search-only error when ddgs is active
"""
from __future__ import annotations

import builtins
import json
import sys
import time
import types

import pytest

from tests.tools.conftest import register_all_web_providers


def _install_fake_ddgs(monkeypatch, *, text_results=None, text_raises=None, text_sleep=None):
    """Install a stub ``ddgs`` module in sys.modules for the duration of a test.

    ``text_results``: iterable of dicts to yield from DDGS().text(...).
    ``text_raises``: if set, DDGS().text raises this exception instead.
    ``text_sleep``: if set, DDGS().text blocks for this many seconds before
        yielding — simulates a hung/slow search for the timeout test.
    """
    import time as _time

    fake = types.ModuleType("ddgs")

    class _FakeDDGS:
        def __init__(self, **kwargs):
            # Accept timeout= (and any other constructor kwargs) — the provider
            # now passes DDGS(timeout=10).
            pass
        def __enter__(self):
            return self
        def __exit__(self, *_a):
            return False
        def text(self, query, max_results=5):
            if text_sleep is not None:
                _time.sleep(text_sleep)
            if text_raises is not None:
                raise text_raises
            for hit in (text_results or []):
                yield hit

    fake.DDGS = _FakeDDGS
    monkeypatch.setitem(sys.modules, "ddgs", fake)
    return fake


def _force_inprocess_search(monkeypatch, prov):
    """Route bounded search through the in-process helper.

    Happy-path unit tests install a fake ``ddgs`` in the parent interpreter;
    spawn workers would not see that fake. Isolation behavior is covered by
    dedicated process tests below.
    """
    monkeypatch.setattr(
        prov,
        "_run_ddgs_search_bounded",
        lambda query, safe_limit: prov._run_ddgs_search(query, safe_limit),
        raising=True,
    )


# ---------------------------------------------------------------------------
# DDGSWebSearchProvider unit tests
# ---------------------------------------------------------------------------


class TestDDGSProviderIsConfigured:
    def test_configured_when_package_importable(self, monkeypatch):
        _install_fake_ddgs(monkeypatch)
        # Drop any cached ``plugins.web.ddgs.provider`` so is_configured re-imports ddgs fresh
        monkeypatch.delitem(sys.modules, "plugins.web.ddgs.provider", raising=False)
        from plugins.web.ddgs.provider import DDGSWebSearchProvider
        assert DDGSWebSearchProvider().is_available() is True


    def test_implements_web_search_provider(self):
        from agent.web_search_provider import WebSearchProvider
        from plugins.web.ddgs.provider import DDGSWebSearchProvider
        assert issubclass(DDGSWebSearchProvider, WebSearchProvider)


class TestDDGSProviderSearch:
    def test_happy_path_normalizes_results(self, monkeypatch):
        _install_fake_ddgs(monkeypatch, text_results=[
            {"title": "A", "href": "https://a.example.com", "body": "desc A"},
            {"title": "B", "href": "https://b.example.com", "body": "desc B"},
            {"title": "C", "href": "https://c.example.com", "body": "desc C"},
        ])
        import plugins.web.ddgs.provider as prov
        _force_inprocess_search(monkeypatch, prov)

        result = prov.DDGSWebSearchProvider().search("q", limit=5)

        assert result["success"] is True
        web = result["data"]["web"]
        assert len(web) == 3
        assert web[0] == {"title": "A", "url": "https://a.example.com", "description": "desc A", "position": 1}
        assert web[2]["position"] == 3


    def test_empty_results(self, monkeypatch):
        _install_fake_ddgs(monkeypatch, text_results=[])
        import plugins.web.ddgs.provider as prov
        _force_inprocess_search(monkeypatch, prov)

        result = prov.DDGSWebSearchProvider().search("nothing", limit=5)
        assert result["success"] is True
        assert result["data"]["web"] == []

    @pytest.mark.live_system_guard_bypass
    def test_hung_search_times_out_and_returns_failure(self, monkeypatch):
        """#36776 / #68096: a hung worker must be bounded by the wall-clock
        timeout and reaped — even when the child never returns to Python."""
        _install_fake_ddgs(monkeypatch)
        import plugins.web.ddgs.provider as prov

        monkeypatch.setattr(prov, "_test_hook", "sleep", raising=True)
        monkeypatch.setattr(prov, "_SEARCH_TIMEOUT_SECS", 0.4, raising=True)
        monkeypatch.setattr(prov, "_TERMINATE_GRACE_SECS", 0.5, raising=True)
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)

        start = time.monotonic()
        result = prov.DDGSWebSearchProvider().search("hangs forever", limit=5)
        elapsed = time.monotonic() - start

        assert result["success"] is False
        assert "timed out" in result["error"].lower()
        assert elapsed < 5.0, f"search did not return promptly ({elapsed:.1f}s)"
        _assert_worker_reaped(prov)

    def test_fast_search_not_affected_by_timeout_wrapper(self, monkeypatch):
        """Happy-path guard: the timeout wrapper must not break a normal,
        fast search — results flow through unchanged."""
        _install_fake_ddgs(
            monkeypatch,
            text_results=[{"title": "T", "href": "https://e.com", "body": "B"}],
        )
        import plugins.web.ddgs.provider as prov
        _force_inprocess_search(monkeypatch, prov)

        result = prov.DDGSWebSearchProvider().search("q", limit=5)
        assert result["success"] is True
        assert result["data"]["web"][0]["url"] == "https://e.com"
        assert result["data"]["web"][0]["title"] == "T"


# ---------------------------------------------------------------------------
# Process isolation (#68096)
# ---------------------------------------------------------------------------


def _assert_worker_reaped(prov) -> None:
    """Assert the last DDGS worker process has exited."""
    proc = prov._last_worker_proc
    assert proc is not None, "expected a DDGS worker process to have been started"
    assert proc.poll() is not None, (
        f"DDGS worker still alive (pid={proc.pid}, returncode={proc.returncode})"
    )


@pytest.mark.live_system_guard_bypass
class TestDDGSProcessIsolation:
    def test_gil_holding_worker_times_out_and_is_reaped(self, monkeypatch):
        """#68096: parent deadline still fires when the child holds its GIL."""
        _install_fake_ddgs(monkeypatch)
        import plugins.web.ddgs.provider as prov

        monkeypatch.setattr(prov, "_test_hook", "gil", raising=True)
        monkeypatch.setattr(prov, "_SEARCH_TIMEOUT_SECS", 0.5, raising=True)
        monkeypatch.setattr(prov, "_TERMINATE_GRACE_SECS", 0.5, raising=True)
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)

        start = time.monotonic()
        result = prov.DDGSWebSearchProvider().search("gil hold", limit=5)
        elapsed = time.monotonic() - start

        assert result["success"] is False
        assert "timed out" in result["error"].lower()
        assert elapsed < 5.0, f"GIL-hold search did not time out promptly ({elapsed:.1f}s)"
        _assert_worker_reaped(prov)

    def test_interrupt_terminates_worker_promptly(self, monkeypatch):
        """TUI/gateway interrupt must kill the DDGS child before the deadline."""
        _install_fake_ddgs(monkeypatch)
        import plugins.web.ddgs.provider as prov

        # Flip interrupt after the first poll so the wait loop observes it.
        calls = {"n": 0}

        def _interrupt_after_poll():
            calls["n"] += 1
            return calls["n"] >= 2

        monkeypatch.setattr(prov, "_test_hook", "sleep", raising=True)
        monkeypatch.setattr(prov, "_SEARCH_TIMEOUT_SECS", 30, raising=True)
        monkeypatch.setattr(prov, "_TERMINATE_GRACE_SECS", 0.5, raising=True)
        monkeypatch.setattr("tools.interrupt.is_interrupted", _interrupt_after_poll)

        start = time.monotonic()
        result = prov.DDGSWebSearchProvider().search("interrupt me", limit=5)
        elapsed = time.monotonic() - start

        assert result["success"] is False
        assert "interrupted" in result["error"].lower()
        assert elapsed < 5.0, f"interrupt did not return promptly ({elapsed:.1f}s)"
        _assert_worker_reaped(prov)


    def test_no_orphan_after_successful_search(self, monkeypatch):
        _install_fake_ddgs(monkeypatch)
        import plugins.web.ddgs.provider as prov

        monkeypatch.setattr(prov, "_test_hook", "empty", raising=True)
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)

        result = prov.DDGSWebSearchProvider().search("q", limit=5)
        assert result["success"] is True
        _assert_worker_reaped(prov)


# ---------------------------------------------------------------------------
# Missing-package self-heal (Docker: the durable package store can be wiped
# by an ABI-bumping image rebuild, so a configured ddgs can vanish)
# ---------------------------------------------------------------------------


def _uninstall_ddgs(monkeypatch):
    """Make ``import ddgs`` fail until the returned state is unblocked."""
    state = {"blocked": True}
    monkeypatch.delitem(sys.modules, "ddgs", raising=False)
    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if state["blocked"] and (name == "ddgs" or name.startswith("ddgs.")):
            raise ImportError("No module named 'ddgs'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    return state


class TestDDGSMissingPackageSelfHeal:
    def test_search_installs_the_pinned_package_when_missing(self, monkeypatch):
        """A missing ddgs must be lazy-installed, not handed back to the user
        as "go run pip" — that's the docker-restart papercut."""
        import plugins.web.ddgs.provider as prov

        monkeypatch.setattr(prov, "_install_attempted", False, raising=True)
        state = _uninstall_ddgs(monkeypatch)

        installed: list[str] = []

        def fake_ensure(feature, *, prompt=True):
            installed.append(feature)
            # Simulate the install making the package importable again.
            state["blocked"] = False
            _install_fake_ddgs(monkeypatch)

        monkeypatch.setattr("tools.lazy_deps.ensure", fake_ensure)

        assert prov._ensure_ddgs_installed() is True
        assert installed == ["search.ddgs"], (
            "search must self-heal through the lazy-dep allowlist entry"
        )

    def test_search_reports_failure_without_raising(self, monkeypatch):
        import plugins.web.ddgs.provider as prov
        from tools.lazy_deps import FeatureUnavailable

        monkeypatch.setattr(prov, "_install_attempted", False, raising=True)
        state = _uninstall_ddgs(monkeypatch)

        def fake_ensure(feature, *, prompt=True):
            raise FeatureUnavailable(feature, ("ddgs",), "offline")

        monkeypatch.setattr("tools.lazy_deps.ensure", fake_ensure)

        result = prov.DDGSWebSearchProvider().search("q", limit=3)
        assert state["blocked"] is True
        assert result["success"] is False
        assert "not installed" in result["error"]

    def test_install_attempted_once_per_process(self, monkeypatch):
        """A permanently-unavailable ddgs must not re-run pip on every query."""
        import plugins.web.ddgs.provider as prov
        from tools.lazy_deps import FeatureUnavailable

        monkeypatch.setattr(prov, "_install_attempted", False, raising=True)
        _uninstall_ddgs(monkeypatch)

        calls = []

        def fake_ensure(feature, *, prompt=True):
            calls.append(feature)
            raise FeatureUnavailable(feature, ("ddgs",), "offline")

        monkeypatch.setattr("tools.lazy_deps.ensure", fake_ensure)

        assert prov._ensure_ddgs_installed() is False
        assert prov._ensure_ddgs_installed() is False
        assert len(calls) == 1

    def test_present_package_is_never_reinstalled(self, monkeypatch):
        """ddgs at any version counts as installed — no version churn."""
        import plugins.web.ddgs.provider as prov

        _install_fake_ddgs(monkeypatch)
        monkeypatch.setattr(prov, "_install_attempted", False, raising=True)

        def boom(feature, *, prompt=True):
            raise AssertionError(f"unexpected install of {feature}")

        monkeypatch.setattr("tools.lazy_deps.ensure", boom)
        assert prov._ensure_ddgs_installed() is True


class TestDDGSWorkerSeesDurableTarget:
    """The search worker is a bare child process, so it must activate the
    durable lazy-install store itself. Without that, a ddgs installed on the
    Docker data volume imports fine in the parent and fails in the child —
    every search returns "No module named 'ddgs'".
    """

    def test_worker_imports_ddgs_from_durable_target(self, tmp_path, monkeypatch):
        target = tmp_path / "lazy-packages"
        target.mkdir()
        (target / "ddgs.py").write_text(
            "class DDGS:\n"
            "    def __init__(self, **kwargs):\n"
            "        pass\n"
            "    def __enter__(self):\n"
            "        return self\n"
            "    def __exit__(self, *a):\n"
            "        return False\n"
            "    def text(self, query, max_results=5):\n"
            "        yield {'title': 'durable', 'href': 'https://d.example.com',\n"
            "               'body': 'from the data volume'}\n",
            encoding="utf-8",
        )

        monkeypatch.setenv("HERMES_LAZY_INSTALL_TARGET", str(target))
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)

        import plugins.web.ddgs.provider as prov

        # The parent resolves ddgs the same way (hermes_bootstrap does this at
        # startup); the point of the test is the CHILD, which is spawned fresh.
        saved_path = list(sys.path)
        try:
            from tools.lazy_deps import activate_durable_lazy_target

            activate_durable_lazy_target()
            result = prov.DDGSWebSearchProvider().search("q", limit=3)
        finally:
            sys.path[:] = saved_path
            sys.modules.pop("ddgs", None)

        assert result["success"] is True, result.get("error")
        assert result["data"]["web"][0]["url"] == "https://d.example.com"
        _assert_worker_reaped(prov)


# ---------------------------------------------------------------------------
# Integration: _is_backend_available / _get_backend / check_web_api_key
# ---------------------------------------------------------------------------


class TestDDGSBackendWiring:
    def test_is_backend_available_true_when_package_importable(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: True)
        assert web_tools._is_backend_available("ddgs") is True


    def test_auto_detect_picks_ddgs_as_last_resort(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})
        for key in ("FIRECRAWL_API_KEY", "FIRECRAWL_API_URL", "PARALLEL_API_KEY",
                    "TAVILY_API_KEY", "EXA_API_KEY", "SEARXNG_URL", "BRAVE_SEARCH_API_KEY"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
        monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: True)
        assert web_tools._get_backend() == "ddgs"

    def test_check_web_api_key_true_when_ddgs_configured(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "ddgs"})
        monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: True)
        assert web_tools.check_web_api_key() is True


# ---------------------------------------------------------------------------
# ddgs is search-only: web_extract returns a clear error
# ---------------------------------------------------------------------------


class TestDDGSSearchOnlyErrors:
    _register_providers = staticmethod(register_all_web_providers)

    @pytest.fixture(autouse=True)
    def _populate_web_registry(self):
        self._register_providers()
        yield
        from agent.web_search_registry import _reset_for_tests
        _reset_for_tests()

    def test_web_extract_returns_search_only_error(self, monkeypatch):
        import asyncio
        from tools import web_tools

        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "ddgs"})
        monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: True)
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
        async def _allow_ssrf(_url: str) -> bool:
            return True

        monkeypatch.setattr(web_tools, "async_is_safe_url", _allow_ssrf)
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False, raising=False)

        result_str = asyncio.get_event_loop().run_until_complete(
            web_tools.web_extract_tool(["https://example.com"])
        )
        result = json.loads(result_str)
        assert result["success"] is False
        assert "search-only" in result["error"].lower()
        assert "duckduckgo" in result["error"].lower() or "ddgs" in result["error"].lower()
