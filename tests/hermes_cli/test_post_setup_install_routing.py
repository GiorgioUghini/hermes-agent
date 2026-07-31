"""Post-setup installs must survive an immutable-image container restart.

Reported symptom: on the published Docker image every ``docker compose``
restart lost the ``ddgs`` package, and re-running ``hermes tools post-setup
ddgs`` inside the container failed with::

    pip not available and ensurepip failed: ... ensurepip ... exit status 1
    Run manually: uv pip install -U ddgs

…even though ``uv pip install -U ddgs`` at the same shell worked. Two
mechanisms were wrong:

1. Post-setup installs targeted the *agent venv*, which the image seals
   (``HERMES_DISABLE_LAZY_INSTALLS=1``, root-owned ``/opt/hermes``) and which
   is thrown away on every container recreate. The writable, durable store on
   the data volume (``HERMES_LAZY_INSTALL_TARGET``) was ignored.
2. uv ran first, failed (no write permission), and its error was discarded —
   the user was shown the *ensurepip* failure from the next tier instead.

These tests pin both behaviours plus the ddgs hook's routing through the
lazy-dep pipeline (which is what keeps the version pin in one place).
"""

from __future__ import annotations

import subprocess

import pytest

import hermes_cli.tools_config as tc


@pytest.fixture
def fake_installer(monkeypatch):
    """Capture install commands instead of running them."""
    calls: list[list[str]] = []

    def fake_run(cmd, *a, **k):
        calls.append(list(cmd))
        if "--version" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "pip 24.0", "")
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    monkeypatch.setattr(tc.subprocess, "run", fake_run)
    return calls


class TestPipInstallRouting:
    def test_durable_target_used_when_configured(self, tmp_path, monkeypatch, fake_installer):
        target = tmp_path / "lazy-packages"
        monkeypatch.setenv("HERMES_LAZY_INSTALL_TARGET", str(target))
        monkeypatch.setattr(tc.shutil, "which", lambda _: None)  # force the pip tier

        result = tc._pip_install(["-U", "ddgs", "--quiet"])

        assert result.returncode == 0
        install_cmd = fake_installer[-1]
        assert "--target" in install_cmd
        assert str(target) in install_cmd
        assert install_cmd[-2:] == ["ddgs", "--quiet"]

    def test_venv_scoped_when_no_durable_target(self, monkeypatch, fake_installer):
        monkeypatch.delenv("HERMES_LAZY_INSTALL_TARGET", raising=False)
        monkeypatch.setattr(tc.shutil, "which", lambda _: None)

        result = tc._pip_install(["-U", "ddgs", "--quiet"])

        assert result.returncode == 0
        assert "--target" not in fake_installer[-1]

    def test_unusable_durable_target_reported(self, tmp_path, monkeypatch, fake_installer):
        blocker = tmp_path / "lazy-packages"
        blocker.write_text("not a directory", encoding="utf-8")
        monkeypatch.setenv("HERMES_LAZY_INSTALL_TARGET", str(blocker / "nested"))

        result = tc._pip_install(["-U", "ddgs"])

        assert result.returncode == 1
        assert result.stderr
        assert not fake_installer, "must not fall back to the sealed venv"

    def test_uv_failure_reported_when_ensurepip_also_fails(self, monkeypatch):
        """The user must see uv's real error, not a misleading pip message."""
        monkeypatch.delenv("HERMES_LAZY_INSTALL_TARGET", raising=False)
        monkeypatch.setattr(tc.shutil, "which", lambda _: "/usr/local/bin/uv")

        def fake_run(cmd, *a, **k):
            if cmd[0] == "/usr/local/bin/uv":
                return subprocess.CompletedProcess(
                    cmd, 1, "", "error: failed to write to /opt/hermes/.venv"
                )
            if "--version" in cmd:
                raise FileNotFoundError("no pip")
            if "ensurepip" in cmd:
                raise subprocess.CalledProcessError(1, cmd)
            raise AssertionError(f"unexpected command: {cmd}")

        monkeypatch.setattr(tc.subprocess, "run", fake_run)

        result = tc._pip_install(["-U", "ddgs"])

        assert result.returncode == 1
        assert "failed to write to /opt/hermes/.venv" in result.stderr
        assert "ensurepip failed" in result.stderr


class TestDdgsPostSetup:
    def test_installs_the_pinned_spec_through_lazy_deps(self, monkeypatch, capsys):
        """The hook and the runtime self-heal must agree on one version, or
        they reinstall over each other on every search."""
        import tools.lazy_deps as ld

        monkeypatch.setattr(ld, "is_available", lambda feature: False)
        installed: list[tuple[str, bool]] = []
        monkeypatch.setattr(
            ld, "ensure",
            lambda feature, *, prompt=True: installed.append((feature, prompt)),
        )
        # A raw pip call would bypass the durable-target routing entirely.
        monkeypatch.setattr(
            tc, "_pip_install",
            lambda *a, **k: pytest.fail("post-setup must not shell out to pip directly"),
        )

        tc._run_post_setup("ddgs")

        assert installed == [("search.ddgs", False)]
        out = capsys.readouterr().out
        assert "ddgs installed" in out

    def test_reports_install_failure_with_reason(self, monkeypatch, capsys):
        import tools.lazy_deps as ld

        monkeypatch.setattr(ld, "is_available", lambda feature: False)

        def boom(feature, *, prompt=True):
            raise ld.FeatureUnavailable(feature, ("ddgs==9.14.4",), "network unreachable")

        monkeypatch.setattr(ld, "ensure", boom)

        tc._run_post_setup("ddgs")

        out = capsys.readouterr().out
        assert "install failed" in out
        assert "network unreachable" in out

    def test_already_installed_is_a_noop(self, monkeypatch, capsys):
        import tools.lazy_deps as ld

        monkeypatch.setattr(ld, "is_available", lambda feature: True)
        monkeypatch.setattr(
            ld, "ensure",
            lambda *a, **k: pytest.fail("must not reinstall a satisfied feature"),
        )

        tc._run_post_setup("ddgs")

        assert "already installed" in capsys.readouterr().out
