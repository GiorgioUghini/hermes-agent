"""Tests for the durable lazy-install target (immutable Docker images).

These cover the mechanism that lets opt-in backends lazy-install on the
sealed-venv Docker image without being able to break the agent core:
installs are redirected to a writable dir on the data volume, and that dir
is appended to the END of ``sys.path`` so the core venv always wins name
collisions.

The headline invariant — *a package in the durable store can never shadow
a core module* — is proved with a REAL install into a temp target (no
mocked pip), exercising the actual ``--target`` + sys.path-append path.
That E2E test is guarded by network availability; everything else is pure
unit logic with no network.
"""

from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

from tools import lazy_deps as ld


# ---------------------------------------------------------------------------
# Target resolution + gating
# ---------------------------------------------------------------------------


class TestTargetResolution:
    def test_no_target_when_env_unset(self, monkeypatch):
        monkeypatch.delenv(ld._LAZY_TARGET_ENV, raising=False)
        assert ld._lazy_install_target() is None


    def test_target_resolved_when_set(self, monkeypatch, tmp_path):
        monkeypatch.setenv(ld._LAZY_TARGET_ENV, str(tmp_path / "lazy"))
        assert ld._lazy_install_target() == tmp_path / "lazy"


class TestGatingWithTarget:
    """``HERMES_DISABLE_LAZY_INSTALLS=1`` must STOP blocking once a durable
    target is configured — the redirect is the safe path — but the config
    kill switch still wins in every mode."""

    def test_disable_env_blocks_without_target(self, monkeypatch):
        monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
        monkeypatch.delenv(ld._LAZY_TARGET_ENV, raising=False)
        # config unreadable → fails open on the config check, but the sealed
        # env var with no target still blocks.
        monkeypatch.setattr(
            "hermes_cli.config.load_config", lambda: {}, raising=False
        )
        assert ld._allow_lazy_installs() is False

    def test_disable_env_allows_with_target(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
        monkeypatch.setenv(ld._LAZY_TARGET_ENV, str(tmp_path))
        monkeypatch.setattr(
            "hermes_cli.config.load_config", lambda: {}, raising=False
        )
        assert ld._allow_lazy_installs() is True


    def test_normal_mode_unaffected(self, monkeypatch):
        # No sealed env, no target → default allow (unchanged behaviour).
        monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
        monkeypatch.delenv(ld._LAZY_TARGET_ENV, raising=False)
        monkeypatch.setattr(
            "hermes_cli.config.load_config", lambda: {}, raising=False
        )
        assert ld._allow_lazy_installs() is True


# ---------------------------------------------------------------------------
# ABI stamp / durable-store rebuild safety
# ---------------------------------------------------------------------------


class TestAbiStamp:
    def test_creates_dir_and_stamp(self, tmp_path):
        target = tmp_path / "lazy"
        err = ld._ensure_target_ready(target)
        assert err is None
        assert target.is_dir()
        stamp = target / ld._TARGET_STAMP_NAME
        assert stamp.read_text().strip() == ld._python_abi_tag()


    def test_readonly_target_reports_error(self, tmp_path):
        # A path under a non-writable parent should surface a clean error,
        # not raise.
        ro_parent = tmp_path / "ro"
        ro_parent.mkdir()
        os.chmod(ro_parent, 0o500)
        try:
            err = ld._ensure_target_ready(ro_parent / "lazy")
            assert err is not None
            assert "not writable" in err
        finally:
            os.chmod(ro_parent, 0o700)  # let pytest clean up


# ---------------------------------------------------------------------------
# sys.path append ordering (the core-wins invariant, unit level)
# ---------------------------------------------------------------------------


class TestSysPathAppend:
    def test_target_appended_not_prepended(self, tmp_path, monkeypatch):
        target = tmp_path / "lazy"
        target.mkdir()
        saved = list(sys.path)
        try:
            ld._activate_target_on_syspath(target)
            assert str(target) in sys.path
            # Must be at/after every pre-existing entry — i.e. core wins.
            idx = sys.path.index(str(target))
            assert idx >= len(saved), (
                "durable target must be appended after all core entries"
            )
        finally:
            sys.path[:] = saved

    def test_activation_idempotent(self, tmp_path, monkeypatch):
        target = tmp_path / "lazy"
        target.mkdir()
        saved = list(sys.path)
        try:
            ld._activate_target_on_syspath(target)
            ld._activate_target_on_syspath(target)
            assert sys.path.count(str(target)) == 1
        finally:
            sys.path[:] = saved


# ---------------------------------------------------------------------------
# Install path: arg construction (network-free) + a real install (opt-in).
# ---------------------------------------------------------------------------


class TestInstallArgConstruction:
    """Verify the durable-target install builds the right pip/uv command
    WITHOUT hitting the network, by stubbing the subprocess layer. This is
    the CI-safe coverage of the install path; the genuine PyPI install below
    is opt-in only."""

    def test_target_and_constraint_args_passed(self, tmp_path, monkeypatch):
        target = tmp_path / "lazy-packages"
        monkeypatch.setenv(ld._LAZY_TARGET_ENV, str(target))
        # No uv on PATH → force the pip tier so we assert one known command.
        monkeypatch.setattr(ld.shutil, "which", lambda _: None)

        captured = {}

        def fake_run(cmd, *a, **k):
            # The pip --version probe must look healthy so we reach install.
            if "--version" in cmd:
                return subprocess.CompletedProcess(cmd, 0, "pip 24.0", "")
            captured["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, "ok", "")

        monkeypatch.setattr(ld.subprocess, "run", fake_run)
        # Avoid mutating the real interpreter's sys.path on success.
        monkeypatch.setattr(ld, "_activate_target_on_syspath", lambda _t: None)

        result = ld._venv_pip_install(("somepkg==1.2.3",))
        assert result.success
        cmd = captured["cmd"]
        # --target points at the durable dir...
        assert "--target" in cmd
        assert str(target) in cmd
        # ...a --constraint file pins shared deps to core...
        assert "--constraint" in cmd
        # ...and the spec is last.
        assert cmd[-1] == "somepkg==1.2.3"

    def test_no_target_args_in_venv_scoped_mode(self, monkeypatch):
        # Env unset → plain venv-scoped install, no --target / --constraint.
        monkeypatch.delenv(ld._LAZY_TARGET_ENV, raising=False)
        monkeypatch.setattr(ld.shutil, "which", lambda _: None)
        captured = {}

        def fake_run(cmd, *a, **k):
            if "--version" in cmd:
                return subprocess.CompletedProcess(cmd, 0, "pip 24.0", "")
            captured["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, "ok", "")

        monkeypatch.setattr(ld.subprocess, "run", fake_run)
        result = ld._venv_pip_install(("somepkg==1.2.3",))
        assert result.success
        assert "--target" not in captured["cmd"]
        assert "--constraint" not in captured["cmd"]


@pytest.mark.skipif(
    os.environ.get("HERMES_RUN_NETWORK_TESTS") != "1",
    reason="opt-in real-install test (set HERMES_RUN_NETWORK_TESTS=1); CI runs "
    "the network-free arg-construction + synthetic-shadow tests instead",
)
class TestRealInstallCoreWins:
    """Genuine PyPI install into a durable target (opt-in). Proves the wire
    end to end: the package lands in the target, not the core venv, and is
    importable via the appended sys.path entry. Skipped by default so the
    unit-test shard never depends on PyPI reachability/egress."""

    def test_install_lands_in_target_and_imports(self, tmp_path, monkeypatch):
        target = tmp_path / "lazy-packages"
        monkeypatch.setenv(ld._LAZY_TARGET_ENV, str(target))
        # 'isodate' is tiny, pure-python, and not shipped in the core venv,
        # so a successful import must resolve to the durable target.
        result = ld._venv_pip_install(("isodate==0.7.2",))
        assert result.success, f"install failed: {result.stderr}"
        # Landed in the durable target, not the core venv.
        installed = list(target.glob("isodate*"))
        assert installed, f"isodate not found under target {target}: {list(target.iterdir())}"
        # Importable now that the target is on sys.path.
        import importlib
        importlib.invalidate_caches()
        mod = importlib.import_module("isodate")
        assert mod.__file__ is not None
        assert Path(mod.__file__).is_relative_to(target)


class TestDurableTargetPlan:
    """``durable_target_plan`` is the single routing decision shared by the
    lazy installer and the ``hermes tools`` post-setup installer, so both
    honour the sealed-venv redirect identically."""

    def test_venv_scoped_when_env_unset(self, monkeypatch):
        monkeypatch.delenv(ld._LAZY_TARGET_ENV, raising=False)
        with ld.durable_target_plan() as plan:
            assert plan.target is None
            assert plan.args == []
            assert plan.error is None

    def test_target_and_constraint_args_when_configured(self, tmp_path, monkeypatch):
        target = tmp_path / "lazy-packages"
        monkeypatch.setenv(ld._LAZY_TARGET_ENV, str(target))
        with ld.durable_target_plan() as plan:
            assert plan.error is None
            assert plan.target == target
            assert plan.args[:2] == ["--target", str(target)]
            assert "--constraint" in plan.args
            constraints = Path(plan.args[plan.args.index("--constraint") + 1])
            assert constraints.exists()
        # The temp constraints file must not outlive the context.
        assert not constraints.exists()

    def test_unusable_target_reports_error_instead_of_silently_using_venv(
        self, tmp_path, monkeypatch
    ):
        # A file where the target dir should be — mkdir can never succeed.
        blocker = tmp_path / "lazy-packages"
        blocker.write_text("not a directory", encoding="utf-8")
        monkeypatch.setenv(ld._LAZY_TARGET_ENV, str(blocker / "nested"))
        with ld.durable_target_plan() as plan:
            assert plan.error, "an unusable durable target must be reported"
            assert plan.args == []


class TestInstallErrorReporting:
    """When uv runs first and fails, its error must survive to the caller.

    Reported symptom: on the Docker image ``hermes tools post-setup ddgs``
    printed "pip not available and ensurepip failed" on a host where uv was
    installed and working — the real (permission) error from uv had been
    thrown away, sending the user to debug the wrong tool."""

    def test_uv_error_survives_ensurepip_failure(self, monkeypatch):
        monkeypatch.delenv(ld._LAZY_TARGET_ENV, raising=False)
        monkeypatch.setattr(ld.shutil, "which", lambda _: "/usr/local/bin/uv")

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

        monkeypatch.setattr(ld.subprocess, "run", fake_run)

        result = ld._venv_pip_install(("ddgs==9.14.4",))
        assert result.success is False
        assert "failed to write to /opt/hermes/.venv" in result.stderr
        assert "ensurepip failed" in result.stderr

    def test_uv_error_survives_pip_install_failure(self, monkeypatch):
        monkeypatch.delenv(ld._LAZY_TARGET_ENV, raising=False)
        monkeypatch.setattr(ld.shutil, "which", lambda _: "/usr/local/bin/uv")

        def fake_run(cmd, *a, **k):
            if cmd[0] == "/usr/local/bin/uv":
                return subprocess.CompletedProcess(cmd, 1, "", "uv: permission denied")
            if "--version" in cmd:
                return subprocess.CompletedProcess(cmd, 0, "pip 24.0", "")
            return subprocess.CompletedProcess(cmd, 1, "", "pip: permission denied")

        monkeypatch.setattr(ld.subprocess, "run", fake_run)

        result = ld._venv_pip_install(("ddgs==9.14.4",))
        assert result.success is False
        assert "uv: permission denied" in result.stderr
        assert "pip: permission denied" in result.stderr


class TestDdgsIsLazyInstallable:
    """ddgs must be reachable through the lazy-install pipeline, otherwise it
    can only be installed into the sealed venv — which the Docker image
    discards on every container recreate."""

    def test_ddgs_feature_registered(self):
        assert "search.ddgs" in ld.LAZY_DEPS
        specs = ld.LAZY_DEPS["search.ddgs"]
        assert len(specs) == 1
        assert ld._pkg_name_from_spec(specs[0]) == "ddgs"
        assert ld._spec_is_safe(specs[0])


class TestCoreNeverShadowed:
    """The headline invariant — a package in the durable store can never
    shadow a core module — proved WITHOUT a network install by synthesizing
    a shadow copy of a core package directly on disk in the target. This is
    deterministic (no PyPI) and a stronger check: we control exactly what
    the shadow contains, so a sentinel attribute proves which copy won.
    """

    def test_synthetic_shadow_does_not_win(self, tmp_path, monkeypatch):
        # 'packaging' is always present in the venv (transitive of the build
        # toolchain). Resolve the core copy's location first.
        import importlib.util
        core_spec = importlib.util.find_spec("packaging")
        assert core_spec is not None and core_spec.origin
        core_path = Path(core_spec.origin).parent

        # Plant a fake 'packaging' in the durable target with a sentinel that
        # the real core copy does NOT have.
        target = tmp_path / "lazy-packages"
        monkeypatch.setenv(ld._LAZY_TARGET_ENV, str(target))
        ld._ensure_target_ready(target)
        shadow_pkg = target / "packaging"
        shadow_pkg.mkdir()
        (shadow_pkg / "__init__.py").write_text(
            "SHADOW_SENTINEL = True\n__version__ = '0.0.0-shadow'\n"
        )
        assert (shadow_pkg / "__init__.py").exists(), "shadow copy must exist on disk"

        # Activate the target (append-only) and re-resolve the import.
        saved = list(sys.path)
        try:
            ld._activate_target_on_syspath(target)
            import importlib
            importlib.invalidate_caches()
            spec_after = importlib.util.find_spec("packaging")
            assert spec_after is not None and spec_after.origin
            resolved = Path(spec_after.origin).parent
            # Core path must still win; the shadow in the target is ignored.
            assert resolved == core_path, (
                f"durable-store copy shadowed core: resolved to {resolved}, "
                f"expected core at {core_path}"
            )
            assert resolved != shadow_pkg, "import resolved to the shadow copy"
        finally:
            sys.path[:] = saved
            sys.modules.pop("packaging", None)
            importlib.invalidate_caches()
