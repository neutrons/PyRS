"""
Shared fixtures for tests of `pyrs/utilities/`.

Fixture conventions
--------------------
- `default_config` — a test-isolated `neutrons_standard.Config` singleton (see
  `pyrs/utilities/config.py`). Every test that requests it gets a genuinely fresh
  instance (`reset_Singletons()`) loaded against a `HOME` pointed at `tmp_path`, so
  a test can never write into (or accidentally read an override from) the real
  user's `~/.pyrs/` directory, and one test's config changes can never leak into
  the next. Required for any test that touches `pyrs.utilities.config.Config` --
  importing that module unconditionally writes a backup file to `~/.pyrs/` as a
  side effect of loading, real home directory included, if not for this fixture.
  Defined here (rather than under `NXstress/`) since `pyrs/utilities/config.py`
  isn't itself NXstress-specific code; being one directory up, it's visible to
  `NXstress/` tests as well as siblings of this file (e.g. `test_config.py`).
"""

from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    # Only for type-checking -- a real runtime import here would race
    # neutrons_standard.init("pyrs") exactly like importing it anywhere else in this
    # codebase would (see pyrs/utilities/config.py's module docstring). TYPE_CHECKING
    # guards this from ever executing.
    from neutrons_standard.config import _Config


@pytest.fixture
def default_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator["_Config"]:
    """Yield a `neutrons_standard.Config` singleton, fully isolated from the real environment.

    Setup: monkeypatches `HOME` to `tmp_path` and clears any `env` override, then
    resets and reloads the singleton (both `neutrons_standard.config` and
    `pyrs.utilities.config`) so it picks up the redirected `HOME` -- `reset_Singletons()`
    alone only clears the `Singleton` decorator's internal state; it does not change
    what an already-imported module's `Config` name refers to.

    Cleanup: resets and reloads the singleton again after the test, so the next test
    (or any later code importing `pyrs.utilities.config.Config`) gets a clean instance
    rather than one still holding this test's `tmp_path`-scoped `HOME` or `env`
    override.

    Args:
        tmp_path: Pytest's built-in per-test temporary directory; used as the fake
            `HOME` so `neutrons_standard.Config`'s real side effects (writing a backup
            file to `~/.{package_name}/`) never touch the real user's home.
        monkeypatch: Pytest's built-in fixture for reversible env-var patching.

    Yields:
        The live `neutrons_standard.Config` singleton (via
        `pyrs.utilities.config.Config`), loaded against the isolated `HOME`.
    """
    # `neutrons_standard.Config` is a process-wide singleton: every `reload()` writes a
    # backup to `~/.{package_name}/application.yml.bak`, and it may auto-swap onto a
    # pre-existing `~/.{package_name}/{package_name}-user.yml` override -- both against
    # the REAL home directory, unless we redirect `HOME` first. `reset_Singletons()`
    # alone only clears the Singleton decorator's internal `instance`/`initialized`
    # state; it does not change what an *already-imported* module's `Config` name
    # refers to, so the modules that bind it must also be reloaded.
    #
    # Import order matters and is easy to get backwards: `pyrs.utilities.config` must
    # be imported (or already have been) *before* `neutrons_standard.config` is ever
    # directly touched, because its module body calls `neutrons_standard.init("pyrs")`
    # before importing `Config` -- `neutrons_standard.config`'s own module-level
    # `package_name = Spec.client_package_name` line is captured once, at whichever
    # import happens first. Importing `neutrons_standard.config` here ourselves, ahead
    # of `pyrs.utilities.config`, would reproduce that exact bug.
    import importlib
    import sys

    from neutrons_standard.decorators.singleton import reset_Singletons

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("env", raising=False)

    reset_Singletons()
    import pyrs.utilities.config  # first-ever import correctly calls init() before Config

    importlib.reload(sys.modules["neutrons_standard.config"])
    importlib.reload(pyrs.utilities.config)

    yield pyrs.utilities.config.Config

    # Leave a clean, freshly-reset singleton behind for the next test, rather than one
    # holding this test's `tmp_path`-scoped `HOME` and any `env` override it applied.
    reset_Singletons()
    importlib.reload(sys.modules["neutrons_standard.config"])
    importlib.reload(pyrs.utilities.config)
