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
- `config_override` — override config values **without** swapping the singleton, so
  that production modules which bound `Config` the documented way still see the
  change. Use this for any test whose subject reads config; use `default_config`
  only when the test is *about* the config machinery itself (reload, `loadEnv`,
  isolation), which is what `test_config.py` does. See `config_override`'s own
  docstring for why the two cannot be the same fixture.
"""

from collections.abc import Callable, Generator
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

    # Captured BEFORE the reload: this is the instance that every module which did
    # `from pyrs.utilities.config import Config` at import time is bound to. The
    # reload below replaces the module attribute but cannot reach those bindings,
    # so unless it is put back at teardown this fixture permanently orphans every
    # consumer for the rest of the session -- and `config_override` then has no
    # way to reach them. See `config_override`'s docstring.
    original_ns_config = sys.modules["neutrons_standard.config"].Config
    original_pyrs_config = pyrs.utilities.config.Config

    importlib.reload(sys.modules["neutrons_standard.config"])
    importlib.reload(pyrs.utilities.config)

    yield pyrs.utilities.config.Config

    # Restore the pre-test singleton rather than reloading to yet another new one:
    # a third instance would be as orphaned as the second. `reset_Singletons()`
    # still clears the decorator's bookkeeping so the next `default_config` can
    # build its own fresh instance.
    reset_Singletons()
    sys.modules["neutrons_standard.config"].Config = original_ns_config
    pyrs.utilities.config.Config = original_pyrs_config


@pytest.fixture
def config_override() -> Generator[Callable[[dict], None]]:
    """Override config values in place, without replacing the `Config` singleton.

    `default_config` resets and reloads the singleton, which constructs a **new**
    `_Config` instance. Any module that bound the name the documented way --
    `from pyrs.utilities.config import Config`, which is what `config.py`'s
    docstring instructs and what `_instrument.py`, `_discriminator.py` and the
    three viewers all do -- still holds the *previous* instance, and so cannot
    see anything such a test overrides. Measured: with `default_config` active,
    `_Instrument._instrument_names()` returns the shipped default no matter what
    the test configured.

    This fixture never swaps the instance. It deep-merges into the live
    singleton's `_config` -- the same dict `refresh` merges into -- and restores
    a deep copy afterwards. Lookups are live (`__getitem__` reads `_config` on
    every access, with no cache), so every binding sees the change, bound-name
    and module-attribute alike.

    Reaching into `_config` is private access, and is deliberate: the ability to
    override at runtime exists for tests, so the accommodation belongs in a test
    fixture rather than in the shape of production code.

    No `HOME` redirection is needed, because nothing is reloaded and so nothing
    is written: `tests/conftest.py` already performed the one unavoidable
    first-ever import under a throwaway `HOME`.

    Each merge is followed by `validate_config()`, so a test cannot install a
    configuration production would have rejected. That is **not** a fix for the
    underlying gap -- `neutrons_standard.Config` re-validates nothing on
    `refresh`/`loadEnv`, and `_Config.validate` is a documented no-op, so a
    runtime override in production would still go unchecked (recorded as DD-1 in
    `plans/PR_review/deferred-defects.md`). It closes the narrower thing a test
    fixture can close: a malformed override here would otherwise be discovered
    as a confusing failure somewhere downstream, or not at all.

    `validate=False` opts out, for the small number of tests whose *subject* is
    a configuration `ConfigSchema` rejects. Those tests exercise checks inside
    consumers -- `_discriminator.field_names()`, `_Instrument._instrument_names()`
    -- that schema validation has made **unreachable in production**: measured,
    a blank discriminator name, a repeated one and a blank instrument name are
    each refused at import, and the `env` OS-variable override is merged before
    `validate_config()` runs, so nothing can deliver such a config to a
    consumer. Whether that defence-in-depth should stay is a layering question
    recorded in DD-1, not one to settle in a test fixture; until it is settled,
    the tests need the state production forbids, and say so at each call.

    Yields:
        A callable taking a nested mapping to deep-merge into the live config,
        e.g. `override({"nxstress": {"discriminator_fields": ["direction"]}})`,
        and an optional `validate=False`. It may be called more than once; each
        call merges onto the last.

    Raises:
        RuntimeError: From `validate_config`, if a merge leaves the
            configuration malformed -- naming the offending key.
    """
    # Safe at fixture time: `pyrs.utilities.config` is long since imported (see
    # `tests/conftest.py`), so `init("pyrs")` has already run and touching
    # `neutrons_standard.config` here cannot race it.
    import copy

    import pyrs.utilities.config as config_module
    from neutrons_standard.config import merge_dicts

    config = config_module.Config
    saved = copy.deepcopy(config._config)

    def override(mapping: dict, *, validate: bool = True) -> None:
        merge_dicts(config._config, mapping)
        if validate:
            config_module.validate_config()

    yield override

    # Restore by clearing and refilling rather than rebinding, so that anything
    # holding a reference to this dict keeps seeing the restored state.
    config._config.clear()
    config._config.update(saved)
