"""
NXstress / legacy-IO output configuration, backed by `neutrons_standard.Config`.

Import `Config` from *this* module -- never `from neutrons_standard...import Config`
directly anywhere else in this codebase. `neutrons_standard.init("pyrs")` must run
before `neutrons_standard.config` is first imported (that module captures its
`package_name` once, at that moment, from `neutrons_standard.Spec.client_package_name`);
this module is the only place that ordering is guaranteed. A stray direct import
elsewhere would race `init()` and silently pin `package_name` to `None` for the rest of
the process (confirmed: this raises `ModuleNotFoundError: No module named 'None'` in
practice, from the `_Config` singleton's constructor).

The default configuration ships at `pyrs/resources/application.yml` -- this exact
filename and location (a genuine `pyrs.resources` subpackage) is a hard requirement of
`neutrons_standard`, not a PyRS convention. Override a value by setting the `env` OS
environment variable to a `.yml` file name or path; its contents are deep-merged on top
of the shipped default. See `neutrons_standard.config._Config.refresh` for the merge
mechanics.

`neutrons_standard.Config` provides no schema validation of its own (`_Config.validate`
is a no-op) -- `validate_config` below is PyRS's own rule.
"""

import neutrons_standard

neutrons_standard.init("pyrs")
from neutrons_standard.config import Config  # noqa: E402  (import must follow init())


def validate_config() -> None:
    """Raise ValueError unless at least one output format is enabled.

    Raises:
        ValueError: If both `nxstress.enable` and `legacy_io.enable` are false --
            PyRS must be able to write at least one output format.
    """
    if not (Config["nxstress.enable"] or Config["legacy_io.enable"]):
        raise ValueError("At least one of nxstress.enable or legacy_io.enable must be true")


# Fail fast at import time, rather than at the first NXstress-I/O callsite that
# happens to need a valid config.
validate_config()
