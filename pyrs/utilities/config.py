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
is a no-op) -- `validate_config` below is PyRS's own rule. It refuses unknown keys as
well as malformed values: a misspelled or misplaced key would otherwise be ignored, and
the setting it was meant to change would silently keep its default.
"""

from typing import Annotated, Any, Iterator

from pydantic import AfterValidator, BaseModel, ConfigDict, ValidationError

import neutrons_standard

neutrons_standard.init("pyrs")
from neutrons_standard.config import Config  # noqa: E402  (import must follow init())


def _non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


def _bare_suffix(value: str) -> str:
    """A filename extension **without** its leading dot.

    The dot belongs to `pathlib`, not to the configuration: `impose_extension`
    adds it. Writing `.nxs` here would make the separator part of the value, so
    a reader could not tell whether it was configured or implied.
    """
    if value.startswith("."):
        raise ValueError(f"must not begin with '.' -- write {value.lstrip('.')!r}, the dot is implied")
    if not value.strip():
        raise ValueError("must not be blank")
    return value


def _distinct(value: list) -> list:
    if len(set(value)) != len(value):
        raise ValueError(f"names the same field more than once: {value}")
    return value


NonBlankText = Annotated[str, AfterValidator(_non_blank)]
BareSuffix = Annotated[str, AfterValidator(_bare_suffix)]
DiscriminatorFields = Annotated[list[NonBlankText], AfterValidator(_distinct)]


class _Section(BaseModel):
    # `strict` is load-bearing, not tidiness. Pydantic's default is permissive
    # enough to accept exactly the malformed values this schema exists to reject:
    # `enable: "false"` validates as `False`, and `use_production_names: "yes"`
    # as `True`. Measured on pydantic 2.13.5.
    #
    # `extra="forbid"` refuses a key the schema does not declare. Without it a
    # misspelled `discriminator_field:`, or a key left over from a removed
    # setting, is dropped without a word and the real key keeps its default.
    model_config = ConfigDict(strict=True, extra="forbid")


class _NXstress(_Section):
    enable: bool
    extension: BareSuffix
    use_production_names: bool
    instrument_name: NonBlankText
    instrument_short_name: NonBlankText
    discriminator_fields: DiscriminatorFields


class _LegacyIO(_Section):
    enable: bool
    extension: BareSuffix


class ConfigSchema(_Section):
    """Schema for `pyrs/resources/application.yml`, one field per shipped key.

    `neutrons_standard.Config` performs no schema validation of its own
    (`_Config.validate` is a no-op), so without this nothing checks the file at
    all -- and a malformed override is not inert. `enable: "false"` is a YAML
    *string* and therefore truthy; a bare `discriminator_fields: direction` is a
    string, and a string is iterable, so it would be read one character at a
    time as five field names.

    Declared as a schema rather than as a list of keys and a pile of `isinstance`
    checks so that `VALIDATED_KEYS` is *derived* from it. The previous version
    kept four hand-maintained tuples whose docstring claimed to cover every key
    while covering four of nine: a key list beside the thing it describes decays,
    and only deriving one from the other stops it.
    """

    nxstress: _NXstress
    legacy_io: _LegacyIO


#: Top-level sections `neutrons_standard` adds to the configuration itself, which
#: `ConfigSchema` does not describe and `validate_config` leaves unchecked:
#: `environment` (the name of the loaded override), `module.root` and
#: `version.default` (added by `_Config.reload`), and `application.version`
#: (written into `~/.pyrs/pyrs-user.yml` by `_Config.swapToUserYml`). Read from
#: neutrons_standard 0.1.0; `test_config.py` fails if a load adds any other.
NEUTRONS_STANDARD_SECTIONS: frozenset[str] = frozenset({"environment", "module", "version", "application"})


def _dotted_keys(model: type[BaseModel], prefix: str = "") -> Iterator[str]:
    """Every leaf key of `model`, dotted, in declaration order."""
    for name, field in model.model_fields.items():
        path = f"{prefix}.{name}" if prefix else name
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            yield from _dotted_keys(annotation, path)
        else:
            yield path


#: Every key `validate_config` checks, derived from `ConfigSchema`.
#: `test_config.py` asserts this matches what `application.yml` ships, in both
#: directions, so a key added to one and not the other fails a test.
VALIDATED_KEYS: tuple[str, ...] = tuple(_dotted_keys(ConfigSchema))


def _pyrs_sections() -> dict[str, Any]:
    """The merged configuration document, without `NEUTRONS_STANDARD_SECTIONS`.

    Read from `Config._config`, which is private to `neutrons_standard`: it offers
    no public way to list the keys a configuration holds, and refusing unknown
    keys needs that list. This module is the one place PyRS touches
    `neutrons_standard` (see the module docstring), so the private access is
    confined here.
    """
    return {name: value for name, value in Config._config.items() if name not in NEUTRONS_STANDARD_SECTIONS}


def _describe(error: dict) -> str:
    if error["type"] == "extra_forbidden":
        return "not a PyRS configuration key; is it misspelled, or in the wrong section?"
    return error["msg"]


def validate_config() -> None:
    """Raise unless the configuration matches `ConfigSchema`.

    Checks the merged document: the shipped `application.yml` with any override
    file on top. Sections in `NEUTRONS_STANDARD_SECTIONS` are not checked.

    Raises:
        RuntimeError: If any key is missing, unknown, or holds the wrong type or
            shape. The message names each offending key and what was wrong with it.
        ValueError: If both `nxstress.enable` and `legacy_io.enable` are false --
            PyRS must be able to write at least one output format. Kept outside
            the schema because it is a rule *between* sections, and because a
            caller catching it should not have to reach into a `ValidationError`
            to find out which rule failed.
    """
    try:
        ConfigSchema.model_validate(_pyrs_sections())
    except ValidationError as error:
        problems = "\n".join(
            "  {}: {}".format(".".join(str(part) for part in item["loc"]), _describe(item)) for item in error.errors()
        )
        raise RuntimeError(f"Invalid PyRS configuration:\n{problems}") from error

    if not (Config["nxstress.enable"] or Config["legacy_io.enable"]):
        raise ValueError("At least one of nxstress.enable or legacy_io.enable must be true")


# Fail fast at import time, rather than at the first NXstress-I/O callsite that
# happens to need a valid config.
validate_config()
