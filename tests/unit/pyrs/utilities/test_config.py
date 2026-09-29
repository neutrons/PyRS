"""
Tests for `pyrs/utilities/config.py`.

Every test here requests the `default_config` fixture (see
`tests/unit/pyrs/utilities/conftest.py`) and imports `pyrs.utilities.config` only
*inside* the test body -- never at module level. A module-level import would run
`pyrs.utilities.config`'s side effects (writing a backup file under `~/.pyrs/`)
against the real home directory at test-collection time, before the fixture has had
a chance to redirect `HOME` to a `tmp_path`. That rule is about
`pyrs.utilities.config` specifically; `pytest` and the rest import normally.

**Two ways to override configuration appear below, and choosing the wrong one
fails silently rather than loudly.**

`default_config.loadEnv(<path to a .yml>)` exercises the **documented production
mechanism** -- the `env` OS variable naming a file whose contents are deep-merged
onto the shipped default. Use it when the override path itself is the subject, as
the `validate_config` tests here are. It reloads the singleton, so any module that
bound `Config` the documented way still holds the *previous* instance and cannot
see the override. These tests survive that only because the module under test is
the module being reloaded.

`config_override({...})` deep-merges into the live singleton without replacing it,
so every binding sees the change. Use it whenever some *other* module has to read
the override -- `_instrument.py`, `_discriminator.py`, the viewers. Pointing
`loadEnv` at those reads the stale instance and the test passes against the
shipped default, asserting nothing; that is a real defect this suite once had.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    # Only for type-checking -- see tests/unit/pyrs/utilities/conftest.py's
    # `default_config` fixture for why a real runtime import here is unsafe.
    from neutrons_standard.config import _Config


def test_default_config_loads_shipped_defaults(default_config: "_Config") -> None:
    """Test that the shipped `pyrs/resources/application.yml` values load unmodified."""
    # Arrange / Act
    config = default_config

    # Assert
    assert config["nxstress.enable"] is True
    assert config["nxstress.extension"] == "nxs"
    assert config["nxstress.use_production_names"] is False
    assert config["legacy_io.enable"] is True
    assert config["legacy_io.extension"] == "h5"


def test_default_config_env_override_merges_on_top_of_default(default_config: "_Config", tmp_path: Path) -> None:
    """Test that an `env`-named override file deep-merges onto the shipped default.

    Only `nxstress.enable` is overridden; `legacy_io.*` (untouched by the override
    file) must still come through from the shipped default -- confirming a merge,
    not a wholesale replacement.
    """
    # Arrange
    override_file = tmp_path / "override.yml"
    override_file.write_text("nxstress:\n  enable: false\n")

    # Act
    config = default_config
    config.loadEnv(str(override_file))

    # Assert
    assert config["nxstress.enable"] is False
    assert config["legacy_io.enable"] is True  # unaffected key survives the merge


def test_validate_config_passes_with_shipped_defaults(default_config: "_Config") -> None:
    """Test that `validate_config()` raises nothing when both formats are enabled."""
    # Arrange
    import pyrs.utilities.config as config_module

    # Act / Assert
    config_module.validate_config()  # no exception


def test_validate_config_raises_when_both_formats_disabled(default_config: "_Config", tmp_path: Path) -> None:
    """Test that `validate_config()` rejects a config with no output format enabled."""
    # Arrange
    import pyrs.utilities.config as config_module

    override_file = tmp_path / "override.yml"
    override_file.write_text("nxstress:\n  enable: false\nlegacy_io:\n  enable: false\n")
    default_config.loadEnv(str(override_file))

    # Act / Assert
    with pytest.raises(ValueError, match="At least one of nxstress.enable or legacy_io.enable must be true"):
        config_module.validate_config()


def test_validate_nxstress_enable_not_bool(default_config: "_Config", tmp_path: Path) -> None:
    """Test that a non-bool `nxstress.enable` (e.g. a quoted YAML string) is rejected
    up front, rather than silently passing the truthiness check (`bool("false")` is
    `True` in Python).
    """
    # Arrange
    import pyrs.utilities.config as config_module

    override_file = tmp_path / "override.yml"
    override_file.write_text('nxstress:\n  enable: "false"\n')  # quoted -- stays a str, not a bool
    default_config.loadEnv(str(override_file))

    # Act / Assert
    with pytest.raises(RuntimeError, match=r"nxstress\.enable: Input should be a valid boolean"):
        config_module.validate_config()


def test_validate_legacy_io_enable_not_bool(default_config: "_Config", tmp_path: Path) -> None:
    """Test that a non-bool `legacy_io.enable` (e.g. an int) is rejected up front."""
    # Arrange
    import pyrs.utilities.config as config_module

    override_file = tmp_path / "override.yml"
    override_file.write_text("legacy_io:\n  enable: 1\n")  # int, not a bool
    default_config.loadEnv(str(override_file))

    # Act / Assert
    with pytest.raises(RuntimeError, match=r"legacy_io\.enable: Input should be a valid boolean"):
        config_module.validate_config()


class TestDiscriminatorConfigKeys:
    """`nxstress.discriminator_fields`.

    Typed, and not a type YAML gets right by accident:
    `discriminator_fields: direction` (no list) parses as a string, and a
    string is iterable, so an unchecked value would be read one character at a
    time as five one-letter field names.

    These use `default_config.loadEnv`, not `config_override`, because the
    override *file* is the subject: `validate_config` has to reject what a user
    could actually write in a `.yml`. Nothing outside `pyrs.utilities.config`
    reads the result, which is the condition that makes `loadEnv` safe here --
    see this module's docstring.
    """

    def test_shipped_defaults(self, default_config: "_Config") -> None:
        assert default_config["nxstress.discriminator_fields"] == []

    def test_validate_config_rejects_a_bare_string(self, default_config: "_Config", tmp_path: Path) -> None:
        import pyrs.utilities.config as config_module

        override_file = tmp_path / "override.yml"
        override_file.write_text("nxstress:\n  discriminator_fields: direction\n")
        default_config.loadEnv(str(override_file))

        with pytest.raises(RuntimeError, match=r"discriminator_fields: Input should be a valid list"):
            config_module.validate_config()

    def test_validate_config_rejects_a_blank_entry(self, default_config: "_Config", tmp_path: Path) -> None:
        import pyrs.utilities.config as config_module

        override_file = tmp_path / "override.yml"
        override_file.write_text("nxstress:\n  discriminator_fields: ['direction', '  ']\n")
        default_config.loadEnv(str(override_file))

        with pytest.raises(RuntimeError, match=r"discriminator_fields\.\d+: .*must not be blank"):
            config_module.validate_config()

    def test_validate_config_rejects_a_repeated_field(self, default_config: "_Config", tmp_path: Path) -> None:
        import pyrs.utilities.config as config_module

        override_file = tmp_path / "override.yml"
        override_file.write_text("nxstress:\n  discriminator_fields: ['direction', 'direction']\n")
        default_config.loadEnv(str(override_file))

        with pytest.raises(RuntimeError, match="more than once"):
            config_module.validate_config()

    def test_validate_config_accepts_a_well_formed_field_list(self, default_config: "_Config", tmp_path: Path) -> None:

        import pyrs.utilities.config as config_module

        override_file = tmp_path / "override.yml"
        override_file.write_text("nxstress:\n  discriminator_fields: ['direction']\n")
        default_config.loadEnv(str(override_file))

        config_module.validate_config()  # must not raise


class TestConfigOverrideReachesBoundNames:
    """`config_override` must be visible to modules that bound `Config` the documented way.

    `config.py`'s docstring instructs `from pyrs.utilities.config import Config`,
    which binds the singleton *by value*. `default_config` reloads the module and
    so replaces that singleton, leaving every such module reading the previous
    instance -- which silently made config-dependent behaviour untestable:
    `_Instrument._instrument_names()` returned the shipped default no matter what
    a test configured.

    These pin the fix. They deliberately exercise **production** modules rather
    than reading `Config` directly, because reading it directly is the one access
    pattern that was never broken.
    """

    def test_production_modules_share_the_live_singleton(self):
        """The invariant underneath all of this, asserted directly.

        Requests no config fixture on purpose: it checks the *ambient* session
        state, so it fails if any fixture earlier in the run swapped the
        singleton without putting it back -- which `default_config` used to do,
        orphaning every consumer for the remainder of the session.
        """
        import pyrs.utilities.config as config_module
        from pyrs.utilities.NXstress._discriminator import Config as discriminator_bound
        from pyrs.utilities.NXstress._instrument import Config as instrument_bound

        assert instrument_bound is config_module.Config
        assert discriminator_bound is config_module.Config

    def test_override_reaches_a_module_that_bound_config(self, config_override):
        # Arrange
        from pyrs.utilities.NXstress._instrument import _Instrument

        assert _Instrument._instrument_names() == ("HB2B", "HB2B")

        # Act
        config_override({"nxstress": {"instrument_name": "OVERRIDDEN"}})

        # Assert
        assert _Instrument._instrument_names() == ("OVERRIDDEN", "HB2B")

    def test_override_reaches_the_discriminator_module(self, config_override):
        # Arrange
        from pyrs.utilities.NXstress import _discriminator

        assert _discriminator.field_names() == ()

        # Act
        config_override({"nxstress": {"discriminator_fields": ["direction"]}})

        # Assert
        assert _discriminator.field_names() == ("direction",)

    def test_the_override_is_undone_afterwards(self, config_override):
        """Guards the teardown: a leaked override would make later tests lie."""
        from pyrs.utilities.config import Config

        assert Config["nxstress.instrument_name"] == "HB2B"
        config_override({"nxstress": {"instrument_name": "LEAKED"}})
        assert Config["nxstress.instrument_name"] == "LEAKED"

    def test_sibling_keys_survive_a_partial_override(self, config_override):
        """A shallow assignment would drop every key the override does not mention."""
        # Arrange
        from pyrs.utilities.config import Config

        enabled = Config["nxstress.enable"]

        # Act
        config_override({"nxstress": {"instrument_name": "OVERRIDDEN"}})

        # Assert
        assert Config["nxstress.instrument_name"] == "OVERRIDDEN"
        assert Config["nxstress.enable"] == enabled


class TestEveryShippedKeyIsValidated:
    """`ConfigSchema` must describe every key `application.yml` ships.

    `validate_config`'s docstring once claimed it checked "every typed key"
    while checking four of nine. A key list maintained beside the thing it
    describes decays, because adding a key to the YAML is not a change to the
    list -- so `VALIDATED_KEYS` is now *derived* from `ConfigSchema`, and this
    compares the schema against the shipped file in both directions.

    It also guards a failure mode the schema introduces: a `BaseModel` with no
    fields validates anything at all. `ConfigSchema` was briefly written with
    only a docstring, which made `VALIDATED_KEYS` empty and `validate_config()`
    pass vacuously -- the assertion that the key set is non-empty is what turns
    that into a failure.
    """

    @staticmethod
    def _shipped_keys() -> set:
        """Dotted leaf keys of the shipped default configuration."""
        import yaml

        import pyrs.resources

        text = (Path(pyrs.resources.__file__).parent / "application.yml").read_text()
        document = yaml.safe_load(text)

        def leaves(node, prefix=""):
            for key, value in node.items():
                path = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    yield from leaves(value, path)
                else:
                    yield path

        return set(leaves(document))

    def test_the_schema_describes_some_keys_at_all(self, default_config):
        """A `BaseModel` with no fields accepts anything, silently."""
        # Arrange
        from pyrs.utilities.config import VALIDATED_KEYS

        # Assert
        assert VALIDATED_KEYS, "`ConfigSchema` declares no fields, so `validate_config` checks nothing"

    def test_every_shipped_key_is_checked(self, default_config):
        # Arrange
        from pyrs.utilities.config import VALIDATED_KEYS

        # Act
        unchecked = self._shipped_keys() - set(VALIDATED_KEYS)

        # Assert
        assert not unchecked, (
            f"key(s) in application.yml that `ConfigSchema` does not describe: {sorted(unchecked)}. "
            "Add each to the matching section model in `pyrs/utilities/config.py`."
        )

    def test_no_checked_key_is_absent_from_the_file(self, default_config):
        """The other direction: a checked key that no longer ships would raise on import."""
        # Arrange
        from pyrs.utilities.config import VALIDATED_KEYS

        # Act
        missing = set(VALIDATED_KEYS) - self._shipped_keys()

        # Assert
        assert not missing, f"`ConfigSchema` names key(s) `application.yml` does not ship: {sorted(missing)}"


class TestConfigOverrideRefusesMalformedConfig:
    """The `config_override` fixture validates after merging, and that is pinned.

    Added because an independent mutation pass deleted the fixture's
    `validate_config()` call and the entire 675-test suite stayed green -- the
    guarantee was real but unconstrained, so a later refactor could drop it
    silently and reintroduce the defect recorded as DD-1 in
    `plans/PR_review/deferred-defects.md`.
    """

    def test_a_malformed_override_is_refused_by_the_fixture(self, config_override):
        with pytest.raises(RuntimeError, match="Invalid PyRS configuration"):
            config_override({"nxstress": {"instrument_name": 123}})

    def test_the_opt_out_installs_it_anyway(self, config_override):
        """`validate=False` is what the three deliberately-malformed tests use."""
        from pyrs.utilities.config import Config

        config_override({"nxstress": {"instrument_name": "   "}}, validate=False)

        assert Config["nxstress.instrument_name"] == "   "


class TestValidateConfigRejectsMalformedValues:
    """One case per rule, since each rule exists for a different silent failure."""

    @staticmethod
    def _validate(config_override, override: dict):
        """Install a malformed override and let the refusal surface.

        `config_override` itself calls `validate_config()` after merging, so the
        refusal comes from the fixture. This used to call `validate_config()`
        again on the next line; once the fixture gained the hook that line
        became unreachable, and these tests were silently asserting against the
        fixture rather than against the subject. The dead line is gone and
        `TestConfigOverrideRefusesMalformedConfig` below pins the hook itself,
        so removing it fails something.
        """
        config_override(override)

    @pytest.mark.parametrize("value", ["false", "yes", 1], ids=["false", "yes", "one"])
    def test_a_non_boolean_for_a_boolean_key_raises(self, config_override, value):
        """Pydantic's *default* accepts all three, which is why the schema is strict.

        `enable: "false"` is a YAML string and therefore truthy, and a permissive
        schema would validate it as `False` -- turning the check into a
        conversion. Measured on pydantic 2.13.5.
        """
        with pytest.raises(RuntimeError, match=r"use_production_names: Input should be a valid boolean"):
            self._validate(config_override, {"nxstress": {"use_production_names": value}})

    def test_a_non_string_instrument_name_raises(self, config_override):
        with pytest.raises(RuntimeError, match=r"instrument_name: Input should be a valid string"):
            self._validate(config_override, {"nxstress": {"instrument_name": 123}})

    def test_a_blank_instrument_name_raises(self, config_override):
        with pytest.raises(RuntimeError, match=r"instrument_short_name: .*must not be blank"):
            self._validate(config_override, {"nxstress": {"instrument_short_name": "   "}})

    def test_an_extension_carrying_its_dot_raises(self, config_override):
        """The dot is a `pathlib` separator, supplied by `impose_extension`."""
        with pytest.raises(RuntimeError, match=r"extension: .*must not begin with"):
            self._validate(config_override, {"nxstress": {"extension": ".nxs"}})

    def test_an_empty_extension_raises(self, config_override):
        with pytest.raises(RuntimeError, match=r"extension: .*must not be blank"):
            self._validate(config_override, {"legacy_io": {"extension": ""}})

    def test_a_bare_string_for_the_field_list_raises(self, config_override):
        """A string is iterable, so it would be read one character at a time."""
        with pytest.raises(RuntimeError, match=r"discriminator_fields: Input should be a valid list"):
            self._validate(config_override, {"nxstress": {"discriminator_fields": "direction"}})

    def test_a_repeated_discriminator_field_raises(self, config_override):
        with pytest.raises(RuntimeError, match=r"names the same field more than once"):
            self._validate(config_override, {"nxstress": {"discriminator_fields": ["a", "a"]}})

    def test_every_offending_key_is_named_at_once(self, config_override):
        """A config with two mistakes should report two, not just the first."""
        # Act
        with pytest.raises(RuntimeError) as caught:
            self._validate(
                config_override,
                {"nxstress": {"enable": "false", "instrument_name": 7}},
            )

        # Assert
        message = str(caught.value)
        assert "nxstress.enable" in message
        assert "nxstress.instrument_name" in message

    def test_the_shipped_defaults_validate(self, default_config):
        from pyrs.utilities.config import validate_config

        validate_config()


class TestUnknownKeysAreRefused:
    """A key `ConfigSchema` does not declare is refused, not ignored.

    Ignoring one is a silent failure: a misspelled `discriminator_field:`, a key
    put in the wrong section, or one left over from a removed setting such as
    `nxstress.merge_workspaces` (Decisions Log row 45 in
    `plans/NXstress-prod/README.md`) changes nothing, and the setting it was
    meant to change keeps its default with no sign that anything is wrong.

    These use `default_config.loadEnv`: the override file is the subject.
    """

    @staticmethod
    def _load(default_config: "_Config", tmp_path: Path, text: str) -> None:
        override_file = tmp_path / "override.yml"
        override_file.write_text(text)
        default_config.loadEnv(str(override_file))

    def test_a_misspelled_key_raises(self, default_config: "_Config", tmp_path: Path) -> None:
        # Arrange
        import pyrs.utilities.config as config_module

        self._load(default_config, tmp_path, "nxstress:\n  discriminator_field: [direction]\n")

        # Act / Assert
        with pytest.raises(RuntimeError, match=r"nxstress\.discriminator_field: not a PyRS configuration key"):
            config_module.validate_config()

    def test_a_key_in_the_wrong_section_raises(self, default_config: "_Config", tmp_path: Path) -> None:
        # Arrange
        import pyrs.utilities.config as config_module

        self._load(default_config, tmp_path, "discriminator_fields: [direction]\n")

        # Act / Assert
        with pytest.raises(RuntimeError, match=r"\n  discriminator_fields: not a PyRS configuration key"):
            config_module.validate_config()

    def test_a_removed_key_raises(self, default_config: "_Config", tmp_path: Path) -> None:
        # Arrange
        import pyrs.utilities.config as config_module

        self._load(default_config, tmp_path, "nxstress:\n  merge_workspaces: true\n")

        # Act / Assert
        with pytest.raises(RuntimeError, match=r"nxstress\.merge_workspaces: not a PyRS configuration key"):
            config_module.validate_config()

    def test_every_unknown_key_is_named_at_once(self, default_config: "_Config", tmp_path: Path) -> None:
        # Arrange
        import pyrs.utilities.config as config_module

        self._load(default_config, tmp_path, "nxstress:\n  extention: h5\nlegacy-io:\n  enable: false\n")

        # Act
        with pytest.raises(RuntimeError) as caught:
            config_module.validate_config()

        # Assert
        message = str(caught.value)
        assert "nxstress.extention" in message
        assert "legacy-io" in message

    def test_neutrons_standard_sections_are_not_checked(self, default_config: "_Config", tmp_path: Path) -> None:
        """What a `~/.pyrs/pyrs-user.yml` written by `swapToUserYml` holds, besides PyRS's own keys."""
        # Arrange
        import pyrs.utilities.config as config_module

        self._load(default_config, tmp_path, "environment: pyrs-user\napplication:\n  version: 1.2.3\n")

        # Act / Assert
        config_module.validate_config()  # must not raise

    def test_a_load_adds_no_unlisted_section(self, default_config: "_Config") -> None:
        """Pins `NEUTRONS_STANDARD_SECTIONS` against what neutrons_standard really adds.

        If an upgrade adds a section of its own, PyRS would refuse to start. This
        names the section instead of leaving that to be read off a failed import.
        """
        # Arrange
        from pyrs.utilities.config import NEUTRONS_STANDARD_SECTIONS, ConfigSchema

        # Act
        added = set(default_config._config) - set(ConfigSchema.model_fields)

        # Assert
        assert added, "neutrons_standard added no section of its own: the probe is not looking at the document"
        assert added <= NEUTRONS_STANDARD_SECTIONS, (
            f"neutrons_standard adds section(s) {sorted(added - NEUTRONS_STANDARD_SECTIONS)} that "
            "`NEUTRONS_STANDARD_SECTIONS` in `pyrs/utilities/config.py` does not list."
        )

    def test_the_fixture_refuses_an_unknown_key(self, config_override) -> None:
        """`config_override` reaches the same check, so a test cannot configure a typo."""
        with pytest.raises(RuntimeError, match=r"nxstress\.instrument: not a PyRS configuration key"):
            config_override({"nxstress": {"instrument": "HB2B"}})
