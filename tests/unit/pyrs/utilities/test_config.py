"""
Tests for `pyrs/utilities/config.py`.

Every test here requests the `default_config` fixture (see
`tests/unit/pyrs/utilities/conftest.py`) and imports `pyrs.utilities.config` only
*inside* the test body -- never at module level. A module-level import would run
`pyrs.utilities.config`'s side effects (writing a backup file under `~/.pyrs/`)
against the real home directory at test-collection time, before the fixture has had
a chance to redirect `HOME` to a `tmp_path`.
"""


def test_default_config_loads_shipped_defaults(default_config):
    """Test that the shipped `pyrs/resources/application.yml` values load unmodified."""
    # Arrange / Act
    config = default_config

    # Assert
    assert config["nxstress.enable"] is True
    assert config["nxstress.extension"] == ".nxs"
    assert config["nxstress.use_production_names"] is False
    assert config["legacy_io.enable"] is True
    assert config["legacy_io.extension"] == ".h5"


def test_default_config_env_override_merges_on_top_of_default(default_config, tmp_path):
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


def test_validate_config_passes_with_shipped_defaults(default_config):
    """Test that `validate_config()` raises nothing when both formats are enabled."""
    # Arrange
    import pyrs.utilities.config as config_module

    # Act / Assert
    config_module.validate_config()  # no exception


def test_validate_config_raises_when_both_formats_disabled(default_config, tmp_path):
    """Test that `validate_config()` rejects a config with no output format enabled."""
    # Arrange
    import pytest

    import pyrs.utilities.config as config_module

    override_file = tmp_path / "override.yml"
    override_file.write_text("nxstress:\n  enable: false\nlegacy_io:\n  enable: false\n")
    default_config.loadEnv(str(override_file))

    # Act / Assert
    with pytest.raises(ValueError, match="At least one of nxstress.enable or legacy_io.enable must be true"):
        config_module.validate_config()
