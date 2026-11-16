"""PyRS refuses to start with a malformed configuration override.

`pyrs.utilities.config` validates the configuration when it is first imported,
so an override file named by the `env` environment variable is checked before
any of PyRS runs. These start a fresh interpreter, because the check runs once
per process.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def _start(override: Path, home: Path) -> subprocess.CompletedProcess:
    """Import `pyrs.utilities.config` in a fresh interpreter with `env=override`."""
    return subprocess.run(
        [sys.executable, "-c", "import pyrs.utilities.config"],
        env={**os.environ, "env": str(override), "HOME": str(home)},
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_an_unknown_key_stops_start_up(tmp_path: Path) -> None:
    # Arrange
    override = tmp_path / "override.yml"
    override.write_text("nxstress:\n  discriminator_field: [direction]\n")

    # Act
    result = _start(override, tmp_path)

    # Assert
    assert result.returncode != 0
    assert "Invalid PyRS configuration" in result.stderr
    assert "nxstress.discriminator_field: not a PyRS configuration key" in result.stderr


def test_a_valid_override_starts(tmp_path: Path) -> None:
    """The control: the subprocess itself is not what fails."""
    # Arrange
    override = tmp_path / "override.yml"
    override.write_text("nxstress:\n  discriminator_fields: [direction]\n")

    # Act
    result = _start(override, tmp_path)

    # Assert
    assert result.returncode == 0, result.stderr
