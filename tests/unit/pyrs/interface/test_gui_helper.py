"""Tests for the NXstress save helpers in pyrs/interface/gui_helper.py"""

from pathlib import Path
from typing import Callable, List, Optional

import pytest

from pyrs.core.workspaces import HidraWorkspace
from pyrs.interface import gui_helper
from pyrs.interface.gui_helper import nxstress_save_available, write_nxstress_or_report
from pyrs.utilities.config import Config
from tests.util.instrument_helpers import ensure_instrument_geometry


def _workspace(with_geometry: bool) -> HidraWorkspace:
    ws = HidraWorkspace()
    if with_geometry:
        ensure_instrument_geometry(ws)
    return ws


@pytest.fixture
def messages(monkeypatch: pytest.MonkeyPatch) -> List[str]:
    """Records message-box titles instead of opening a dialog."""
    recorded: List[str] = []
    monkeypatch.setattr(gui_helper, "pop_message", lambda parent, message, *args, **kwargs: recorded.append(message))
    return recorded


@pytest.mark.parametrize(
    "make_ws, needs_config",
    [
        (lambda: None, False),
        (lambda: _workspace(with_geometry=False), False),
        (lambda: _workspace(with_geometry=True), True),
    ],
    ids=["no-workspace", "no-geometry", "geometry"],
)
def test_nxstress_save_available(make_ws: Callable[[], Optional[HidraWorkspace]], needs_config: bool) -> None:
    """Available only with instrument geometry, and then only if NXstress is enabled."""
    # Arrange
    ws = make_ws()

    # Act
    available = nxstress_save_available(ws)

    # Assert
    assert available == (needs_config and bool(Config["nxstress.enable"]))


def test_write_nxstress_imposes_extension(tmp_path: Path, messages: List[str]) -> None:
    # Arrange
    def write(path: str) -> None:
        Path(path).write_text("new")

    # Act
    written = write_nxstress_or_report(None, str(tmp_path / "out.h5"), write)

    # Assert
    assert written
    assert (tmp_path / "out.nxs").read_text() == "new"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.nxs"]
    assert messages == []


def test_write_nxstress_failure_keeps_existing_target(tmp_path: Path, messages: List[str]) -> None:
    """A failed write reports the error, removes its partial file, and leaves an
    existing file at the destination untouched."""
    # Arrange
    target = tmp_path / "out.nxs"
    target.write_text("keep")

    def write(path: str) -> None:
        Path(path).write_text("trunc")
        raise RuntimeError("Unable to parse peak tag 'peak0'")

    # Act
    written = write_nxstress_or_report(None, str(target), write)

    # Assert
    assert not written
    assert target.read_text() == "keep"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.nxs"]
    assert messages == ["Failed to save NXstress file"]
