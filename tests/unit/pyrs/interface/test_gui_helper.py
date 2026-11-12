"""Tests for the save helpers in pyrs/interface/gui_helper.py"""

from pathlib import Path
from typing import Callable, List, Optional

import pytest

from pyrs.core.workspaces import HidraWorkspace
from pyrs.interface import gui_helper
from pyrs.interface.gui_helper import (
    dotted,
    impose_extension,
    nxstress_save_available,
    resolve_save_target,
    write_nxstress_or_report,
)
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


@pytest.fixture
def replace_answer(monkeypatch: pytest.MonkeyPatch) -> List[bool]:
    """Answers the replace-existing-file question without a dialog: set
    `replace_answer[0]`; each question asked appends its file to the list."""
    answer = [True]

    def confirm(parent, filename: str) -> bool:
        answer.append(filename)  # type: ignore[arg-type]
        return answer[0]

    monkeypatch.setattr(gui_helper, "confirm_replace", confirm)
    return answer


@pytest.mark.parametrize(
    "filename, expected",
    [("out", "out.nxs"), ("out.nxs", "out.nxs"), ("out.h5", "out.h5"), ("HB2B_1327.v2", "HB2B_1327.v2")],
    ids=["appended", "already-there", "other-kept", "dotted-name-kept"],
)
def test_impose_extension_appends_only_when_missing(filename: str, expected: str) -> None:
    assert impose_extension(filename, ".nxs") == expected


@pytest.mark.parametrize("extension", ["nxs", ".nxs"], ids=["bare", "dotted"])
def test_dotted_has_one_leading_dot(extension: str) -> None:
    assert dotted(extension) == ".nxs"


def test_impose_extension_bare_extension_gets_dot() -> None:
    """Configured extensions carry no dot (`ConfigSchema` refuses one)."""
    assert impose_extension("out", "nxs") == "out.nxs"


@pytest.mark.parametrize(
    "typed, expected",
    [("out.nxs", "out.nxs"), ("out.hdf5", "out.hdf5"), ("out.h5", None)],
    ids=["as-typed", "also", "other"],
)
def test_resolve_save_target_bare_extensions(typed: str, expected: Optional[str], messages: List[str]) -> None:
    """The configured form, without a dot, decides the same as the dotted one."""
    target = resolve_save_target(None, typed, extension="nxs", format_name="NXstress", also_accepted=("hdf5",))
    assert target == expected


def test_resolve_save_target_right_extension_as_typed(
    tmp_path: Path, messages: List[str], replace_answer: List[bool]
) -> None:
    """The dialog already asked about this exact name, so nothing more is asked."""
    # Arrange
    typed = tmp_path / "out.nxs"
    typed.write_text("old")

    # Act
    target = resolve_save_target(None, str(typed), extension=".nxs", format_name="NXstress")

    # Assert
    assert target == str(typed)
    assert messages == [] and replace_answer == [True]


def test_resolve_save_target_also_accepted_as_typed(messages: List[str]) -> None:
    target = resolve_save_target(
        None, "out.hdf5", extension=".h5", format_name="HidraProjectFile", also_accepted=(".hdf5",)
    )
    assert target == "out.hdf5"
    assert messages == []


@pytest.mark.parametrize("typed", ["out.h5", "HB2B_1327.v2"], ids=["other-format", "dotted-name"])
def test_resolve_save_target_other_extension_refused(typed: str, messages: List[str]) -> None:
    """Another extension is refused and reported, never silently replaced."""
    # Act
    target = resolve_save_target(None, typed, extension=".nxs", format_name="NXstress")

    # Assert
    assert target is None
    assert messages == [f"Unable to save NXstress format as '*{Path(typed).suffix}'"]


def test_resolve_save_target_no_extension_new_file(tmp_path: Path, replace_answer: List[bool]) -> None:
    target = resolve_save_target(None, str(tmp_path / "out"), extension=".nxs", format_name="NXstress")
    assert target == str(tmp_path / "out.nxs")
    assert replace_answer == [True]  # nothing to replace, so nothing asked


@pytest.mark.parametrize("replace", [True, False], ids=["replace", "decline"])
def test_resolve_save_target_no_extension_existing_file_asks(
    tmp_path: Path, replace_answer: List[bool], replace: bool
) -> None:
    """`out` becomes `out.nxs`, a different file from the one the dialog asked
    about, so replacing an existing `out.nxs` is asked here."""
    # Arrange
    (tmp_path / "out.nxs").write_text("old")
    replace_answer[0] = replace

    # Act
    target = resolve_save_target(None, str(tmp_path / "out"), extension=".nxs", format_name="NXstress")

    # Assert
    assert replace_answer[1:] == [str(tmp_path / "out.nxs")]
    assert target == (str(tmp_path / "out.nxs") if replace else None)


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


def test_write_nxstress_appends_extension(tmp_path: Path, messages: List[str]) -> None:
    # Arrange
    def write(path: str) -> None:
        Path(path).write_text("new")

    # Act
    written = write_nxstress_or_report(None, str(tmp_path / "out"), write)

    # Assert
    assert written
    assert (tmp_path / "out.nxs").read_text() == "new"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.nxs"]
    assert messages == []


def test_write_nxstress_other_extension_writes_nothing(tmp_path: Path, messages: List[str]) -> None:
    # Arrange
    calls: List[str] = []

    # Act
    written = write_nxstress_or_report(None, str(tmp_path / "out.h5"), calls.append)

    # Assert
    assert not written
    assert calls == []
    assert list(tmp_path.iterdir()) == []
    assert messages == ["Unable to save NXstress format as '*.h5'"]


def test_write_nxstress_existing_partial_file_refused(tmp_path: Path, messages: List[str]) -> None:
    """A `.partial.nxs` already in the way -- e.g. left by an interrupted save --
    is neither overwritten nor removed, and the destination is untouched."""
    # Arrange
    target = tmp_path / "out.nxs"
    target.write_text("keep")
    partial = tmp_path / "out.partial.nxs"
    partial.write_text("leftover")
    calls: List[str] = []

    # Act
    written = write_nxstress_or_report(None, str(target), calls.append)

    # Assert
    assert not written
    assert calls == []
    assert target.read_text() == "keep"
    assert partial.read_text() == "leftover"
    assert messages == ["Not saved: 'out.partial.nxs' already exists"]


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
    assert messages == ["Failed to save 'out.nxs'"]


def test_nxstress_path_and_filter_follow_the_configured_extension(config_override):
    config_override({"nxstress": {"extension": "nx5"}})

    assert gui_helper.is_nxstress_path("run.nx5")
    assert not gui_helper.is_nxstress_path("run.nxs")
    assert gui_helper.nxstress_file_filter() == "NXstress (*.nx5)"
