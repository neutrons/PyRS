"""Combine runs turns typed run numbers into the runs' project files (#931)

The real run lookup runs over a synthetic ``<tmp>/shared/manualreduce`` tree; only the IPTS lookup
is replaced. Combining by run number used to pass the reduction *folders* on as if they were
project files, silently dropped some runs, overflowed above run 32767 and left out a range's end.
"""

import os
from unittest.mock import MagicMock

import pytest

from pyrs.interface.combine_runs.combine_runs_crtl import CombineRunsCrtl
from pyrs.utilities import file_util


@pytest.fixture
def manualreduce(tmp_path, monkeypatch):
    """A synthetic IPTS whose shared/manualreduce holds the runs added with ``_add_runs``."""
    monkeypatch.setattr(file_util, "get_ipts_dir", lambda run_number: tmp_path)
    folder = tmp_path / "shared" / "manualreduce"
    folder.mkdir(parents=True)
    return folder


def _add_runs(folder, runs):
    for run in runs:
        (folder / f"HB2B_{run}.h5").write_bytes(b"")


@pytest.mark.parametrize(
    "entry, runs",
    [
        ("1327, 1328", [1327, 1328]),
        ("1327:1330", [1327, 1328, 1329, 1330]),
        ("1330;1327", [1327, 1328, 1329, 1330]),
        ("40000, 40001", [40000, 40001]),
    ],
    ids=["list", "range-includes-end", "reversed-range", "above-int16"],
)
def test_parse_entry_list_gives_project_files(manualreduce, entry, runs):
    """Each run becomes its existing project file; a range includes both ends."""
    # Arrange
    _add_runs(manualreduce, runs)

    # Act
    files = CombineRunsCrtl(_model=None).parse_entry_list(entry)

    # Assert
    assert files == [str(manualreduce / f"HB2B_{run}.h5") for run in runs]
    assert all(os.path.isfile(name) for name in files)


def test_parse_entry_list_missing_run_raises(manualreduce):
    """A run without a project file raises the lookup's error; it is never silently dropped."""
    # Arrange
    _add_runs(manualreduce, [1327])

    # Act / Assert
    with pytest.raises(RuntimeError, match="HB2B_1328.h5"):
        CombineRunsCrtl(_model=None).parse_entry_list("1327, 1328")


def test_parse_entry_list_ipts_not_found_raises(monkeypatch):
    """An IPTS lookup failure (e.g. no ``finddata`` tool) propagates instead of dropping the run."""

    # Arrange
    def no_finddata(run_number):
        raise FileNotFoundError("finddata")

    monkeypatch.setattr(file_util, "get_ipts_dir", no_finddata)

    # Act / Assert
    with pytest.raises(FileNotFoundError):
        CombineRunsCrtl(_model=None).parse_entry_list("1327")


def test_export_ignores_cancelled_dialog():
    """A cancelled save dialog ("") does not try to write a file."""
    # Arrange
    model = MagicMock()

    # Act
    CombineRunsCrtl(model).export_combined_projectfile("")

    # Assert
    model.export_project_files.assert_not_called()


def test_range_with_three_ends_raises(manualreduce):
    """A range needs exactly two run numbers; more is a clear error, not a guess."""
    # Act / Assert
    with pytest.raises(ValueError, match="needs exactly two run numbers"):
        CombineRunsCrtl(_model=None).parse_entry_list("1327:1330:1332")
