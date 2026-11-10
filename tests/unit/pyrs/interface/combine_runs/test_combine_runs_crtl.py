"""Combine runs turns typed run numbers into the runs' project files (#931)

The run lookup is replaced, so no data archive is needed. Combining by run number used to pass
the reduction *folders* on as if they were project files.
"""

import pytest

from pyrs.interface.combine_runs import combine_runs_crtl
from pyrs.interface.combine_runs.combine_runs_crtl import CombineRunsCrtl


def test_parse_entry_list_gives_project_files(monkeypatch):
    """Each run number becomes <its folder>/HB2B_<run>.h5."""
    # Arrange
    monkeypatch.setattr(combine_runs_crtl, "get_input_project_file", lambda run: f"/ipts/shared/r{run}")
    controller = CombineRunsCrtl(_model=None)

    # Act
    files = controller.parse_entry_list("1327, 1328")

    # Assert
    assert files == ["/ipts/shared/r1327/HB2B_1327.h5", "/ipts/shared/r1328/HB2B_1328.h5"]


def test_parse_entry_list_missing_run_raises(monkeypatch):
    """A run without a project file raises the lookup's error for the window to show."""

    # Arrange
    def missing(run):
        raise RuntimeError(f"No project file HB2B_{run}.h5 for run {run}")

    monkeypatch.setattr(combine_runs_crtl, "get_input_project_file", missing)
    controller = CombineRunsCrtl(_model=None)

    # Act / Assert
    with pytest.raises(RuntimeError, match="HB2B_1327.h5"):
        controller.parse_entry_list("1327")
