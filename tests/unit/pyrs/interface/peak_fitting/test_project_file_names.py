"""Run numbers and file names are taken from the file name, never the directory.

A dot or ``HB2B`` in a directory name (``~/.local``, ``v1.2``, ``/HFIR/HB2B/IPTS-...``)
used to truncate the path, so the project name and the CSV export path came out wrong.
"""

from unittest.mock import MagicMock

import pytest

from pyrs.interface.peak_fitting.peak_fitting_model import PeakFittingModel


@pytest.fixture
def model():
    return PeakFittingModel(peak_fit_core=MagicMock())


@pytest.mark.parametrize(
    "project_files, expected",
    [
        (["/data/v1.2/HB2B_1423.h5"], "HB2B_1423"),
        (["/home/u/.local/HB2B_1423.h5", "/home/u/.local/HB2B_1424.h5"], "HB2B_1423_1424"),
        ("/data/v1.2/HB2B_1423.h5", "HB2B_1423"),
    ],
    ids=["one-run", "two-runs", "plain-string"],
)
def test_project_name_ignores_dotted_directory(model, project_files, expected):
    """Dots in a directory name do not end up in the project name."""
    # Act
    model._set_up_project_name(project_files)

    # Assert
    assert model.project_name == expected


@pytest.mark.parametrize("directory", ["/HFIR/HB2B/IPTS-1234/shared", "/home/u/v1.2"])
def test_parse_working_files_keeps_full_directory(model, directory):
    """The combined file of several runs sits in the runs' own directory."""
    # Arrange
    project_files = [f"{directory}/HB2B_1423.h5", f"{directory}/HB2B_1424.h5"]

    # Act
    working_file = model._parse_working_files(project_files)

    # Assert
    assert working_file == f"{directory}/HB2B_1423HB2B_1424.h5"


def test_parse_working_files_single_run_unchanged(model):
    """A single run is used as is."""
    # Act
    working_file = model._parse_working_files(["/HFIR/HB2B/IPTS-1234/shared/HB2B_1423.h5"])

    # Assert
    assert working_file == "/HFIR/HB2B/IPTS-1234/shared/HB2B_1423.h5"
