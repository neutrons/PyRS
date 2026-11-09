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
        (["/data/v1.2/HB2B_1423.nxs.h5", "/data/v1.2/HB2B_1424.nxs.h5"], "HB2B_1423_1424"),
    ],
    ids=["one-run", "two-runs", "plain-string", "nxs-h5-suffix"],
)
def test_project_name_ignores_dotted_directory(model, project_files, expected):
    """Dots in a directory name do not end up in the project name."""
    # Act
    model._set_up_project_name(project_files)

    # Assert
    assert model.project_name == expected


@pytest.mark.parametrize(
    "directory, expected",
    [
        ("/HFIR/HB2B/IPTS-1234/shared/", "/HFIR/HB2B/IPTS-1234/shared/HB2B_1423HB2B_1424.h5"),
        ("/home/u/v1.2/", "/home/u/v1.2/HB2B_1423HB2B_1424.h5"),
        # no directory at all: the name must not gain a leading "/"
        ("", "HB2B_1423HB2B_1424.h5"),
    ],
    ids=["hfir-path", "dotted-directory", "bare-file-names"],
)
def test_parse_working_files_keeps_full_directory(model, directory, expected):
    """The combined file of several runs sits in the runs' own directory."""
    # Arrange
    project_files = [f"{directory}HB2B_1423.h5", f"{directory}HB2B_1424.h5"]

    # Act
    working_file = model._parse_working_files(project_files)

    # Assert
    assert working_file == expected


def test_parse_working_files_single_run_unchanged(model):
    """A single run is used as is."""
    # Act
    working_file = model._parse_working_files(["/HFIR/HB2B/IPTS-1234/shared/HB2B_1423.h5"])

    # Assert
    assert working_file == "/HFIR/HB2B/IPTS-1234/shared/HB2B_1423.h5"


@pytest.mark.parametrize(
    "project_files, expected",
    [
        (["/d/HB2B_1423_v2.h5"], "HB2B_1423_v2"),
        (["/d/HB2B_1423_v2.h5", "/d/HB2B_1424_v2.h5"], "HB2B_1423_1424"),
        (["/d/HB2B_1423_1424.h5"], "HB2B_1423_1424"),
        (["/d/Test_a.h5", "/d/Test_b.h5"], "HB2B_Test_a_Test_b"),
    ],
    ids=["one-suffixed-run", "two-suffixed-runs", "one-combined-file", "no-run-number"],
)
def test_project_name_of_suffixed_files(model, project_files, expected):
    """One file keeps its whole stem; several use the number after ``HB2B_`` (#1080)."""
    # Act
    model._set_up_project_name(project_files)

    # Assert
    assert model.project_name == expected


@pytest.mark.parametrize(
    "project_files, expected",
    [
        (["/d/HB2B_1423_v2.h5", "/d/HB2B_1424_v2.h5"], "/d/HB2B_1423HB2B_1424.h5"),
        (["/d/Test_a.h5", "/d/Test_b.h5"], "/d/Test_aTest_b.h5"),
        # a bare number is a stem, not an HB2B_ run number: no prefix is invented
        (["/d/1423.h5", "/d/1424.h5"], "/d/14231424.h5"),
    ],
    ids=["suffixed-runs", "no-run-number", "numeric-stems"],
)
def test_parse_working_files_suffixed_runs(model, project_files, expected):
    """Several runs are named by their ``HB2B_`` numbers, else their whole stems (#1080)."""
    # Act
    working_file = model._parse_working_files(project_files)

    # Assert
    assert working_file == expected
