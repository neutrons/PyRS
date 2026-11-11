#!/usr/bin/python
# Test utilities methods without GUI
import os
from pathlib import Path
from pyrs.interface import gui_helper
from pyrs.utilities import get_default_output_dir, get_input_project_file, get_ipts_dir, get_nexus_file  # type: ignore
import pytest


MISSING_RUNNUMBER = 112123260


@pytest.mark.parametrize(
    "text, expected",
    [
        ("3, 4, 5", [3, 4, 5]),
        ("1:4", [1, 2, 3, 4]),
        ("3:5, 4:10, 19", [3, 4, 5, 6, 7, 8, 9, 10, 19]),
        ("2-3", [2, 3]),
    ],
    ids=["list", "range-includes-end", "overlapping-ranges", "dash-range"],
)
def test_parse_integers(text, expected):
    """Ranges include both ends (#931), as in combine runs and manual reduction; repeats are merged."""
    # Act
    integers = gui_helper.parse_integers(text)

    # Assert
    assert integers == expected


def test_parse_integers_not_integer_raises():
    """A non-integer entry is refused."""
    # Act / Assert
    with pytest.raises(RuntimeError):
        gui_helper.parse_integers("3.2, 4")


@pytest.mark.skipif(not os.path.exists("/HFIR/HB2B/shared/"), reason="HFIR data archive is not mounted")
@pytest.mark.integration
def test_get_ipts_dir():
    """Test to get IPTS directory from run number

    Returns
    -------

    """
    IPTS_22731 = Path("/HFIR/HB2B/IPTS-22731/")
    # test run number
    assert get_ipts_dir(1060) == IPTS_22731, "IPTS directory is not correct for run 1060"

    # test absolute path
    assert get_ipts_dir("/HFIR/HB2B/IPTS-22731/nexus/HB2B_1060.nxs.h5") == IPTS_22731, (
        "IPTS directory is not correct for run 1060"
    )

    # Test no such run
    with pytest.raises(RuntimeError):
        get_ipts_dir(MISSING_RUNNUMBER)

    # Test exception
    with pytest.raises(TypeError):
        get_ipts_dir(1.2)
    with pytest.raises(ValueError):
        get_ipts_dir("1.2")
    with pytest.raises(ValueError):
        get_ipts_dir("abc")


@pytest.mark.skipif(not os.path.exists("/HFIR/HB2B/shared/"), reason="HFIR data archive is not mounted")
@pytest.mark.integration
def test_get_default_output_dir():
    assert get_default_output_dir(1060) == "/HFIR/HB2B/IPTS-22731/shared/manualreduce", (
        "Output directory is not correct for run 1060"
    )

    # Test no such run
    with pytest.raises(RuntimeError):
        get_default_output_dir(MISSING_RUNNUMBER)


@pytest.mark.skipif(not os.path.exists("/HFIR/HB2B/shared/"), reason="HFIR data archive is not mounted")
@pytest.mark.integration
def test_get_input_project_file():
    # the folder returned holds the run's project file (#931); which reduction folder that is
    # depends on the archive, so only its IPTS is checked
    folder = get_input_project_file(1060)
    assert folder.startswith("/HFIR/HB2B/IPTS-22731/shared/")
    assert os.path.isfile(os.path.join(folder, "HB2B_1060.h5"))

    # Test no such run
    with pytest.raises(RuntimeError):
        get_input_project_file(MISSING_RUNNUMBER)
    # an unknown preferred type is tested without the archive in test_input_project_file.py


@pytest.mark.skipif(not os.path.exists("/HFIR/HB2B/shared/"), reason="HFIR data archive is not mounted")
@pytest.mark.integration
def test_get_nexus_file():
    assert get_nexus_file(1060) == "/HFIR/HB2B/IPTS-22731/nexus/HB2B_1060.nxs.h5"
    assert get_nexus_file(1017) == "/HFIR/HB2B/IPTS-22731/nexus/HB2B_1017.nxs.h5"

    # Test no such run
    with pytest.raises(RuntimeError):
        get_nexus_file(MISSING_RUNNUMBER)


if __name__ == "__main__":
    pytest.main()
