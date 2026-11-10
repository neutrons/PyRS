"""Failures of the NeXus-file lookup surface as RuntimeError, which the windows show (#931)

``finddata`` exiting with an error used to escape as ``CalledProcessError``, and an empty
``FileFinder`` result as ``IndexError``; neither is caught by the windows' error pop-ups.
"""

from subprocess import CalledProcessError

import pytest

from pyrs.utilities import file_util


def _failing_finddata(*args, **kwargs):
    raise CalledProcessError(returncode=1, cmd=["finddata", "hb2b", "1234"])


def test_finddata_error_raises_runtime_error(monkeypatch):
    """finddata exiting with an error becomes a RuntimeError naming the run."""
    # Arrange
    monkeypatch.setattr(file_util, "check_output", _failing_finddata)

    # Act / Assert
    with pytest.raises(RuntimeError, match="HB2B_1234"):
        file_util.__dict__["__run_finddata"](1234)


def test_empty_file_finder_result_raises_runtime_error(monkeypatch):
    """No run found by FileFinder, nor by finddata, is a RuntimeError, not an IndexError."""
    # Arrange
    monkeypatch.setattr(file_util.FileFinder, "findRuns", lambda *args: [])
    monkeypatch.setattr(file_util, "check_output", _failing_finddata)

    # Act / Assert
    with pytest.raises(RuntimeError, match="HB2B_1234"):
        file_util.get_nexus_file(1234)
