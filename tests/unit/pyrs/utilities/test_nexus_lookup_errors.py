"""Failures of the NeXus-file lookup surface as RuntimeError, which the windows show (#931)

``finddata`` exiting with an error used to escape as ``CalledProcessError``, and an empty
``FileFinder`` result as ``IndexError``; neither is caught by the windows' error pop-ups. Both are
exercised through the public ``get_nexus_file``: the archive search fails (or finds nothing), and so
does the ``finddata`` fallback.
"""

from subprocess import CalledProcessError

import pytest

from pyrs.utilities import file_util


def _failing_finddata(*args, **kwargs):
    raise CalledProcessError(returncode=1, cmd=["finddata", "hb2b", "1234"])


def _failing_archive_search(*args):
    raise RuntimeError("ArchiveSearch failed")


@pytest.mark.parametrize(
    "find_runs",
    [_failing_archive_search, lambda *args: []],
    ids=["archive-search-fails", "archive-search-finds-nothing"],
)
def test_run_not_found_raises(monkeypatch, find_runs):
    """With the archive search and finddata both failing, the caller gets a RuntimeError naming the run."""
    # Arrange
    monkeypatch.setattr(file_util.FileFinder, "findRuns", find_runs)
    monkeypatch.setattr(file_util, "check_output", _failing_finddata)

    # Act / Assert
    with pytest.raises(RuntimeError, match="HB2B_1234"):
        file_util.get_nexus_file(1234)


def test_missing_finddata_tool_raises(monkeypatch):
    """Without the finddata tool, FileNotFoundError reaches the caller (the windows catch it too)."""

    # Arrange
    def no_finddata(*args, **kwargs):
        raise FileNotFoundError("finddata")

    monkeypatch.setattr(file_util.FileFinder, "findRuns", lambda *args: [])
    monkeypatch.setattr(file_util, "check_output", no_finddata)

    # Act / Assert
    with pytest.raises(FileNotFoundError):
        file_util.get_nexus_file(1234)
