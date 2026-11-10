"""`get_input_project_file` finds the folder that actually holds a run's project file (#931)

The IPTS lookup is replaced by a synthetic ``<tmp>/shared/{manualreduce,autoreduce}`` tree, so no
data archive is needed. It used to return a folder as soon as it existed, even without the run's
``HB2B_<run>.h5``; the load then failed later, ending in a ``NoneType`` error in the windows.
"""

import pytest

from pyrs.utilities import file_util
from pyrs.utilities.file_util import get_input_project_file

RUN = 1234


@pytest.fixture
def shared(tmp_path, monkeypatch):
    """An IPTS folder with empty manualreduce and autoreduce folders."""
    monkeypatch.setattr(file_util, "get_ipts_dir", lambda run_number: tmp_path)
    for name in ("manualreduce", "autoreduce"):
        (tmp_path / "shared" / name).mkdir(parents=True)
    return tmp_path / "shared"


def _add_run(folder):
    (folder / f"HB2B_{RUN}.h5").write_bytes(b"")


@pytest.mark.parametrize(
    "preferred, holders, expected",
    [
        ("manual", ["manualreduce", "autoreduce"], "manualreduce"),
        ("manual", ["autoreduce"], "autoreduce"),
        ("auto", ["manualreduce", "autoreduce"], "autoreduce"),
    ],
    ids=["manual-has-run", "only-auto-has-run", "auto-preferred"],
)
def test_returns_folder_holding_the_run(shared, preferred, holders, expected):
    """The folder returned is one that holds HB2B_<run>.h5, the preferred one first."""
    # Arrange
    for holder in holders:
        _add_run(shared / holder)

    # Act
    folder = get_input_project_file(RUN, preferredType=preferred)

    # Assert
    assert folder == str(shared / expected)


@pytest.mark.parametrize(
    "preferred, holders",
    [("manual", []), ("auto", []), ("auto", ["manualreduce"])],
    ids=["manual-no-file", "auto-no-file", "auto-only-manual-has-it"],
)
def test_missing_run_file_raises(shared, preferred, holders):
    """Folders without the run's file are an error that names the file, not a folder to load from."""
    # Arrange
    for holder in holders:
        _add_run(shared / holder)

    # Act / Assert
    with pytest.raises(RuntimeError, match=f"HB2B_{RUN}.h5"):
        get_input_project_file(RUN, preferredType=preferred)
