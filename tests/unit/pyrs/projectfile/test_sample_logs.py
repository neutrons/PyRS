"""
String sample logs are `bytes` on disk and `str` in memory.

The two halves of one invariant. `SampleLogs.__setitem__` normalizes string log
values to `str` so that in memory a log never depends on where the workspace came
from; the encode back to `bytes` therefore belongs at the HDF5 boundary, which is
also the only place it *can* live -- h5py has no conversion path for numpy's `<U`
dtype, so a project file simply cannot be written without it.

The in-memory half is pinned by
`tests/unit/pyrs/dataobjects/test_sample_logs.py::TestStringLogsAreTextInMemory`.
"""

import h5py
import numpy as np
import pytest

from pyrs.dataobjects.constants import HidraConstants
from pyrs.projectfile.file_object import HidraProjectFile, HidraProjectFileMode


class TestStringLogsAreBytesOnDisk:
    @pytest.fixture
    def project(self, tmp_path):
        return HidraProjectFile(str(tmp_path / "logs.h5"), HidraProjectFileMode.OVERWRITE)

    @staticmethod
    def _stored(project, name):
        return project._project_h5[HidraConstants.RAW_DATA][HidraConstants.SAMPLE_LOGS][name]

    def test_a_str_log_is_written_as_variable_length_bytes(self, project):
        """h5py cannot store `<U` at all, so without the encode this raises.

        Variable-length, not fixed-width `|S`: a `|S` column's width is fixed by
        the longest value present when it is created, and a longer value written
        later is truncated **silently**. Encoding with `numpy.char.encode` would
        produce exactly that, and would quietly convert an existing file's
        variable-length columns on a re-save.
        """
        # Act
        project.append_experiment_log("direction", np.array(["11", "22", "33"]))

        # Assert
        stored = self._stored(project, "direction")
        assert h5py.check_string_dtype(stored.dtype) is not None
        assert stored[()].dtype.kind == "O"

    def test_a_bytes_log_is_written_unchanged(self, project):
        # Act
        project.append_experiment_log("direction", np.array([b"11", b"22", b"33"]))

        # Assert -- already bytes; the encode has nothing to do
        assert self._stored(project, "direction").dtype.kind == "S"

    def test_a_longer_value_written_later_is_not_truncated(self, project):
        """The property the variable-length dtype buys, stated directly."""
        # Arrange
        project.write_sub_runs(np.array([1]))
        project.append_experiment_log("short", np.array(["ab"]))

        # Act
        project.append_experiment_log("long", np.array(["a_considerably_longer_value"]))

        # Assert
        assert project.read_sample_logs()["long"][0] == "a_considerably_longer_value"

    def test_a_str_log_reads_back_as_str(self, project):
        """The round trip the pairing exists to make total.

        Note what h5py hands back here: an **object** array of Python `bytes`,
        not a `|S` array. A normalization checked against `|S` alone passes a
        hand-built fixture and leaves this untouched -- which is how the first
        version of this change shipped green while doing nothing for any log that
        came off a real file.
        """
        # Arrange
        project.write_sub_runs(np.array([1, 2, 3]))
        project.append_experiment_log("direction", np.array(["11", "22", "33"]))
        assert self._stored(project, "direction")[()].dtype.kind == "O"

        # Act
        recovered = project.read_sample_logs()

        # Assert -- `str` out, not the `bytes` h5py handed back
        assert recovered["direction"].dtype.kind == "U"
        assert list(recovered["direction"]) == ["11", "22", "33"]

    def test_numeric_logs_keep_their_dtype(self, project):
        """The encode must not reach dtypes it has no business converting."""
        # Act
        project.append_experiment_log("vx", np.array([0.0, 0.1, 0.2]))

        # Assert
        assert self._stored(project, "vx").dtype.kind == "f"
