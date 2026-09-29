"""
Tests for pyrs/utilities/NXstress/_input_data.py
"""

from collections.abc import Callable
import numpy as np
from nexusformat.nexus import NXdata, nxopen
from pathlib import Path
import pytest

from pyrs.core.workspaces import HidraWorkspace
from pyrs.utilities.NXstress._input_data import _InputData


def _subset_logs(logs, rows, sub_runs):
    """A `SampleLogs` carrying only the selected scan points.

    `readSubruns` compares the target workspace's sub-runs against the selected
    scan points for exact equality, so the target must already hold the subset.
    """
    from pyrs.dataobjects.sample_logs import SampleLogs
    from pyrs.dataobjects.constants import HidraConstants

    subset = SampleLogs()
    subset[HidraConstants.SUB_RUNS] = np.asarray(sub_runs)[rows]
    return subset


class TestInputData:
    """Test suite for _input_data.py"""

    def test_init_group_appends_to_an_existing_group(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
    ):
        """Passing `data` tail-appends rather than raising (spec 04c)."""
        # Arrange
        first = minimal_HidraWorkspace(with_instrument=True, with_raw_counts=True, sub_runs=np.array([1, 2, 3]))
        second = minimal_HidraWorkspace(with_instrument=True, with_raw_counts=True, sub_runs=np.array([4, 5]))
        data = _InputData.init_group([first])
        before = np.asarray(data["detector_counts"].nxdata).copy()

        # Act
        _InputData.init_group([second], data=data)

        # Assert
        assert data["scan_point"].nxdata.tolist() == [1, 2, 3, 4, 5]
        assert data["detector_counts"].shape[0] == 5
        # The rows already present are untouched -- a tail-append never rewrites them.
        assert np.array_equal(np.asarray(data["detector_counts"].nxdata)[:3], before)

    def test_init_group_append_raises_when_counts_loaded_on_one_side(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
    ):
        """An entry with no raw counts cannot gain them by append, or vice versa."""
        # Arrange: an entry written WITHOUT raw counts, and an input that has them.
        without = minimal_HidraWorkspace(with_instrument=True, with_raw_counts=False)
        data = _InputData.init_group([without])
        with_counts = minimal_HidraWorkspace(with_instrument=True, with_raw_counts=True, sub_runs=np.array([4, 5]))

        # Act / Assert
        with pytest.raises(RuntimeError, match=r".*does not have raw detector counts.*"):
            _InputData.init_group([with_counts], data=data)

    def test_InputData_init_group_data_values(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
    ):
        """Verify detector_counts shape and scan_point values match workspace"""
        ws = minimal_HidraWorkspace(with_instrument=True, with_raw_counts=True)

        data = _InputData.init_group([ws])

        # Verify structure
        assert isinstance(data, NXdata)
        assert "detector_counts" in data
        assert "scan_point" in data

        # Verify data shape
        scan_points = list(ws._raw_counts.keys())
        N_scan = len(scan_points)

        # Get detector size from first scan point
        first_counts = ws.get_detector_counts(scan_points[0])
        N_pixels = len(first_counts)

        assert data["detector_counts"].shape == (N_scan, N_pixels)
        assert len(data["scan_point"]) == N_scan

        # Verify scan_point values match
        np.testing.assert_array_equal(data["scan_point"], scan_points)

    def test_InputData_readSubruns(
        self,
        tmp_path: Path,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
    ):
        """Verify readSubruns round-trip: write then read back"""
        # Build workspace with raw counts
        ws_write = minimal_HidraWorkspace(name="test_workspace_write", with_instrument=True, with_raw_counts=True)

        # Create input data
        data = _InputData.init_group([ws_write])

        # Write to file
        file_path = tmp_path / "test_readSubruns.nxs"
        with nxopen(str(file_path), "w") as nx:
            nx["input_data"] = data

        # Create empty workspace for reading
        ws_read = HidraWorkspace("test_workspace_read")
        # `SampleLogs` must already be attached:
        #   otherwise the workspace will have no `Subruns`!
        ws_read._sample_logs = ws_write._sample_logs

        # Read back
        with nxopen(str(file_path), "r") as nx:
            _InputData.readSubruns(ws_read, nx["input_data"])

        # Verify round-trip
        assert len(ws_read.get_sub_runs()) == len(ws_write.get_sub_runs())

        # Check that all scan points are present
        original_scan_points = list(ws_write._raw_counts.keys())
        read_scan_points = list(ws_read._raw_counts.keys())

        for scan_point in original_scan_points:
            assert scan_point in read_scan_points
            original_counts = ws_write.get_detector_counts(scan_point)
            read_counts = ws_read.get_detector_counts(scan_point)
            np.testing.assert_array_equal(read_counts, original_counts)

    def test_InputData_readSubruns_raises_on_scanpoint_mismatch(
        self,
        tmp_path: Path,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
    ):
        """Verify RuntimeError when workspace has subruns that don't match those from input data"""
        # Build workspace with data
        ws = minimal_HidraWorkspace(with_instrument=True, with_raw_counts=True)

        # Create input data and write to file
        data = _InputData.init_group([ws])
        file_path = tmp_path / "test_existing_subruns.nxs"
        with nxopen(str(file_path), "w") as nx:
            nx["input_data"] = data

        # Try to read into workspace that has subruns that do not match
        existing_subruns = ws._sample_logs._subruns._value
        ws._sample_logs._subruns._value = np.append(
            existing_subruns, [max(existing_subruns) + 1, max(existing_subruns) + 2]
        )
        with nxopen(str(file_path), "r") as nx:
            with pytest.raises(
                RuntimeError, match=r".*not implemented: append or change detector_counts data on existing workspace.*"
            ):
                _InputData.readSubruns(ws, nx["input_data"])


class TestRowSelectionHappensInH5py:
    """`readSubruns` selects rows through the `NXfield`, not after `.nxdata`.

    `detector_counts` is the bulk of a real entry, so reading all of it to keep
    one workspace's rows costs the whole array in memory -- measured at 67.2 MB
    peak versus 3.2 MB for a 5% selection of a 64 MB dataset. Indexing the field
    pushes the selection into h5py's hyperslab machinery instead.

    These pin the part that can silently go wrong: that an index-list selection
    picks the same rows a boolean mask would, in the same order, including when
    the selected rows are not contiguous.
    """

    @staticmethod
    def _written(tmp_path, minimal_HidraWorkspace, sub_runs):
        ws = minimal_HidraWorkspace(with_instrument=True, with_raw_counts=True, sub_runs=np.array(sub_runs))
        path = tmp_path / "rows.nxs"
        with nxopen(str(path), "w") as nx:
            nx["input_data"] = _InputData.init_group([ws])
        return ws, path

    def _recovered(self, path, source_ws, rows, sub_runs):
        target = HidraWorkspace("target")
        target._sample_logs = source_ws._sample_logs
        if rows is not None:
            target._sample_logs = _subset_logs(source_ws._sample_logs, rows, sub_runs)
        with nxopen(str(path), "r") as nx:
            _InputData.readSubruns(target, nx["input_data"], rows)
        return target

    def test_a_contiguous_selection_recovers_those_rows(
        self, tmp_path: Path, minimal_HidraWorkspace: Callable[..., HidraWorkspace]
    ):
        # Arrange
        sub_runs = [1, 2, 3, 4]
        ws, path = self._written(tmp_path, minimal_HidraWorkspace, sub_runs)
        rows = np.array([False, True, True, False])

        # Act
        target = self._recovered(path, ws, rows, sub_runs)

        # Assert
        assert sorted(target._raw_counts) == [2, 3]
        for point in (2, 3):
            np.testing.assert_array_equal(target.get_detector_counts(point), ws.get_detector_counts(point))

    def test_a_non_contiguous_selection_recovers_those_rows(
        self, tmp_path: Path, minimal_HidraWorkspace: Callable[..., HidraWorkspace]
    ):
        """The case an index list could get wrong where a boolean mask would not."""
        # Arrange
        sub_runs = [1, 2, 3, 4]
        ws, path = self._written(tmp_path, minimal_HidraWorkspace, sub_runs)
        rows = np.array([True, False, True, False])

        # Act
        target = self._recovered(path, ws, rows, sub_runs)

        # Assert
        assert sorted(target._raw_counts) == [1, 3]
        for point in (1, 3):
            np.testing.assert_array_equal(target.get_detector_counts(point), ws.get_detector_counts(point))

    def test_no_mask_still_reads_every_row(
        self, tmp_path: Path, minimal_HidraWorkspace: Callable[..., HidraWorkspace]
    ):
        # Arrange
        sub_runs = [1, 2, 3]
        ws, path = self._written(tmp_path, minimal_HidraWorkspace, sub_runs)

        # Act
        target = self._recovered(path, ws, None, sub_runs)

        # Assert
        assert sorted(target._raw_counts) == sub_runs


class TestValidateAppendRefusesAMixedBatch:
    """`validateAppend` refuses a batch with raw counts on some workspaces only.

    `init_group` refuses the same batch, but during the append's mutation phase --
    after the entry's `start_time`/`end_time` have already grown -- so the check
    has to happen here, before anything is resized. This replaces a test that
    pinned the old behaviour, in which such a batch passed.
    """

    def test_an_unloaded_workspace_among_loaded_ones_raises(
        self, minimal_HidraWorkspace: Callable[..., HidraWorkspace]
    ):
        # Arrange
        loaded = minimal_HidraWorkspace(with_instrument=True, with_raw_counts=True, sub_runs=np.array([1, 2]))
        data = _InputData.init_group([loaded])
        empty = minimal_HidraWorkspace(with_instrument=True, with_raw_counts=False, sub_runs=np.array([3, 4]))

        # Act / Assert
        with pytest.raises(RuntimeError, match="some incoming workspaces and not others"):
            _InputData.validateAppend([loaded, empty], data)
