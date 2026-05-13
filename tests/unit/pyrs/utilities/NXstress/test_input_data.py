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


class TestInputData:
    """Test suite for _input_data.py"""

    PROJECT_FILE_A = "HB2B_1017.h5"  # instrument, input data, reduced data, no mask
    PROJECT_FILE_C = "HB2B_1017_w_mask.h5"  # instrument, mask, input data, reduced data

    def test_InputData_init_group_raises_on_existing_data(
        self,
        load_HidraWorkspace: Callable[..., HidraWorkspace],
    ):
        """Verify RuntimeError when trying to append detector_counts data"""
        ws = load_HidraWorkspace(
            file_name=self.PROJECT_FILE_A, name="test_workspace", load_raw_counts=True, load_reduced_diffraction=True
        )

        # Create an existing NXdata group
        existing_data = NXdata()

        with pytest.raises(RuntimeError, match=r".*not implemented: append detector_counts data to NXstress file.*"):
            _InputData.init_group(ws, data=existing_data)

    def test_InputData_init_group_data_values(
        self,
        load_HidraWorkspace: Callable[..., HidraWorkspace],
    ):
        """Verify detector_counts shape and scan_point values match workspace"""
        ws = load_HidraWorkspace(
            file_name=self.PROJECT_FILE_A, name="test_workspace", load_raw_counts=True, load_reduced_diffraction=True
        )

        data = _InputData.init_group(ws)

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
        load_HidraWorkspace: Callable[..., HidraWorkspace],
    ):
        """Verify readSubruns round-trip: write then read back"""
        # Load workspace with raw counts
        ws_write = load_HidraWorkspace(
            file_name=self.PROJECT_FILE_A,
            name="test_workspace_write",
            load_raw_counts=True,
            load_reduced_diffraction=True,
        )

        # Create input data
        data = _InputData.init_group(ws_write)

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
        read_scan_points = list(ws_write._raw_counts.keys())

        for scan_point in original_scan_points:
            assert scan_point in read_scan_points
            original_counts = ws_write.get_detector_counts(scan_point)
            read_counts = ws_read.get_detector_counts(scan_point)
            np.testing.assert_array_equal(read_counts, original_counts)

    def test_InputData_readSubruns_raises_on_scanpoint_mismatch(
        self,
        tmp_path: Path,
        load_HidraWorkspace: Callable[..., HidraWorkspace],
    ):
        """Verify RuntimeError when workspace has subruns that don't match those from input data"""
        # Load workspace with data
        ws = load_HidraWorkspace(
            file_name=self.PROJECT_FILE_A, name="test_workspace", load_raw_counts=True, load_reduced_diffraction=True
        )

        # Create input data and write to file
        data = _InputData.init_group(ws)
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
