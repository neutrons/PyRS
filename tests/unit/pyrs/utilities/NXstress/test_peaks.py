# ruff: noqa: E741, F841
"""
Tests for pyrs/utilities/NXstress/_peaks.py
"""

from collections.abc import Callable
import numpy as np
from nexusformat.nexus import NXreflections
import pytest

from pyrs.core.workspaces import HidraWorkspace
from pyrs.peaks.peak_collection import PeakCollection
from pyrs.utilities.NXstress._peaks import _Peaks


class TestPeaks:
    """Test suite for _peaks.py"""

    def test_Peaks_init_empty(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
    ):
        """Verify _Peaks._init creates empty datasets with correct dtypes/units"""
        ws = minimal_HidraWorkspace(with_instrument=False)

        logs = ws._sample_logs
        peaks = _Peaks._init(logs)

        assert isinstance(peaks, NXreflections)

        # Verify all required fields exist and array fields are empty
        array_fields = [
            "scan_point",
            "h",
            "k",
            "l",
            "phase_name",
            "mask",
            "qx",
            "qy",
            "qz",
            "center",
            "center_errors",
            "sx",
            "sy",
            "sz",
        ]

        for field in array_fields:
            assert field in peaks
            assert peaks[field].shape[0] == 0

        # Verify scalar field
        assert "center_type" in peaks
        assert peaks["center_type"].nxdata == "d-spacing"

    def test_Peaks_init_group_data_values(
        self, minimal_HidraWorkspace: Callable[..., HidraWorkspace], createPeakCollection: Callable[..., PeakCollection]
    ):
        """Verify one PeakCollection creates N_scan rows with correct values"""
        ws = minimal_HidraWorkspace(with_instrument=False)

        subruns = ws._sample_logs.subruns.raw_copy()
        N_subrun = len(subruns)

        peak0 = createPeakCollection(
            peak_tag="Al 251540",
            peak_profile="Gaussian",
            background_type="Quadratic",
            wavelength=25.4,
            projectfilename="/does/not/exist.h5",
            runnumber=12345,
            N_subrun=N_subrun,
        )

        peaks = _Peaks.init_group([peak0], ws._sample_logs)

        assert isinstance(peaks, NXreflections)

        # Verify shape
        assert peaks["h"].shape[0] == N_subrun
        assert peaks["k"].shape[0] == N_subrun
        assert peaks["l"].shape[0] == N_subrun
        assert peaks["phase_name"].shape[0] == N_subrun
        assert peaks["mask"].shape[0] == N_subrun
        assert peaks["scan_point"].shape[0] == N_subrun
        assert peaks["center"].shape[0] == N_subrun
        assert peaks["center_errors"].shape[0] == N_subrun

        # Verify values
        # Parse peak tag to get expected h, k, l
        phase, (h, k, l) = _Peaks._parse_peak_tag(peak0.peak_tag)

        # All rows should have same h, k, l
        assert all(peaks["h"].nxdata == h)
        assert all(peaks["k"].nxdata == k)
        assert all(peaks["l"].nxdata == l)

        # All rows should have same phase_name
        assert all(p == phase for p in peaks["phase_name"].nxdata)

        # scan_point should match subruns
        np.testing.assert_array_equal(peaks["scan_point"].nxdata, subruns)

    def test_Peaks_init_group_multiple_peaks(
        self, minimal_HidraWorkspace: Callable[..., HidraWorkspace], createPeakCollection: Callable[..., PeakCollection]
    ):
        """Verify two PeakCollections create 2×N_scan rows in lexicographic sort order"""
        ws = minimal_HidraWorkspace(with_instrument=False)

        subruns = ws._sample_logs.subruns.raw_copy()
        N_subrun = len(subruns)

        # Create two peaks - they will be sorted by PeakIndex.sort_key
        peak0 = createPeakCollection(
            peak_tag="Al 251540",  # (2, 5, 1540) -> (25, 15, 40) after parsing "251540"
            peak_profile="Gaussian",
            background_type="Quadratic",
            wavelength=25.4,
            projectfilename="/does/not/exist.h5",
            runnumber=12345,
            N_subrun=N_subrun,
        )

        peak1 = createPeakCollection(
            peak_tag="Al 111",
            peak_profile="Gaussian",
            background_type="Quadratic",
            wavelength=25.4,
            projectfilename="/does/not/exist.h5",
            runnumber=12345,
            N_subrun=N_subrun,
        )

        peaks = _Peaks.init_group([peak0, peak1], ws._sample_logs)

        # Should have 2 * N_subrun rows
        assert peaks["h"].shape[0] == 2 * N_subrun

        # Verify sorting - first N_subrun rows should be from peak with lower sort key
        # Sort key is (phase_name, h, k, l, mask)
        # Both are "Al", so it's sorted by (h, k, l)
        phase0, hkl0 = _Peaks._parse_peak_tag(peak0.peak_tag)
        phase1, hkl1 = _Peaks._parse_peak_tag(peak1.peak_tag)

        # Determine which peak should come first
        key0 = (phase0, *hkl0, peak0.mask)
        key1 = (phase1, *hkl1, peak1.mask)

        if key0 < key1:
            first_peak = peak0
            first_hkl = hkl0
        else:
            first_peak = peak1
            first_hkl = hkl1

        # Verify first N_subrun rows match the first peak in sort order
        h, k, l = first_hkl
        assert all(peaks["h"].nxdata[:N_subrun] == h)
        assert all(peaks["k"].nxdata[:N_subrun] == k)
        assert all(peaks["l"].nxdata[:N_subrun] == l)

    def test_PeakIndex_sort_key(
        self, minimal_HidraWorkspace: Callable[..., HidraWorkspace], createPeakCollection: Callable[..., PeakCollection]
    ):
        """Verify PeakIndex.sort_key returns correct tuple for sorting"""
        ws = minimal_HidraWorkspace(with_instrument=False)

        subruns = ws._sample_logs.subruns.raw_copy()
        N_subrun = len(subruns)

        peak0 = createPeakCollection(
            peak_tag="Al 111",
            peak_profile="Gaussian",
            background_type="Quadratic",
            wavelength=25.4,
            projectfilename="/does/not/exist.h5",
            runnumber=12345,
            N_subrun=N_subrun,
        )

        peak1 = createPeakCollection(
            peak_tag="Si 222",
            peak_profile="Gaussian",
            background_type="Quadratic",
            wavelength=25.4,
            projectfilename="/does/not/exist.h5",
            runnumber=12345,
            N_subrun=N_subrun,
        )

        # Get sort keys
        key0 = _Peaks.PeakIndex.sort_key(peak0)
        key1 = _Peaks.PeakIndex.sort_key(peak1)

        # Keys should be tuples (phase_name, h, k, l, mask)
        assert len(key0) == 5
        assert len(key1) == 5

        # Verify they can be compared for sorting
        assert key0 != key1
        assert (key0 < key1) or (key0 > key1)

    def test_Peaks_qxyz_nan(
        self, minimal_HidraWorkspace: Callable[..., HidraWorkspace], createPeakCollection: Callable[..., PeakCollection]
    ):
        """Verify qx, qy, qz fields exist but remain empty after init_group since implementation doesn't populate them"""
        ws = minimal_HidraWorkspace(with_instrument=False)

        subruns = ws._sample_logs.subruns.raw_copy()
        N_subrun = len(subruns)

        peak0 = createPeakCollection(
            peak_tag="Al 111",
            peak_profile="Gaussian",
            background_type="Quadratic",
            wavelength=25.4,
            projectfilename="/does/not/exist.h5",
            runnumber=12345,
            N_subrun=N_subrun,
        )

        peaks = _Peaks.init_group([peak0], ws._sample_logs)

        # qx, qy, qz exist but remain empty (implementation doesn't populate them)
        assert "qx" in peaks
        assert "qy" in peaks
        assert "qz" in peaks

        assert peaks["qx"].shape[0] == 0
        assert peaks["qy"].shape[0] == 0
        assert peaks["qz"].shape[0] == 0

    def test_Peaks_sxyz_nan(
        self, minimal_HidraWorkspace: Callable[..., HidraWorkspace], createPeakCollection: Callable[..., PeakCollection]
    ):
        """Verify sx, sy, sz are filled with NaN after init_group"""
        ws = minimal_HidraWorkspace(with_instrument=False)

        subruns = ws._sample_logs.subruns.raw_copy()
        N_subrun = len(subruns)

        peak0 = createPeakCollection(
            peak_tag="Al 111",
            peak_profile="Gaussian",
            background_type="Quadratic",
            wavelength=25.4,
            projectfilename="/does/not/exist.h5",
            runnumber=12345,
            N_subrun=N_subrun,
        )

        peaks = _Peaks.init_group([peak0], ws._sample_logs)

        # sx, sy, sz should exist and be filled with NaN
        assert "sx" in peaks
        assert "sy" in peaks
        assert "sz" in peaks

        assert peaks["sx"].shape[0] == N_subrun
        assert peaks["sy"].shape[0] == N_subrun
        assert peaks["sz"].shape[0] == N_subrun

        # All values should be NaN
        assert all(np.isnan(peaks["sx"].nxdata))
        assert all(np.isnan(peaks["sy"].nxdata))
        assert all(np.isnan(peaks["sz"].nxdata))
