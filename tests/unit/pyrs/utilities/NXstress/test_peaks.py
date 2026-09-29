# ruff: noqa: E741, F841
"""
Tests for pyrs/utilities/NXstress/_peaks.py
"""

from collections.abc import Callable
import numpy as np
from nexusformat.nexus import NXreflections

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
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        createPeakCollection: Callable[..., PeakCollection],
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

        peaks = _Peaks.init_group(_Peaks.indexed([peak0], ws._sample_logs), ws._sample_logs)

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
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        createPeakCollection: Callable[..., PeakCollection],
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

        peaks = _Peaks.init_group(_Peaks.indexed([peak0, peak1], ws._sample_logs), ws._sample_logs)

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
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        createPeakCollection: Callable[..., PeakCollection],
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
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        createPeakCollection: Callable[..., PeakCollection],
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

        peaks = _Peaks.init_group(_Peaks.indexed([peak0], ws._sample_logs), ws._sample_logs)

        # qx, qy, qz exist but remain empty (implementation doesn't populate them)
        assert "qx" in peaks
        assert "qy" in peaks
        assert "qz" in peaks

        assert peaks["qx"].shape[0] == 0
        assert peaks["qy"].shape[0] == 0
        assert peaks["qz"].shape[0] == 0

    # NXstress requires `peaks/sx,sy,sz` and defines them as the sample position in
    # the sample reference frame. In PyRS that is `PointList.(vx, vy, vz)`, not the
    # identically-spelled stage-position logs -- so these tests assert the values
    # track `vx`/`vy`/`vz`. They replace a test that pinned the previous behaviour,
    # where all three were written as NaN placeholders.

    def test_Peaks_sample_position_from_coords(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        createPeakCollection: Callable[..., PeakCollection],
    ):
        """Verify sx, sy, sz carry the sample coordinates, tagged with their source"""
        # Arrange
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

        # Act
        peaks = _Peaks.init_group(_Peaks.indexed([peak0], ws._sample_logs), ws._sample_logs)

        # Assert: values are the sample coordinates, not NaN
        for peak_axis, coord_axis in zip(("sx", "sy", "sz"), ("vx", "vy", "vz")):
            assert peaks[peak_axis].shape[0] == N_subrun
            np.testing.assert_allclose(peaks[peak_axis].nxdata, ws._sample_logs[coord_axis])
            # The field name is NXstress's; the value's provenance is PyRS's.
            assert peaks[peak_axis].attrs["local_name"] == coord_axis

    def test_Peaks_sample_position_subset(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        createPeakCollection: Callable[..., PeakCollection],
    ):
        """Verify a PeakCollection covering some scan points gets only those positions.

        Regression for the defect that kept this block commented out: reading the log
        wholesale yields one value per workspace scan point, but this slice needs one
        per scan point *of this PeakCollection*.
        """
        # Arrange: the collection covers fewer subruns than the workspace holds.
        ws = minimal_HidraWorkspace(with_instrument=False)
        N_workspace = len(ws._sample_logs.subruns.raw_copy())
        N_subset = N_workspace - 1
        assert N_subset >= 1, "fixture must supply more than one subrun"
        peak0 = createPeakCollection(
            peak_tag="Al 111",
            peak_profile="Gaussian",
            background_type="Quadratic",
            wavelength=25.4,
            projectfilename="/does/not/exist.h5",
            runnumber=12345,
            N_subrun=N_subset,
        )

        # Act
        peaks = _Peaks.init_group(_Peaks.indexed([peak0], ws._sample_logs), ws._sample_logs)

        # Assert: length and values follow the collection, not the workspace
        assert peaks["sx"].shape[0] == N_subset
        np.testing.assert_allclose(peaks["sx"].nxdata, ws._sample_logs["vx"][:N_subset])

    def test_Peaks_sample_position_nan_fallback(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        createPeakCollection: Callable[..., PeakCollection],
    ):
        """Verify absent sample coordinates give NaN rather than failing the write"""
        # Arrange: drop the coordinate logs entirely
        ws = minimal_HidraWorkspace(with_instrument=False)
        subruns = ws._sample_logs.subruns.raw_copy()
        N_subrun = len(subruns)
        for coord_axis in ("vx", "vy", "vz"):
            del ws._sample_logs[coord_axis]
        peak0 = createPeakCollection(
            peak_tag="Al 111",
            peak_profile="Gaussian",
            background_type="Quadratic",
            wavelength=25.4,
            projectfilename="/does/not/exist.h5",
            runnumber=12345,
            N_subrun=N_subrun,
        )

        # Act
        peaks = _Peaks.init_group(_Peaks.indexed([peak0], ws._sample_logs), ws._sample_logs)

        # Assert
        for peak_axis in ("sx", "sy", "sz"):
            assert all(np.isnan(peaks[peak_axis].nxdata))


class TestPeaksTailAppend:
    """`init_group(..., data=...)` grows an existing index instead of building one.

    Per-family coverage, independent of the end-to-end round trip in
    `test_append.py`: the peak-index family's three groups stay positionally
    aligned only because each grows by the same count from its own current end.
    """

    def test_append_grows_by_exactly_the_new_row_count(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        createPeakCollection: Callable[..., PeakCollection],
    ):
        # Arrange
        ws = minimal_HidraWorkspace(with_instrument=False, sub_runs=np.array([1, 2, 3, 4, 5]))
        logs = ws._sample_logs
        first = createPeakCollection(
            peak_tag="Fe110",
            peak_profile="Gaussian",
            background_type="Linear",
            wavelength=1.486,
            projectfilename="x.h5",
            runnumber=1,
            N_subrun=3,
            sub_runs=np.array([1, 2, 3]),
        )
        peaks = _Peaks.init_group(_Peaks.indexed([first], logs), logs)
        before = np.asarray(peaks["scan_point"].nxdata).copy()

        second = createPeakCollection(
            peak_tag="Si111",
            peak_profile="Gaussian",
            background_type="Linear",
            wavelength=1.486,
            projectfilename="x.h5",
            runnumber=1,
            N_subrun=2,
            sub_runs=np.array([4, 5]),
        )

        # Act
        returned = _Peaks.init_group(_Peaks.indexed([second], logs), logs, data=peaks)

        # Assert
        assert returned is peaks
        assert peaks["scan_point"].shape[0] == 5
        assert np.array_equal(np.asarray(peaks["scan_point"].nxdata)[:3], before)
        assert np.asarray(peaks["scan_point"].nxdata).tolist() == [1, 2, 3, 4, 5]

    def test_append_does_not_re_sort_what_is_already_there(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        createPeakCollection: Callable[..., PeakCollection],
    ):
        """'Locally sorted, globally segmented': the new block follows, it does not merge in."""
        # Arrange -- 'Al' sorts BEFORE 'Fe', so a re-sorting append would put it first
        ws = minimal_HidraWorkspace(with_instrument=False, sub_runs=np.array([1, 2, 3, 4, 5]))
        logs = ws._sample_logs
        first = createPeakCollection(
            peak_tag="Fe110",
            peak_profile="Gaussian",
            background_type="Linear",
            wavelength=1.486,
            projectfilename="x.h5",
            runnumber=1,
            N_subrun=3,
            sub_runs=np.array([1, 2, 3]),
        )
        peaks = _Peaks.init_group(_Peaks.indexed([first], logs), logs)
        second = createPeakCollection(
            peak_tag="Al111",
            peak_profile="Gaussian",
            background_type="Linear",
            wavelength=1.486,
            projectfilename="x.h5",
            runnumber=1,
            N_subrun=2,
            sub_runs=np.array([4, 5]),
        )

        # Act
        _Peaks.init_group(_Peaks.indexed([second], logs), logs, data=peaks)

        # Assert
        names = [v.decode() if isinstance(v, bytes) else str(v) for v in peaks["phase_name"].nxdata]
        assert names == ["Fe", "Fe", "Fe", "Al", "Al"]

    def test_the_appended_index_still_satisfies_the_reader(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        createPeakCollection: Callable[..., PeakCollection],
    ):
        """Contiguity and within-run monotonicity are all `peakCollectionRanges` requires.

        The rule is restated here rather than inferred from the splitter's own
        output, so this fails if the appended layout stops satisfying it.
        """
        # Arrange
        ws = minimal_HidraWorkspace(with_instrument=False, sub_runs=np.array([1, 2, 3, 4, 5, 6, 7]))
        logs = ws._sample_logs
        peaks = None
        for tag, points in (("Fe110", [1, 2, 3]), ("Al111", [4, 5]), ("Si220", [6, 7])):
            collection = createPeakCollection(
                peak_tag=tag,
                peak_profile="Gaussian",
                background_type="Linear",
                wavelength=1.486,
                projectfilename="x.h5",
                runnumber=1,
                N_subrun=len(points),
                sub_runs=np.array(points),
            )
            peaks = _Peaks.init_group(_Peaks.indexed([collection], logs), logs, data=peaks)

        # Act
        ranges = _Peaks.peakCollectionRanges(peaks)

        # Assert -- one contiguous run per key, each covering a strictly increasing span
        assert [(start, end) for _, _, start, end in ranges] == [(0, 3), (3, 5), (5, 7)]
        scan_point = np.asarray(peaks["scan_point"].nxdata)
        for _, _, start, end in ranges:
            block = scan_point[start:end]
            assert np.all(block[1:] > block[:-1])
