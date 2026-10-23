# ruff: noqa: E741, F841
"""
Tests for NXstress read functionality in pyrs/utilities/NXstress/_peaks.py and _fit.py
"""

import numpy as np
from nexusformat.nexus import NXparameters, NXfield
import ast
import pytest
from pathlib import Path
import tempfile

from pyrs.dataobjects.constants import HidraConstants
from pyrs.dataobjects.sample_logs import SampleLogs, SubRuns
from pyrs.utilities.NXstress import _peaks as _peaks_module
from pyrs.utilities.NXstress._peaks import _Peaks
from pyrs.utilities.NXstress._fit import _PeakParameters, _BackgroundParameters
from pyrs.utilities.NXstress.NXstress import NXstress
from pyrs.utilities.NXstress._definitions import FIELD_DTYPE, GROUP_NAME


class TestPeakCollectionRanges:
    """Test suite for _Peaks.peakCollectionRanges"""

    def test_peakCollectionRanges_happy_path(self, minimal_HidraWorkspace, createPeakCollection):
        """Write 3 PeakCollections with distinct keys, read ranges, verify count and span"""
        ws = minimal_HidraWorkspace(with_instrument=False)

        subruns = ws._sample_logs.subruns.raw_copy()
        N_subrun = len(subruns)

        # Create 3 distinct PeakCollections
        peak1 = createPeakCollection(
            peak_tag="Al 111",
            peak_profile="Gaussian",
            background_type="Linear",
            wavelength=25.4,
            projectfilename="/tmp/test.h5",
            runnumber=1,
            N_subrun=N_subrun,
        )

        peak2 = createPeakCollection(
            peak_tag="Si 200",
            peak_profile="Gaussian",
            background_type="Linear",
            wavelength=25.4,
            projectfilename="/tmp/test.h5",
            runnumber=1,
            N_subrun=N_subrun,
        )

        peak3 = createPeakCollection(
            peak_tag="Fe 110",
            peak_profile="Gaussian",
            background_type="Linear",
            wavelength=25.4,
            projectfilename="/tmp/test.h5",
            runnumber=1,
            N_subrun=N_subrun,
        )

        # Write to NXreflections group
        peaks_group = _Peaks.init_group(_Peaks.indexed([peak1, peak2, peak3], ws._sample_logs), ws._sample_logs)

        # Read ranges
        ranges = _Peaks.peakCollectionRanges(peaks_group)

        # Verify we got 3 ranges
        assert len(ranges) == 3

        # Verify each range spans N_subrun entries
        for _discriminators, (phase_name, h, k, l, mask), start, end in ranges:
            assert end - start == N_subrun

        # Verify ranges are contiguous
        expected_start = 0
        for _discriminators, (phase_name, h, k, l, mask), start, end in ranges:
            assert start == expected_start
            expected_start = end

    def test_peakCollectionRanges_interleaved_blocks(self, minimal_HidraWorkspace):
        """Construct NXreflections with non-contiguous blocks for same key → RuntimeError"""
        ws = minimal_HidraWorkspace(with_instrument=False)

        # Manually create NXreflections with interleaved blocks
        peaks = _Peaks._init(ws._sample_logs)

        # Create data for interleaved pattern: Al-111, Si-200, Al-111 (duplicate)
        phase_names = np.array(["Al", "Si", "Al"])
        h_vals = np.array([1, 2, 1])
        k_vals = np.array([1, 0, 1])
        l_vals = np.array([1, 0, 1])
        masks = np.array(["_DEFAULT_", "_DEFAULT_", "_DEFAULT_"])
        scan_points = np.array([1, 1, 2])

        # Resize and fill datasets
        peaks["phase_name"].resize((3,))
        peaks["h"].resize((3,))
        peaks["k"].resize((3,))
        peaks["l"].resize((3,))
        peaks["mask"].resize((3,))
        peaks["scan_point"].resize((3,))
        peaks["center"].resize((3,))
        peaks["center_errors"].resize((3,))

        peaks["phase_name"][:] = phase_names
        peaks["h"][:] = h_vals
        peaks["k"][:] = k_vals
        peaks["l"][:] = l_vals
        peaks["mask"][:] = masks
        peaks["scan_point"][:] = scan_points
        peaks["center"][:] = [1.0, 2.0, 1.5]
        peaks["center_errors"][:] = [0.01, 0.02, 0.015]

        # Should raise RuntimeError about interleaved blocks
        with pytest.raises(RuntimeError, match="Interleaved blocks detected"):
            _Peaks.peakCollectionRanges(peaks)

    def test_peakCollectionRanges_scan_point_order_violation(self, minimal_HidraWorkspace):
        """Non-increasing scan points within block → RuntimeError"""
        ws = minimal_HidraWorkspace(with_instrument=False)

        # Manually create NXreflections with non-increasing scan_point
        peaks = _Peaks._init(ws._sample_logs)

        # Create data with scan_point not strictly increasing: 1, 3, 2 (wrong!)
        phase_names = np.array(["Al", "Al", "Al"])
        h_vals = np.array([1, 1, 1])
        k_vals = np.array([1, 1, 1])
        l_vals = np.array([1, 1, 1])
        masks = np.array(["_DEFAULT_", "_DEFAULT_", "_DEFAULT_"])
        scan_points = np.array([1, 3, 2])  # Not strictly increasing!

        # Resize and fill datasets
        peaks["phase_name"].resize((3,))
        peaks["h"].resize((3,))
        peaks["k"].resize((3,))
        peaks["l"].resize((3,))
        peaks["mask"].resize((3,))
        peaks["scan_point"].resize((3,))
        peaks["center"].resize((3,))
        peaks["center_errors"].resize((3,))

        peaks["phase_name"][:] = phase_names
        peaks["h"][:] = h_vals
        peaks["k"][:] = k_vals
        peaks["l"][:] = l_vals
        peaks["mask"][:] = masks
        peaks["scan_point"][:] = scan_points
        peaks["center"][:] = [1.0, 1.0, 1.0]
        peaks["center_errors"][:] = [0.01, 0.01, 0.01]

        # Should raise RuntimeError about scan_point not strictly increasing
        with pytest.raises(RuntimeError, match="scan_point values are not strictly increasing"):
            _Peaks.peakCollectionRanges(peaks)


class TestValidateNoDuplicatePeaks:
    """Test suite for _Peaks.validateNoDuplicatePeaks"""

    def test_validateNoDuplicatePeaks_no_duplicates(self, createPeakCollection):
        """3 distinct PeakCollections → no error"""
        peak1 = createPeakCollection(
            peak_tag="Al 111",
            peak_profile="Gaussian",
            background_type="Linear",
            wavelength=25.4,
            projectfilename="/tmp/test.h5",
            runnumber=1,
            N_subrun=5,
        )

        peak2 = createPeakCollection(
            peak_tag="Si 200",
            peak_profile="Gaussian",
            background_type="Linear",
            wavelength=25.4,
            projectfilename="/tmp/test.h5",
            runnumber=1,
            N_subrun=5,
        )

        peak3 = createPeakCollection(
            peak_tag="Fe 110",
            peak_profile="Gaussian",
            background_type="Linear",
            wavelength=25.4,
            projectfilename="/tmp/test.h5",
            runnumber=1,
            N_subrun=5,
        )

        # Should not raise any error
        _Peaks.validateNoDuplicatePeaks(_Peaks.indexed([peak1, peak2, peak3]))

    def test_validateNoDuplicatePeaks_with_duplicates(self, createPeakCollection):
        """2 PeakCollections with same key → ValueError with 'Duplicate PeakCollection detected'"""
        peak1 = createPeakCollection(
            peak_tag="Al 111",
            peak_profile="Gaussian",
            background_type="Linear",
            wavelength=25.4,
            projectfilename="/tmp/test.h5",
            runnumber=1,
            N_subrun=5,
        )

        # Create another with same peak_tag and mask (duplicate!)
        peak2 = createPeakCollection(
            peak_tag="Al 111",  # Same as peak1
            peak_profile="Gaussian",
            background_type="Linear",
            wavelength=25.4,
            projectfilename="/tmp/test.h5",
            runnumber=1,
            N_subrun=5,
        )

        # Should raise ValueError
        with pytest.raises(ValueError, match="Duplicate PeakCollection detected"):
            _Peaks.validateNoDuplicatePeaks(_Peaks.indexed([peak1, peak2]))


class TestPeakParametersForRange:
    """Test suite for _PeakParameters.peakParametersForRange"""

    def test_peakParametersForRange(self):
        """Manual NXparameters with known data, slice, verify native params including form_factor→Mixing inversion"""
        # Create manual NXparameters group for Gaussian peak
        pp = NXparameters()
        pp["title"] = NXfield("gaussian", dtype=FIELD_DTYPE.STRING.value)

        # Create datasets with known values
        N = 10
        centers = np.linspace(10.0, 20.0, N)
        heights = np.linspace(100.0, 200.0, N)
        fwhms = np.linspace(0.5, 1.5, N)
        form_factors = np.linspace(0.2, 0.8, N)  # Will be inverted to Mixing

        pp["center"] = NXfield(centers)
        pp["center_errors"] = NXfield(centers * 0.01)
        pp["height"] = NXfield(heights)
        pp["height_errors"] = NXfield(heights * 0.01)
        pp["fwhm"] = NXfield(fwhms)
        pp["fwhm_errors"] = NXfield(fwhms * 0.01)
        pp["form_factor"] = NXfield(form_factors)
        pp["form_factor_errors"] = NXfield(form_factors * 0.01)

        # Slice range [2:5]
        start, end = 2, 5
        native_values, native_errors = _PeakParameters.peakParametersForRange(pp, start, end)

        # Verify we got 3 entries
        assert len(native_values) == 3
        assert len(native_errors) == 3

        # Verify native parameter fields exist (Gaussian: Height, PeakCentre, Sigma, A0, A1)
        assert "Height" in native_values.dtype.names
        assert "PeakCentre" in native_values.dtype.names
        assert "Sigma" in native_values.dtype.names

        # Verify values match sliced data (account for float32 precision)
        np.testing.assert_allclose(native_values["Height"], heights[start:end], rtol=1e-6)
        np.testing.assert_allclose(native_values["PeakCentre"], centers[start:end], rtol=1e-6)

        # Verify errors match (account for float32 precision)
        np.testing.assert_allclose(native_errors["Height"], heights[start:end] * 0.01, rtol=1e-6)
        np.testing.assert_allclose(native_errors["PeakCentre"], centers[start:end] * 0.01, rtol=1e-6)

        # CRITICAL: Verify form_factor was inverted to Mixing (not directly visible in native Gaussian)
        # For Gaussian, we converted via effective parameters where Mixing=1-form_factor
        # Then converted back to native Sigma = FWHM / (2*sqrt(2*ln(2)))
        expected_sigma = fwhms[start:end] / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        np.testing.assert_array_almost_equal(native_values["Sigma"], expected_sigma, decimal=5)


class TestBackgroundParametersForRange:
    """Test suite for _BackgroundParameters.backgroundParametersForRange"""

    def test_backgroundParametersForRange(self):
        """Manual NXparameters, slice, verify A0/A1/A2"""
        # Create manual NXparameters group for background
        bp = NXparameters()
        bp["title"] = NXfield("quadratic", dtype=FIELD_DTYPE.STRING.value)

        # Create datasets with known values
        N = 10
        A0_vals = np.linspace(1.0, 10.0, N)
        A1_vals = np.linspace(0.1, 1.0, N)
        A2_vals = np.linspace(0.01, 0.1, N)

        bp["A0"] = NXfield(A0_vals)
        bp["A0_errors"] = NXfield(A0_vals * 0.05)
        bp["A1"] = NXfield(A1_vals)
        bp["A1_errors"] = NXfield(A1_vals * 0.05)
        bp["A2"] = NXfield(A2_vals)
        bp["A2_errors"] = NXfield(A2_vals * 0.05)

        # Slice range [3:7]
        start, end = 3, 7
        bg_values, bg_errors = _BackgroundParameters.backgroundParametersForRange(bp, start, end)

        # Verify we got 4 entries
        assert len(bg_values) == 4
        assert len(bg_errors) == 4

        # Verify A0, A1, A2 fields exist
        assert "A0" in bg_values.dtype.names
        assert "A1" in bg_values.dtype.names
        assert "A2" in bg_values.dtype.names

        # Verify values match sliced data
        np.testing.assert_array_almost_equal(bg_values["A0"], A0_vals[start:end])
        np.testing.assert_array_almost_equal(bg_values["A1"], A1_vals[start:end])
        np.testing.assert_array_almost_equal(bg_values["A2"], A2_vals[start:end])

        # Verify errors match
        np.testing.assert_array_almost_equal(bg_errors["A0"], A0_vals[start:end] * 0.05)
        np.testing.assert_array_almost_equal(bg_errors["A1"], A1_vals[start:end] * 0.05)
        np.testing.assert_array_almost_equal(bg_errors["A2"], A2_vals[start:end] * 0.05)


class TestPeakCollectionsFromNexus:
    """Test suite for full round-trip read/write"""

    def test_peakCollectionsFromNexus_roundtrip(self, minimal_HidraWorkspace, createPeakCollection):
        """Write PeakCollections via NXstress.write(), read back via peakCollectionsFromNexus, verify match"""
        ws = minimal_HidraWorkspace(with_instrument=True, with_masks=True, with_raw_counts=True)

        subruns = ws._sample_logs.subruns.raw_copy()
        N_subrun = len(subruns)

        # Create test PeakCollections (must use same peak profile)
        peak1 = createPeakCollection(
            peak_tag="Al 111",
            peak_profile="Gaussian",
            background_type="Quadratic",
            wavelength=25.4,
            projectfilename="/tmp/test.h5",
            runnumber=1,
            N_subrun=N_subrun,
        )

        peak2 = createPeakCollection(
            peak_tag="Si 200",
            peak_profile="Gaussian",  # Must match peak1
            background_type="Quadratic",
            wavelength=25.4,
            projectfilename="/tmp/test.h5",
            runnumber=1,
            N_subrun=N_subrun,
        )

        original_peaks = [peak1, peak2]

        # Write to temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test_roundtrip.nxs"

            with NXstress(file_path, mode="w") as nxs:
                nxs.write([ws], [original_peaks])

            # Read back
            with NXstress(file_path, mode="r") as nxs:
                # Access the first entry
                entry_name = "entry"
                entry = nxs._root[entry_name]

                peaks_group = entry[GROUP_NAME.PEAKS]
                fit_group = entry[GROUP_NAME.FIT]

                # Read PeakCollections
                reconstructed_peaks = _Peaks.peakCollectionsFromNexus(peaks_group, fit_group)

        # Verify we got 2 PeakCollections back
        assert len(reconstructed_peaks) == 2

        # Match by sub-index key (not by list position)
        original_by_key = {_Peaks.PeakIndex.sort_key(p): p for p in original_peaks}
        reconstructed_by_key = {
            _Peaks.PeakIndex.sort_key(item.collection): item.collection for item in reconstructed_peaks
        }

        assert set(original_by_key.keys()) == set(reconstructed_by_key.keys())

        # Verify each PeakCollection
        for key in original_by_key:
            orig = original_by_key[key]
            recon = reconstructed_by_key[key]

            # Verify peak_tag parses to same (phase, h, k, l)
            # Exact string match is not required (spaces are not significant)
            orig_phase, orig_hkl = _Peaks._parse_peak_tag(orig.peak_tag)
            recon_phase, recon_hkl = _Peaks._parse_peak_tag(recon.peak_tag)
            assert orig_phase == recon_phase
            assert orig_hkl == recon_hkl

            # Verify mask
            assert orig.mask == recon.mask

            # Verify sub_runs match
            np.testing.assert_array_equal(orig.sub_runs.raw_copy(), recon.sub_runs.raw_copy())

            # Verify d_reference (should be constant for all subruns)
            orig_d, orig_d_err = orig.get_d_reference()
            recon_d, recon_d_err = recon.get_d_reference()
            np.testing.assert_almost_equal(orig_d, recon_d, decimal=5)
            np.testing.assert_almost_equal(orig_d_err, recon_d_err, decimal=5)

            # Verify effective parameters match (within tolerance)
            orig_eff_vals, orig_eff_errs = orig.get_effective_params()
            recon_eff_vals, recon_eff_errs = recon.get_effective_params()

            # Check all effective parameter fields
            for field in ["Center", "Height", "FWHM", "Mixing"]:
                np.testing.assert_allclose(
                    orig_eff_vals[field], recon_eff_vals[field], atol=1e-5, err_msg=f"Mismatch in {field} values"
                )
                np.testing.assert_allclose(
                    orig_eff_errs[field], recon_eff_errs[field], atol=1e-5, err_msg=f"Mismatch in {field} errors"
                )

    def test_peak_tag_roundtrip_multidigit_miller(self, minimal_HidraWorkspace, createPeakCollection):
        """PeakCollection with peak_tag='Fe120100' (h=12,k=1,l=0) round-trips correctly"""
        ws = minimal_HidraWorkspace(with_instrument=True, with_masks=True, with_raw_counts=True)

        subruns = ws._sample_logs.subruns.raw_copy()
        N_subrun = len(subruns)

        # Create PeakCollection with multi-digit Miller indices
        peak = createPeakCollection(
            peak_tag="Fe120100",  # h=12, k=1, l=0 → N_d=2, each index is 2 digits
            peak_profile="Gaussian",
            background_type="Linear",
            wavelength=25.4,
            projectfilename="/tmp/test.h5",
            runnumber=1,
            N_subrun=N_subrun,
        )

        original_peaks = [peak]

        # Write and read back
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test_multidigit.nxs"

            with NXstress(file_path, mode="w") as nxs:
                nxs.write([ws], [original_peaks])

            with NXstress(file_path, mode="r") as nxs:
                entry = nxs._root["entry"]
                peaks_group = entry[GROUP_NAME.PEAKS]
                fit_group = entry[GROUP_NAME.FIT]

                reconstructed_peaks = _Peaks.peakCollectionsFromNexus(peaks_group, fit_group)

        # Verify peak_tag matches
        assert len(reconstructed_peaks) == 1
        assert reconstructed_peaks[0].collection.peak_tag == "Fe120100"

        # Verify parsing produces correct Miller indices
        phase, (h, k, l) = _Peaks._parse_peak_tag(reconstructed_peaks[0].collection.peak_tag)
        assert phase == "Fe"
        assert h == 12
        assert k == 1
        assert l == 0


class TestValidateNoDuplicatePeaksIntegration:
    """Test validateNoDuplicatePeaks integration in NXstress.write()"""

    def test_validateNoDuplicatePeaks_integration_in_write(self, minimal_HidraWorkspace, createPeakCollection):
        """NXstress.write() with duplicates → ValueError before any file content written"""
        ws = minimal_HidraWorkspace(with_instrument=False)

        subruns = ws._sample_logs.subruns.raw_copy()
        N_subrun = len(subruns)

        # Create duplicate PeakCollections
        peak1 = createPeakCollection(
            peak_tag="Al 111",
            peak_profile="Gaussian",
            background_type="Linear",
            wavelength=25.4,
            projectfilename="/tmp/test.h5",
            runnumber=1,
            N_subrun=N_subrun,
        )

        peak2 = createPeakCollection(
            peak_tag="Al 111",  # Duplicate!
            peak_profile="Gaussian",
            background_type="Linear",
            wavelength=25.4,
            projectfilename="/tmp/test.h5",
            runnumber=1,
            N_subrun=N_subrun,
        )

        duplicate_peaks = [peak1, peak2]

        # Attempt to write should raise ValueError
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test_duplicate_check.nxs"

            # The validation happens before any writes, so the error is raised early
            with pytest.raises(ValueError, match="Duplicate PeakCollection detected"):
                with NXstress(file_path, mode="w") as nxs:
                    nxs.write([ws], [duplicate_peaks])


class TestSplitterEnforcesOnlyContiguityAndMonotonicity:
    """What `peakCollectionRanges` requires of row order -- and what it does not.

    Two subspecs rest their design on this: 04b puts discriminators first in the
    sort key so each workspace forms one contiguous super-block, and 04c makes
    append a plain tail-append, on the shared premise that the reader never
    required *global* order. Promoted here from
    `plans/NXstress-prod/probes/a5_peakcollection_ranges.py` so the premise is
    re-checked on every commit rather than once per audit.

    The block rule is reproduced locally in `expected_block_count` rather than
    imported from `_peaks.py`. A test that asked `_peaks.py` for its own rule
    would agree with any rule it later adopted, and so would pass vacuously --
    the point is to fail if the splitter's contract changes.
    """

    @staticmethod
    def peaks_group(phase, h, k, l_, mask, scan_point):
        """A real `NXreflections`, built by the writer and then filled by hand.

        Built through `_Peaks._init` rather than assembled as a dict of
        `NXfield`s. A hand-built stand-in carried only the six columns the
        splitter happens to read today, so it both diverged from the group it
        stands for and would fail with a bare `KeyError` the moment the splitter
        read a seventh -- "a test that reimplements its counterparty is testing
        the reimplementation", as `probes/README.md` puts it about probes.

        The row values are still set by hand, because the point is to present
        orderings the writer would never emit.
        """
        logs = SampleLogs()
        logs[HidraConstants.SUB_RUNS] = np.arange(1, len(phase) + 1)
        group = _Peaks._init(logs, {})

        n = len(phase)
        for name, values in (
            ("phase_name", np.array(phase)),
            ("h", np.array(h, dtype=np.int32)),
            ("k", np.array(k, dtype=np.int32)),
            ("l", np.array(l_, dtype=np.int32)),
            ("mask", np.array(mask)),
            ("scan_point", np.array(scan_point, dtype=np.int32)),
        ):
            group[name].resize((n,))
            group[name][:] = values
        return group

    @staticmethod
    def expected_block_count(phase, h, k, l_, mask) -> int:
        """The rule, restated independently: a block boundary is any key change."""
        keys = list(zip(phase, h, k, l_, mask))
        return 1 + sum(1 for i in range(1, len(keys)) if keys[i] != keys[i - 1])

    def test_globally_unsorted_but_locally_contiguous_is_accepted(self):
        """ "Locally sorted, globally segmented" -- what an appended file looks like."""
        # Arrange: Ni before Fe, so the index is not in lexicographic order.
        phase, h, k, l_, mask = (
            ["Ni"] * 3 + ["Fe"] * 3,
            [2] * 3 + [1] * 3,
            [2] * 3 + [1] * 3,
            [2] * 3 + [1] * 3,
            ["m"] * 6,
        )
        group = self.peaks_group(phase, h, k, l_, mask, [1, 2, 3, 1, 2, 3])

        # Act
        ranges = _Peaks.peakCollectionRanges(group)

        # Assert
        assert len(ranges) == self.expected_block_count(phase, h, k, l_, mask) == 2
        assert [(start, end) for _disc, _key, start, end in ranges] == [(0, 3), (3, 6)]

    def test_key_split_into_two_runs_raises(self):
        """Contiguity IS enforced: the same key may not reappear after another."""
        group = self.peaks_group(["Fe", "Ni", "Fe"], [1, 2, 1], [1, 2, 1], [1, 2, 1], ["m"] * 3, [1, 1, 2])

        with pytest.raises(RuntimeError, match="Interleaved blocks detected"):
            _Peaks.peakCollectionRanges(group)

    def test_descending_scan_point_within_a_run_raises(self):
        """Monotonicity IS enforced, within a run."""
        group = self.peaks_group(["Fe"] * 3, [1] * 3, [1] * 3, [1] * 3, ["m"] * 3, [3, 2, 1])

        with pytest.raises(RuntimeError, match="not strictly increasing"):
            _Peaks.peakCollectionRanges(group)

    def test_scan_points_need_not_be_ordered_across_runs(self):
        """And nothing more than those two: run B may start below where run A ended."""
        phase, h, k, l_, mask = (
            ["Fe"] * 3 + ["Ni"] * 3,
            [1] * 3 + [2] * 3,
            [1] * 3 + [2] * 3,
            [1] * 3 + [2] * 3,
            ["m"] * 6,
        )
        group = self.peaks_group(phase, h, k, l_, mask, [10, 11, 12, 1, 2, 3])

        ranges = _Peaks.peakCollectionRanges(group)

        assert len(ranges) == self.expected_block_count(phase, h, k, l_, mask) == 2

    #: Calls that would mean the splitter searches or reorders the **on-disk index**.
    #: `sorted` is deliberately absent: `_peaks.init_group` sorts the incoming
    #: batch in memory, which is not a search over the index and is what makes each
    #: written block contiguous in the first place.
    ORDERING_CALLS = frozenset({"searchsorted", "argsort", "sort", "bisect", "bisect_left", "bisect_right"})

    def test_no_global_ordering_machinery_in_the_module(self):
        """There is no search over the index, which is why global order is free.

        A source scan rather than a behavioural check, because the claim is about
        absence: a binary search introduced later would not fail any of the cases
        above until a file happened to be globally unsorted.

        Parsed rather than text-matched. `source.count("argsort")` would also fire
        on a *comment* saying "deliberately no argsort over the index here" -- the
        comment this test invites -- so it would punish the documentation written
        because of it. `ast` sees calls only, which is what the claim is about, and
        follows `tests/unit/pyrs/projectfile/test_file_object.py`'s idiom.
        """
        # Arrange
        tree = ast.parse(Path(_peaks_module.__file__).read_text())

        # Act -- `np.argsort(...)` is an Attribute; a bare `argsort(...)` is a Name.
        called = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name in self.ORDERING_CALLS:
                called.add(name)

        # Assert
        assert not called, f"{sorted(called)} would search or reorder the on-disk index"

    def test_the_scan_would_notice_such_a_call(self):
        """The scan above is only worth having if it can fail.

        Its subject is the *absence* of something, so a scan that silently
        matched nothing would pass for the wrong reason -- a mis-spelled node
        type, say, or a walk that never reaches a `Call`.
        """
        # Arrange -- a module shaped like `_peaks.py` with one offending call
        tree = ast.parse("import numpy as np\n\ndef f(a):\n    return np.argsort(a)\n")

        # Act
        called = {
            (node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None))
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
        }

        # Assert
        assert called & self.ORDERING_CALLS == {"argsort"}

    def test_monotonic_scan_points_are_guaranteed_upstream(self):
        """The invariant the reader enforces is one PyRS cannot violate anyway.

        `SubRuns.set` rejects a non-increasing array, so every
        `PeakCollection.sub_runs` is strictly increasing by construction --
        independently of anything NXstress sorts.
        """
        with pytest.raises(RuntimeError, match="not sorted in increasing order"):
            SubRuns(np.array([3, 1, 2]))


class TestDiscriminatorNamesAreCheckedAgainstTheGroup:
    """Passing the wrong discriminator names must fail loudly, not quietly.

    This is the one mistake the splitter could not survive: with the names
    omitted, a two-workspace entry whose blocks happen to be adjacent and share
    a compound key collapses into a single range, and the monotonicity check
    passes because the concatenated scan points are still increasing. Measured
    before the guard: 1 range returned where 2 were correct, **no exception**.
    With differently shaped data the same mistake surfaced instead as
    "Interleaved blocks detected", naming the wrong cause.
    """

    @staticmethod
    def _two_block_group(logs, with_column: bool):
        """Two adjacent blocks sharing one compound key, told apart only by direction."""
        dtypes = {"direction": FIELD_DTYPE.STRING.value} if with_column else {}
        peaks = _Peaks._init(logs, dtypes)
        n = 6
        for name in ("phase_name", "h", "k", "l", "mask", "scan_point", "center", "center_errors"):
            peaks[name].resize((n,))
        peaks["phase_name"][:] = np.array(["Fe"] * n)
        peaks["h"][:] = np.ones(n, dtype=int)
        peaks["k"][:] = np.ones(n, dtype=int)
        peaks["l"][:] = np.zeros(n, dtype=int)
        peaks["mask"][:] = np.array(["_DEFAULT_"] * n)
        peaks["scan_point"][:] = np.arange(1, n + 1)
        peaks["center"][:] = np.full(n, 1.0)
        peaks["center_errors"][:] = np.full(n, 0.01)
        if with_column:
            peaks["direction"].resize((n,))
            peaks["direction"][:] = np.array(["11"] * 3 + ["22"] * 3)
        return peaks

    def test_omitting_the_names_raises_instead_of_merging_two_workspaces(self, minimal_HidraWorkspace):
        # Arrange
        logs = minimal_HidraWorkspace(with_instrument=False, sub_runs=np.arange(1, 7))._sample_logs
        peaks = self._two_block_group(logs, with_column=True)

        # Act / Assert
        with pytest.raises(RuntimeError, match=r"discriminator names given do not match"):
            _Peaks.peakCollectionRanges(peaks)

    def test_naming_a_column_the_group_does_not_have_raises(self, minimal_HidraWorkspace):
        # Arrange
        logs = minimal_HidraWorkspace(with_instrument=False, sub_runs=np.arange(1, 7))._sample_logs
        peaks = self._two_block_group(logs, with_column=False)

        # Act / Assert
        with pytest.raises(RuntimeError, match=r"has no column for discriminator field"):
            _Peaks.peakCollectionRanges(peaks, ("direction",))

    def test_the_correct_names_split_into_two_ranges(self, minimal_HidraWorkspace):
        """The case the guard must not break -- and what the silent failure got wrong."""
        # Arrange
        logs = minimal_HidraWorkspace(with_instrument=False, sub_runs=np.arange(1, 7))._sample_logs
        peaks = self._two_block_group(logs, with_column=True)

        # Act
        ranges = _Peaks.peakCollectionRanges(peaks, ("direction",))

        # Assert
        assert [(start, end) for _, _, start, end in ranges] == [(0, 3), (3, 6)]
        assert [dict(disc)["direction"] for disc, _, _, _ in ranges] == ["11", "22"]

    def test_a_column_less_group_still_reads_with_no_names(self, minimal_HidraWorkspace):
        """Empty is legitimate: it means the entry records no workspace boundary."""
        # Arrange
        logs = minimal_HidraWorkspace(with_instrument=False, sub_runs=np.arange(1, 7))._sample_logs
        peaks = self._two_block_group(logs, with_column=False)

        # Act
        ranges = _Peaks.peakCollectionRanges(peaks)

        # Assert -- one run, because nothing on disk distinguishes the two halves.
        # Not "merged" in any NXstress sense: the rows were concatenated with no
        # boundary recorded, and the splitter has nothing to split on.
        assert [(start, end) for _, _, start, end in ranges] == [(0, 6)]
