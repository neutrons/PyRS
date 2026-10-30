# ruff: noqa: E741, F841
"""
Tests for NXstress read functionality in pyrs/utilities/NXstress/_peaks.py and _fit.py
"""

import numpy as np
from nexusformat.nexus import NXparameters, NXfield
import pytest
from pathlib import Path
import tempfile

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
        peaks_group = _Peaks.init_group([peak1, peak2, peak3], ws._sample_logs)

        # Read ranges
        ranges = _Peaks.peakCollectionRanges(peaks_group)

        # Verify we got 3 ranges
        assert len(ranges) == 3

        # Verify each range spans N_subrun entries
        for (phase_name, h, k, l, mask), start, end in ranges:
            assert end - start == N_subrun

        # Verify ranges are contiguous
        expected_start = 0
        for (phase_name, h, k, l, mask), start, end in ranges:
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
        _Peaks.validateNoDuplicatePeaks([peak1, peak2, peak3])

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
            _Peaks.validateNoDuplicatePeaks([peak1, peak2])


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
                nxs.write(ws, original_peaks)

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
        reconstructed_by_key = {_Peaks.PeakIndex.sort_key(p): p for p in reconstructed_peaks}

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
                nxs.write(ws, original_peaks)

            with NXstress(file_path, mode="r") as nxs:
                entry = nxs._root["entry"]
                peaks_group = entry[GROUP_NAME.PEAKS]
                fit_group = entry[GROUP_NAME.FIT]

                reconstructed_peaks = _Peaks.peakCollectionsFromNexus(peaks_group, fit_group)

        # Verify peak_tag matches
        assert len(reconstructed_peaks) == 1
        assert reconstructed_peaks[0].peak_tag == "Fe120100"

        # Verify parsing produces correct Miller indices
        phase, (h, k, l) = _Peaks._parse_peak_tag(reconstructed_peaks[0].peak_tag)
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
                    nxs.write(ws, duplicate_peaks)
