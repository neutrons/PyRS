"""
Tests for NXstress workspace read functionality (Part 2)
"""

import numpy as np
from nexusformat.nexus import (
    NXsample,
    NXinstrument,
    NXcollection,
    NXfield,
    NXdetector,
    NXdetector_module,
    NXmonochromator,
    NXsource,
    NXtransformations,
)
from nexusformat.nexus.tree import NeXusError

from pyrs.dataobjects.constants import HidraConstants
from pyrs.utilities.NXstress.NXstress import NXstress
from pyrs.utilities.NXstress._sample import _Sample
from pyrs.utilities.NXstress._instrument import _Instrument, _Masks
from pyrs.utilities.NXstress._definitions import DEFAULT_TAG, FIELD_DTYPE
from pyrs.utilities.NXstress._peaks import _Peaks

import pytest


@pytest.fixture
def roundtrip_nxstress(minimal_HidraWorkspace, createPeakCollection, tmp_path):
    """Fixture that writes and reads back a workspace with peaks"""

    ws_original = minimal_HidraWorkspace(
        with_instrument=True, with_masks=True, with_raw_counts=True, with_reduced_diffraction=True
    )

    # Create 2 PeakCollection objects
    subruns = ws_original._sample_logs.subruns.raw_copy()
    N_subrun = len(subruns)

    peak1 = createPeakCollection(
        peak_tag="Al 111",
        peak_profile="Gaussian",
        background_type="Linear",
        wavelength=1.486,
        projectfilename="test.h5",
        runnumber=1017,
        N_subrun=N_subrun,
    )

    peak2 = createPeakCollection(
        peak_tag="Si 220",
        peak_profile="Gaussian",
        background_type="Linear",
        wavelength=1.486,
        projectfilename="test.h5",
        runnumber=1017,
        N_subrun=N_subrun,
    )

    peaks_original = [peak1, peak2]

    # Write to NXstress file
    nxstress_file = tmp_path / "test_roundtrip.nxs"
    with NXstress(nxstress_file, mode="w") as nxs:
        nxs.write(ws_original, peaks_original)

    # Re-open and read back
    with NXstress(nxstress_file, mode="r") as nxs:
        ws_readback, peaks_readback = nxs.read()

    yield ws_original, peaks_original, ws_readback, peaks_readback


class TestWorkspaceRoundtrip:
    """Test suite for workspace reading via roundtrip"""

    def test_workspace_roundtrip_sample_logs(self, roundtrip_nxstress):
        """Verify sample log names and values match between original and readback"""
        ws_original, _, ws_readback, _ = roundtrip_nxstress

        # Verify sample log names match
        original_logs = set(ws_original.get_sample_log_names())
        readback_logs = set(ws_readback.get_sample_log_names())

        # All original logs should be present in readback
        assert original_logs.issubset(readback_logs), f"Missing logs: {original_logs - readback_logs}"

        # Verify sample log values for vx, vy, vz
        for coord_name in HidraConstants.SAMPLE_COORDINATE_NAMES:
            if coord_name in ws_original.get_sample_log_names():
                orig_values = ws_original.get_sample_log_values(coord_name)
                read_values = ws_readback.get_sample_log_values(coord_name)
                np.testing.assert_allclose(
                    orig_values, read_values, atol=1e-5, err_msg=f"Mismatch in {coord_name} values"
                )

        # Verify subruns match
        assert np.array_equal(ws_original.get_sub_runs().raw_copy(), ws_readback.get_sub_runs().raw_copy())

    def test_workspace_roundtrip_wavelength(self, roundtrip_nxstress):
        """Verify wavelength round-trips correctly"""
        ws_original, _, ws_readback, _ = roundtrip_nxstress

        # here `get_wavelength` returns `float | dict[int, float]`
        wl_original = ws_original.get_wavelength(calibrated=True, throw_if_not_set=False)
        wl_readback = ws_readback.get_wavelength(calibrated=True, throw_if_not_set=False)

        if wl_original is not None:
            # `HidraWorkspace` *may* hold its wavelength as a `float`,
            #    so here we just normalize it over all scan-points.
            #      Readback from NeXus will always return a non-scalar.
            if not isinstance(wl_original, dict):
                wl_original = {n: wl_original for n in ws_original.get_sub_runs()}
            assert wl_readback is not None
            np.testing.assert_allclose(list(wl_original.values()), list(wl_readback.values()), rtol=1e-6)

    def test_workspace_roundtrip_instrument(self, roundtrip_nxstress):
        """Verify instrument geometry and detector shift round-trip"""
        ws_original, _, ws_readback, _ = roundtrip_nxstress

        geom_original = ws_original.get_instrument_setup()
        geom_readback = ws_readback.get_instrument_setup()

        # Verify detector size
        assert geom_original.detector_size == geom_readback.detector_size

        # Verify pixel dimensions
        orig_px = geom_original.pixel_dimension
        read_px = geom_readback.pixel_dimension
        np.testing.assert_allclose(orig_px, read_px, rtol=1e-6)

        # Verify arm length
        np.testing.assert_allclose(geom_original.arm_length, geom_readback.arm_length, rtol=1e-6)

        # Verify detector shift (if present)
        shift_original = ws_original.get_detector_shift()
        shift_readback = ws_readback.get_detector_shift()

        if shift_original is not None:
            assert shift_readback is not None
            np.testing.assert_allclose(shift_original.center_shift_x, shift_readback.center_shift_x, rtol=1e-6)
            np.testing.assert_allclose(shift_original.center_shift_y, shift_readback.center_shift_y, rtol=1e-6)
            np.testing.assert_allclose(shift_original.center_shift_z, shift_readback.center_shift_z, rtol=1e-6)
        else:
            assert shift_readback is None

    def test_workspace_roundtrip_masks(self, roundtrip_nxstress):
        """Verify masks round-trip correctly"""
        ws_original, _, ws_readback, _ = roundtrip_nxstress

        # Verify default mask
        default_orig = ws_original.get_detector_mask(is_default=True)
        default_read = ws_readback.get_detector_mask(is_default=True)

        if default_orig is not None:
            assert default_read is not None
            assert np.array_equal(default_orig, default_read)

        # Verify user masks
        for mask_id in ws_original._mask_dict.keys():
            mask_orig = ws_original.get_detector_mask(is_default=False, mask_id=mask_id)
            mask_read = ws_readback.get_detector_mask(is_default=False, mask_id=mask_id)
            assert np.array_equal(mask_orig, mask_read), f"Mask {mask_id} doesn't match"

    def test_workspace_roundtrip_reduced_data(self, roundtrip_nxstress):
        """Verify reduced diffraction data round-trips correctly"""
        ws_original, _, ws_readback, _ = roundtrip_nxstress

        # Verify 2theta matrix
        if ws_original._2theta_matrix is not None:
            assert ws_readback._2theta_matrix is not None
            assert ws_original._2theta_matrix.shape == ws_readback._2theta_matrix.shape
            np.testing.assert_allclose(ws_original._2theta_matrix, ws_readback._2theta_matrix, atol=1e-5)

        # Verify diff_data_set and var_data_set for each mask
        for mask_id in ws_original._diff_data_set.keys():
            assert mask_id in ws_readback._diff_data_set, f"Mask {mask_id} missing in readback diff_data_set"

            orig_data = ws_original._diff_data_set[mask_id]
            read_data = ws_readback._diff_data_set[mask_id]
            assert orig_data.shape == read_data.shape
            np.testing.assert_allclose(orig_data, read_data, atol=1e-5)

        for mask_id in ws_original._var_data_set.keys():
            assert mask_id in ws_readback._var_data_set, f"Mask {mask_id} missing in readback var_data_set"

            orig_var = ws_original._var_data_set[mask_id]
            read_var = ws_readback._var_data_set[mask_id]
            assert orig_var.shape == read_var.shape
            np.testing.assert_allclose(orig_var, read_var, atol=1e-5)

    def test_workspace_roundtrip_raw_counts(self, roundtrip_nxstress):
        """Verify raw counts round-trip correctly"""
        ws_original, _, ws_readback, _ = roundtrip_nxstress

        # Verify all subruns have raw counts
        for subrun in ws_original._raw_counts.keys():
            assert subrun in ws_readback._raw_counts, f"Subrun {subrun} missing in readback raw_counts"

            orig_counts = ws_original.get_detector_counts(subrun)
            read_counts = ws_readback.get_detector_counts(subrun)
            assert orig_counts.shape == read_counts.shape
            np.testing.assert_allclose(orig_counts, read_counts, atol=1e-5)

    def test_full_roundtrip(self, roundtrip_nxstress):
        """Comprehensive test: verify workspace and peaks together"""
        ws_original, peaks_original, ws_readback, peaks_readback = roundtrip_nxstress

        # Verify peak collection count
        assert len(peaks_original) == len(peaks_readback)

        # Verify each peak collection
        for peak_orig, peak_read in zip(peaks_original, peaks_readback):
            # we don't care about small changes to the format (e.g. omitted spaces), we only care
            #   that they parse to the same `(<phase>, h, k, l)` tuples
            assert _Peaks._parse_peak_tag(peak_orig.peak_tag) == _Peaks._parse_peak_tag(peak_read.peak_tag)
            assert peak_orig.peak_profile == peak_read.peak_profile
            assert peak_orig.background_type == peak_read.background_type

            # Verify subruns match
            assert np.array_equal(peak_orig._sub_run_array.raw_copy(), peak_read._sub_run_array.raw_copy())


class TestReadErrors:
    """Test error handling in read operations"""

    def test_read_nonexistent_entry(self, minimal_HidraWorkspace, tmp_path):
        """Attempt to read non-existent entry → KeyError"""
        ws = minimal_HidraWorkspace(with_instrument=True)

        nxstress_file = tmp_path / "test_nonexistent.nxs"
        with NXstress(nxstress_file, mode="w") as nxs:
            nxs.write(ws, [])

        with pytest.raises(NeXusError, match=r".*Invalid path.*"):
            with NXstress(nxstress_file, mode="r") as nxs:
                nxs.read(entry_number=99)

    def test_read_outside_context_manager(self, minimal_HidraWorkspace, tmp_path):
        """Call read() outside context manager → RuntimeError"""
        # Build a valid NXstress file
        ws = minimal_HidraWorkspace(with_instrument=True)

        nxstress_file = tmp_path / "test_outside_context.nxs"
        with NXstress(nxstress_file, mode="w") as nxs:
            nxs.write(ws, [])

        # Now try to read without context manager
        nxs = NXstress(nxstress_file, mode="r")
        with pytest.raises(RuntimeError, match="context manager"):
            nxs.read()


class TestStandaloneMethods:
    """Test standalone read methods"""

    def test_sampleLogsFromNexus_standalone(self):
        """Build NXsample manually, read with sampleLogsFromNexus"""
        # Create NXsample group manually
        sample = NXsample()

        scan_points = np.array([1, 2, 3], dtype=np.int32)
        sample["scan_point"] = NXfield(scan_points, units="")

        # Add coordinates
        sample["vx"] = NXfield(np.array([0.0, 1.0, 2.0], dtype=np.float32), units="mm")
        sample["vy"] = NXfield(np.array([0.0, 0.0, 0.0], dtype=np.float32), units="mm")
        sample["vz"] = NXfield(np.array([0.0, 0.0, 0.0], dtype=np.float32), units="mm")

        # Add name and formula
        sample["name"] = NXfield("TestSample")
        sample["chemical_formula"] = NXfield("H2O")

        # Add logs collection
        logs_coll = NXcollection()
        logs_coll["test_log"] = NXfield(np.array([10.0, 20.0, 30.0]), local_name="test:log:pv", units="V")
        sample["logs"] = logs_coll

        # Read back
        sample_logs = _Sample.sampleLogsFromNexus(sample)

        # Verify
        assert np.array_equal(sample_logs.subruns.raw_copy(), scan_points)
        assert "vx" in sample_logs
        assert "test:log:pv" in sample_logs
        np.testing.assert_array_equal(sample_logs["vx"], np.array([0.0, 1.0, 2.0]))
        np.testing.assert_array_equal(sample_logs["test:log:pv"], np.array([10.0, 20.0, 30.0]))

    def test_instrumentFromNexus_standalone(self):
        """Build NXinstrument manually, read with instrumentFromNexus"""
        # Create NXinstrument group manually
        inst = NXinstrument()
        inst["name"] = "HB2B"

        # Add source
        inst["SOURCE"] = NXsource(type="Reactor Neutron Source", probe="neutron")

        # Add monochromator with wavelength
        mono = NXmonochromator()
        mono["wavelength"] = NXfield(1.486, units="angstrom")
        inst["monochromator"] = mono

        # Add detector
        det = NXdetector()
        det["type"] = "He_3 PSD"

        # Detector bank
        det["detector_bank"] = NXdetector_module(
            data_size=NXfield(np.array([512, 512], dtype=np.int64)),
            fast_pixel_direction=NXfield(np.array(0.001, dtype=np.float64), units="m"),
            slow_pixel_direction=NXfield(np.array(0.001, dtype=np.float64), units="m"),
        )

        # Transformations
        trans = NXtransformations()
        trans.attrs["calibrated"] = True
        trans["distance"] = NXfield(2.0, units="m")
        trans["translation_x"] = NXfield(0.01, units="m")
        trans["translation_y"] = NXfield(0.02, units="m")
        trans["translation_z"] = NXfield(0.03, units="m")
        trans["rotation_x"] = NXfield(1.0, units="deg")
        trans["rotation_y"] = NXfield(2.0, units="deg")
        trans["rotation_z"] = NXfield(3.0, units="deg")
        trans["two_theta_zero"] = NXfield(0.5, units="deg")

        det["transformations"] = trans
        inst["DETECTOR"] = det

        # Read back
        geometry, shift, wavelength = _Instrument.instrumentFromNexus(inst)

        # Verify geometry
        assert geometry.detector_size == (512, 512)
        np.testing.assert_allclose(geometry.pixel_dimension, (0.001, 0.001))
        np.testing.assert_allclose(geometry.arm_length, 2.0)

        # Verify shift
        assert shift is not None
        np.testing.assert_allclose(shift.center_shift_x, 0.01)
        np.testing.assert_allclose(shift.center_shift_y, 0.02)
        np.testing.assert_allclose(shift.center_shift_z, 0.03)
        np.testing.assert_allclose(shift.rotation_x, 1.0)
        np.testing.assert_allclose(shift.rotation_y, 2.0)
        np.testing.assert_allclose(shift.rotation_z, 3.0)
        np.testing.assert_allclose(shift.two_theta_0, 0.5)

        # Verify wavelength
        np.testing.assert_allclose(wavelength, 1.486)

    def test_masksFromNexus_standalone(self):
        """Build masks NXcollection manually, read with masksFromNexus"""
        # Create masks collection manually
        masks = NXcollection()

        # Add mask names
        mask_names = np.array([DEFAULT_TAG, "mask_A", "mask_B"], dtype=FIELD_DTYPE.STRING.value)
        masks["names"] = NXfield(mask_names)

        # Add detector masks
        det_coll = NXcollection()
        det_coll[DEFAULT_TAG] = NXfield(np.ones(100, dtype=bool), units="")
        det_coll["mask_A"] = NXfield(np.zeros(100, dtype=bool), units="")
        masks["detector"] = det_coll

        # Add solid_angle mask
        sa_coll = NXcollection()
        sa_coll["mask_B"] = NXfield(np.ones(100, dtype=bool), units="")
        masks["solid_angle"] = sa_coll

        # Read back
        default_mask, mask_dict = _Masks.masksFromNexus(masks)

        # Verify
        assert default_mask is not None
        assert len(default_mask) == 100
        assert "mask_A" in mask_dict
        assert "mask_B" in mask_dict
        assert len(mask_dict) == 2
