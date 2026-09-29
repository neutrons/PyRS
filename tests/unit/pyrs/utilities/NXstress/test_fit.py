"""
Tests for pyrs/utilities/NXstress/_fit.py
"""

from collections.abc import Callable
import h5py
import numpy as np
from nexusformat.nexus import NXdata, NXnote, NXparameters, NXprocess
import pytest

from pyrs.core.workspaces import HidraWorkspace
from pyrs.peaks.peak_collection import PeakCollection
from pyrs.utilities.NXstress._fit import _BackgroundParameters, _Diffractogram, _Fit, _PeakParameters
from pyrs.utilities.NXstress._definitions import DEFAULT_TAG, GROUP_NAME
from pyrs.utilities.NXstress._peaks import _Peaks
from pyrs.utilities.NXstress.NXstress import NXstress


class TestFit:
    """Test suite for _fit.py"""

    def test_PeakParameters_data_values(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        createPeakCollection: Callable[..., PeakCollection],
    ):
        """Verify numeric values in peak parameters match get_effective_params()"""
        ws = minimal_HidraWorkspace(with_instrument=False)

        subruns = ws._sample_logs.subruns.raw_copy()
        N_subrun = len(subruns)

        peak0 = createPeakCollection(
            peak_tag="Al 251540",
            peak_profile="PseudoVoigt",
            background_type="Quadratic",
            wavelength=25.4,
            projectfilename="/does/not/exist.h5",
            runnumber=12345,
            N_subrun=N_subrun,
        )

        params_value, params_error = peak0.get_effective_params()

        peak_params = _PeakParameters.init_group(_Peaks.indexed([peak0]))

        assert isinstance(peak_params, NXparameters)

        # Verify all required fields exist
        assert "center" in peak_params
        assert "center_errors" in peak_params
        assert "height" in peak_params
        assert "height_errors" in peak_params
        assert "fwhm" in peak_params
        assert "fwhm_errors" in peak_params
        assert "form_factor" in peak_params
        assert "form_factor_errors" in peak_params

        # Verify data values match
        np.testing.assert_array_almost_equal(peak_params["center"].nxdata, params_value["Center"].astype(np.float64))
        np.testing.assert_array_almost_equal(peak_params["height"].nxdata, params_value["Height"].astype(np.float64))
        np.testing.assert_array_almost_equal(peak_params["fwhm"].nxdata, params_value["FWHM"].astype(np.float64))

        # Form factor is (1.0 - Mixing)
        expected_form_factor = (1.0 - params_value["Mixing"]).astype(np.float64)
        np.testing.assert_array_almost_equal(peak_params["form_factor"].nxdata, expected_form_factor)

    def test_PeakParameters_multiple_peaks(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        createPeakCollection: Callable[..., PeakCollection],
    ):
        """Verify two PeakCollections create 2×N_scan rows in sort order"""
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

        peak1 = createPeakCollection(
            peak_tag="Al 111",
            peak_profile="Gaussian",
            background_type="Quadratic",
            wavelength=25.4,
            projectfilename="/does/not/exist.h5",
            runnumber=12345,
            N_subrun=N_subrun,
        )

        peak_params = _PeakParameters.init_group(_Peaks.indexed([peak0, peak1]))

        # Should have 2 * N_subrun rows
        assert peak_params["center"].shape[0] == 2 * N_subrun

    def test_PeakParameters_mismatched_profile_raises(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        createPeakCollection: Callable[..., PeakCollection],
    ):
        """Verify ValueError when PeakCollections have different peak_profile"""
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
            peak_tag="Si 111",
            peak_profile="PseudoVoigt",  # Different!
            background_type="Quadratic",
            wavelength=25.4,
            projectfilename="/does/not/exist.h5",
            runnumber=12345,
            N_subrun=N_subrun,
        )

        with pytest.raises(ValueError, match=r".*must share the same peak profile.*"):
            _PeakParameters.init_group(_Peaks.indexed([peak0, peak1]))

    def test_BackgroundParameters_data_values(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        createPeakCollection: Callable[..., PeakCollection],
    ):
        """Verify A0, A1, A2 (and errors) match get_effective_params()"""
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

        params_value, params_error = peak0.get_effective_params()

        bg_params = _BackgroundParameters.init_group(_Peaks.indexed([peak0]))

        assert isinstance(bg_params, NXparameters)

        # Verify all background parameters
        for param in ["A0", "A1", "A2"]:
            assert param in bg_params
            assert f"{param}_errors" in bg_params

            np.testing.assert_array_almost_equal(bg_params[param].nxdata, params_value[param].astype(np.float64))
            np.testing.assert_array_almost_equal(
                bg_params[f"{param}_errors"].nxdata, params_error[param].astype(np.float64)
            )

    def test_BackgroundParameters_multiple_peaks(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        createPeakCollection: Callable[..., PeakCollection],
    ):
        """Verify two PeakCollections create 2×N_scan rows"""
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
            peak_tag="Al 222",
            peak_profile="Gaussian",
            background_type="Quadratic",
            wavelength=25.4,
            projectfilename="/does/not/exist.h5",
            runnumber=12345,
            N_subrun=N_subrun,
        )

        bg_params = _BackgroundParameters.init_group(_Peaks.indexed([peak0, peak1]))

        # Should have 2 * N_subrun rows
        assert bg_params["A0"].shape[0] == 2 * N_subrun

    def test_BackgroundParameters_mismatched_type_raises(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        createPeakCollection: Callable[..., PeakCollection],
    ):
        """Verify ValueError when PeakCollections have different background_type"""
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
            peak_tag="Si 111",
            peak_profile="Gaussian",
            background_type="Linear",  # Different!
            wavelength=25.4,
            projectfilename="/does/not/exist.h5",
            runnumber=12345,
            N_subrun=N_subrun,
        )

        with pytest.raises(ValueError, match=r".*must share the same background type.*"):
            _BackgroundParameters.init_group(_Peaks.indexed([peak0, peak1]))

    def test_Diffractogram_data_key_default(self):
        """Verify _diffraction_data_key returns `None` for DEFAULT_TAG"""
        data_key = _Diffractogram._diffraction_data_key(DEFAULT_TAG)

        assert data_key is None

    def test_Diffractogram_data_keys_named(self):
        """Verify _diffraction_data_keys returns proper keys for named mask"""
        data_key = _Diffractogram._diffraction_data_key("my_mask")

        assert data_key == "my_mask"

    def test_Diffractogram_init_no_reduced_data_raises(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
    ):
        """Verify RuntimeError when workspace._2theta_matrix is None"""
        ws = minimal_HidraWorkspace(with_instrument=False)

        # Set _2theta_matrix to None to simulate no reduced data
        ws._2theta_matrix = None

        with pytest.raises(RuntimeError, match=r".*doesn't include any reduced data.*"):
            _Diffractogram._init([ws])

    def test_Diffractogram_init_group_missing_mask_raises(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        createPeakCollection: Callable[..., PeakCollection],
    ):
        """Verify RuntimeError when mask data not in workspace"""
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

        # Try to create diffractogram for non-existent mask
        with pytest.raises(RuntimeError, match=r".*is not present in the workspace.*"):
            _Diffractogram.init_group([ws], "non_existent_mask", _Peaks.indexed([peak0]))

    def test_Diffractogram_data_values(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        createPeakCollection: Callable[..., PeakCollection],
    ):
        """Verify diffractogram/diffractogram_errors match workspace arrays"""
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

        dgram = _Diffractogram.init_group([ws], DEFAULT_TAG, _Peaks.indexed([peak0]))

        assert isinstance(dgram, NXdata)

        # Verify required fields
        assert "diffractogram" in dgram
        assert "diffractogram_errors" in dgram
        assert "fit" in dgram
        assert "fit_errors" in dgram

        # Verify data matches workspace (use allclose for float32 comparison)
        data_key = _Diffractogram._diffraction_data_key(DEFAULT_TAG)
        expected_data = ws._diff_data_set[data_key]
        expected_errors = ws._var_data_set[data_key]

        np.testing.assert_allclose(dgram["diffractogram"].nxdata, expected_data, rtol=1e-6, equal_nan=True)
        np.testing.assert_allclose(dgram["diffractogram_errors"].nxdata, expected_errors, rtol=1e-6, equal_nan=True)

        # fit and fit_errors should be empty
        assert dgram["fit"].shape == (0, 0)
        assert dgram["fit_errors"].shape == (0, 0)

    def test_Fit_init_fields(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
    ):
        """Verify _Fit._init creates fields: date, program, raw_data_file, DESCRIPTION"""
        ws = minimal_HidraWorkspace(with_instrument=False)

        logs = ws._sample_logs
        fit = _Fit._init([logs], processing_description="Test description", processing_time="2024-01-15T10:30:00")

        assert isinstance(fit, NXprocess)
        assert "date" in fit
        assert "program" in fit
        assert "raw_data_file" in fit
        assert "DESCRIPTION" in fit

        assert fit["program"] == "PyRS"
        assert fit["date"] == "2024-01-15T10:30:00"
        assert isinstance(fit["DESCRIPTION"], NXnote)

    def test_Fit_multiple_masks(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        createPeakCollection: Callable[..., PeakCollection],
    ):
        """Verify a workspace with multiple named reduced-diffraction masks creates exactly
        one DIFFRACTOGRAM per configured mask, plus the always-present default.
        """
        ws = minimal_HidraWorkspace(with_instrument=False, with_reduced_diffraction=False)

        subruns = ws._sample_logs.subruns.raw_copy()
        N_subrun = len(subruns)

        # `_Fit.init_group` counts one DIFFRACTOGRAM per key in `ws._diff_data_set` (plus
        # the always-present default, keyed `None`) -- `minimal_HidraWorkspace`'s own
        # `with_reduced_diffraction` only ever creates that single default entry, so the
        # extra named masks are configured here directly.
        n_two_theta = 20
        two_theta_matrix = np.tile(np.linspace(60.0, 120.0, n_two_theta), (N_subrun, 1))
        mask_names = ("mask1", "mask2")
        diff_data_set: dict[str | None, np.ndarray] = {None: np.ones((N_subrun, n_two_theta))}
        var_data_set: dict[str | None, np.ndarray] = {None: np.ones((N_subrun, n_two_theta))}
        for mask_name in mask_names:
            diff_data_set[mask_name] = np.ones((N_subrun, n_two_theta))
            var_data_set[mask_name] = np.ones((N_subrun, n_two_theta))
            # The mask itself must exist, not just a reduction keyed by its name: a
            # diffractogram key references a detector mask, and `_Fit.init_group` now
            # checks that the reference resolves. A real reduction using `mask1` had
            # `mask1` loaded; this test previously only pretended.
            ws.set_detector_mask(np.ones(16, dtype=np.int64), False, mask_name)
        ws.set_reduced_diffraction_data_set(two_theta_matrix, diff_data_set, var_data_set)

        peak0 = createPeakCollection(
            peak_tag="Al 111",
            peak_profile="Gaussian",
            background_type="Quadratic",
            wavelength=25.4,
            projectfilename="/does/not/exist.h5",
            runnumber=12345,
            N_subrun=N_subrun,
        )

        fit = _Fit.init_group([ws], _Peaks.indexed([peak0], ws._sample_logs), [ws._sample_logs])

        # Count NXdata groups (diffractograms)
        diffractogram_count = sum(1 for key in fit.keys() if isinstance(fit[key], NXdata))

        # One diffractogram per configured mask, plus the always-present default.
        assert diffractogram_count == len(mask_names) + 1

    def test_Fit_duplicate_diffractogram_raises(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        createPeakCollection: Callable[..., PeakCollection],
    ):
        """Verify RuntimeError when diffractogram name collision occurs"""
        # This test checks the internal logic - would need to manipulate
        # workspace to have duplicate mask names, which is prevented elsewhere
        # For now, we'll skip this as it's hard to trigger in practice
        pass

    def test_validateWorkspaceAndPeaksData_valid(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        createPeakCollection: Callable[..., PeakCollection],
    ):
        """Verify validation passes for matching workspace and peaks data"""
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

        # Should not raise
        _Fit.validateWorkspaceAndPeaksData(ws, [peak0])

    def test_validateWorkspaceAndPeaksData_missing_scan_points(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        createPeakCollection: Callable[..., PeakCollection],
    ):
        """Verify ValueError when PeakCollection references missing scan points"""
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

        # Add a non-existent scan point to the peak collection
        import numpy as np
        from pyrs.dataobjects.sample_logs import SubRuns

        # Create sub_runs with extra scan points not in workspace
        extra_subruns = np.append(subruns, [9999, 10000])
        peak0._sub_run_array = SubRuns(extra_subruns)

        with pytest.raises(ValueError, match=r".*not present in workspace.*"):
            _Fit.validateWorkspaceAndPeaksData(ws, [peak0])

    def test_validateWorkspaceAndPeaksData_missing_mask_data(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        createPeakCollection: Callable[..., PeakCollection],
    ):
        """Verify ValueError when PeakCollection references missing mask data"""
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
            mask="non_existent_mask",  # names a diffractogram the workspace never reduced
        )

        with pytest.raises(ValueError, match=r".*not present in the workspace.*"):
            _Fit.validateWorkspaceAndPeaksData(ws, [peak0])

    def test_peakParametersForRange_intensity_error_roundtrip(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        createPeakCollection: Callable[..., PeakCollection],
    ):
        """σ_Intensity survives a write→read round-trip for PseudoVoigt; Gaussian does not crash.

        PseudoVoigt write path converts Intensity → Height (storing σ_h derived from σ_I, σ_Γ, σ_η).
        The read path must invert that exactly via algebraic cancellation of the correlated terms;
        naive addition would over-estimate σ_I by ≈ √2 for the σ_Γ / σ_η contributions.

        Gaussian stores Height as a native parameter, so Intensity is not in the native output of
        peakParametersForRange.  For Gaussian we verify that the call succeeds and that Height and
        Sigma (the actual native parameters) round-trip correctly.
        """
        ws = minimal_HidraWorkspace(with_instrument=False)

        subruns = ws._sample_logs.subruns.raw_copy()
        N_subrun = len(subruns)

        # --- PseudoVoigt: full σ_I round-trip ---
        peak_pv = createPeakCollection(
            peak_tag="Fe 311",
            peak_profile="PseudoVoigt",
            background_type="Linear",
            wavelength=1.5,
            projectfilename="/does/not/exist.h5",
            runnumber=99,
            N_subrun=N_subrun,
        )

        _, orig_errors_pv = peak_pv.get_effective_params()
        sigma_I_orig = orig_errors_pv["Intensity"].astype(np.float64)

        pp_pv = _PeakParameters.init_group(_Peaks.indexed([peak_pv]))
        native_values_pv, native_errors_pv = _PeakParameters.peakParametersForRange(pp_pv, 0, N_subrun)

        # For PseudoVoigt, Intensity is a native parameter — verify exact round-trip.
        # A factor-of-two over-count would produce errors ~√2× too large and fail here.
        #
        # rtol=1e-4, not 1e-6: this recovery subtracts comparable-magnitude terms derived from
        # a float32-quantized sigma_Height (the only lossy step; the algebraic inversion itself
        # is exact -- verified in float128 with no float32 anywhere, median rel. error 1e-19).
        # That subtraction amplifies the float32 rounding noise by a factor that depends on the
        # ratio between the three input uncertainties' relative sizes. createPeakCollection now
        # bounds every parameter's fractional uncertainty to [error_fraction_min, error_fraction_max]
        # (0.5%-5%), which caps that ratio at max/min=10 and, per a 2,000,000-draw Monte Carlo
        # against this exact formula, caps the resulting relative error at ~9.3e-6 -- rtol=1e-4
        # keeps roughly a 10x margin over that observed worst case. Do not tighten this back
        # toward 1e-6 without re-deriving the bound; it will flake again.
        np.testing.assert_allclose(
            native_errors_pv["Intensity"].astype(np.float64),
            sigma_I_orig,
            rtol=1e-4,
            err_msg="PseudoVoigt σ_Intensity round-trip failed: check peakParametersForRange",
        )

        # --- Gaussian: native Height / Sigma round-trip (Intensity is not a native parameter) ---
        peak_g = createPeakCollection(
            peak_tag="Fe 311",
            peak_profile="Gaussian",
            background_type="Linear",
            wavelength=1.5,
            projectfilename="/does/not/exist.h5",
            runnumber=99,
            N_subrun=N_subrun,
        )

        orig_values_g, orig_errors_g = peak_g.get_effective_params()

        pp_g = _PeakParameters.init_group(_Peaks.indexed([peak_g]))
        native_values_g, native_errors_g = _PeakParameters.peakParametersForRange(pp_g, 0, N_subrun)

        # Height is stored directly on the write path — must survive exactly.
        np.testing.assert_allclose(
            native_values_g["Height"].astype(np.float64),
            orig_values_g["Height"].astype(np.float64),
            rtol=1e-6,
            err_msg="Gaussian Height round-trip failed",
        )
        np.testing.assert_allclose(
            native_errors_g["Height"].astype(np.float64),
            orig_errors_g["Height"].astype(np.float64),
            rtol=1e-6,
            err_msg="Gaussian σ_Height round-trip failed",
        )


class TestFitTailAppend:
    """`init_group(..., data=...)` grows an existing group instead of building one.

    Per-family coverage, independent of the end-to-end round trip in
    `test_append.py`.
    """

    @staticmethod
    def _collection(createPeakCollection, tag, points, profile="Gaussian"):
        return createPeakCollection(
            peak_tag=tag,
            peak_profile=profile,
            background_type="Linear",
            wavelength=1.486,
            projectfilename="x.h5",
            runnumber=1,
            N_subrun=len(points),
            sub_runs=np.array(points),
        )

    def test_PeakParameters_append_grows_by_exactly_the_new_row_count(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        createPeakCollection: Callable[..., PeakCollection],
    ):
        # Arrange
        logs = minimal_HidraWorkspace(with_instrument=False, sub_runs=np.array([1, 2, 3, 4, 5]))._sample_logs
        first = self._collection(createPeakCollection, "Fe110", [1, 2, 3])
        pp = _PeakParameters.init_group(_Peaks.indexed([first], logs))
        before = np.asarray(pp["center"].nxdata).copy()
        second = self._collection(createPeakCollection, "Si111", [4, 5])

        # Act
        returned = _PeakParameters.init_group(_Peaks.indexed([second], logs), data=pp)

        # Assert
        assert returned is pp
        assert pp["center"].shape[0] == 5
        assert np.array_equal(np.asarray(pp["center"].nxdata)[:3], before)

    def test_BackgroundParameters_append_grows_by_exactly_the_new_row_count(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        createPeakCollection: Callable[..., PeakCollection],
    ):
        # Arrange
        logs = minimal_HidraWorkspace(with_instrument=False, sub_runs=np.array([1, 2, 3, 4, 5]))._sample_logs
        first = self._collection(createPeakCollection, "Fe110", [1, 2, 3])
        bp = _BackgroundParameters.init_group(_Peaks.indexed([first], logs))
        before = np.asarray(bp["A0"].nxdata).copy()
        second = self._collection(createPeakCollection, "Si111", [4, 5])

        # Act
        _BackgroundParameters.init_group(_Peaks.indexed([second], logs), data=bp)

        # Assert
        assert bp["A0"].shape[0] == 5
        assert np.array_equal(np.asarray(bp["A0"].nxdata)[:3], before)

    def test_the_three_position_aligned_groups_stay_the_same_length(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        createPeakCollection: Callable[..., PeakCollection],
    ):
        """Their alignment is the only reason row `n` of each describes the same peak."""
        # Arrange
        logs = minimal_HidraWorkspace(with_instrument=False, sub_runs=np.array([1, 2, 3, 4, 5]))._sample_logs
        first = _Peaks.indexed([self._collection(createPeakCollection, "Fe110", [1, 2, 3])], logs)
        peaks = _Peaks.init_group(first, logs)
        pp = _PeakParameters.init_group(first)
        bp = _BackgroundParameters.init_group(first)
        second = _Peaks.indexed([self._collection(createPeakCollection, "Si111", [4, 5])], logs)

        # Act
        _Peaks.init_group(second, logs, data=peaks)
        _PeakParameters.init_group(second, data=pp)
        _BackgroundParameters.init_group(second, data=bp)

        # Assert
        assert peaks["scan_point"].shape[0] == pp["center"].shape[0] == bp["A0"].shape[0] == 5

    def test_Diffractogram_append_grows_every_scan_point_dataset(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        createPeakCollection: Callable[..., PeakCollection],
    ):
        # Arrange
        first = minimal_HidraWorkspace(with_instrument=False, sub_runs=np.array([1, 2, 3]))
        second = minimal_HidraWorkspace(with_instrument=False, sub_runs=np.array([4, 5]))
        indexed = _Peaks.indexed([self._collection(createPeakCollection, "Fe110", [1, 2, 3])], first._sample_logs)
        dg = _Diffractogram.init_group([first], DEFAULT_TAG, indexed)
        before = np.asarray(dg[GROUP_NAME.DGRAM_DIFFRACTOGRAM].nxdata).copy()

        # Act
        returned = _Diffractogram.init_group([second], DEFAULT_TAG, indexed, data=dg)

        # Assert
        assert returned is dg
        assert dg["scan_point"].nxdata.tolist() == [1, 2, 3, 4, 5]
        for name in (
            GROUP_NAME.DGRAM_TWO_THETA_NAME,
            GROUP_NAME.DGRAM_DIFFRACTOGRAM,
            GROUP_NAME.DGRAM_DIFFRACTOGRAM_ERRORS,
        ):
            assert dg[name].shape[0] == 5, name
        assert np.array_equal(np.asarray(dg[GROUP_NAME.DGRAM_DIFFRACTOGRAM].nxdata)[:3], before)

    def test_Diffractogram_append_at_a_different_two_theta_width_raises(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        createPeakCollection: Callable[..., PeakCollection],
    ):
        """A silently misaligned spectrum is worse than a refused append."""
        # Arrange
        first = minimal_HidraWorkspace(with_instrument=False, sub_runs=np.array([1, 2, 3]), n_two_theta=20)
        wider = minimal_HidraWorkspace(with_instrument=False, sub_runs=np.array([4, 5]), n_two_theta=25)
        indexed = _Peaks.indexed([self._collection(createPeakCollection, "Fe110", [1, 2, 3])], first._sample_logs)
        dg = _Diffractogram.init_group([first], DEFAULT_TAG, indexed)

        # Act / Assert
        with pytest.raises(RuntimeError, match=r"trailing shape"):
            _Diffractogram.init_group([wider], DEFAULT_TAG, indexed, data=dg)


class TestReducedDataRowCountIsGuarded:
    """A workspace whose reduced data and sub-runs disagree must never reach a file.

    NXstress writes `scan_point` from the sub-runs and the diffractogram rows from
    `_2theta_matrix`. If those disagree the rows do not correspond to the scan
    points, and nothing says so -- the file is readable and wrong. With several
    inputs in one `NXentry` the damage is not even local: one short input shifts
    every later input onto the wrong rows.

    **The invariant is guarded in two places, and both are needed.**

    `HidraWorkspace.set_reduced_diffraction_data_set` guards its own end, but it is
    one of **four** writers of `_2theta_matrix`, and the other three bypass it:

    - `_load_reduced_diffraction_data`    (reading a HidraProjectFile)
    - `_append_reduced_diffraction_data`  (`append_hidra_project`)
    - `set_reduced_diffraction_data`      (row-wise; `reduction_manager`,
                                           `mantid_peakfit_calibration`)

    All four live in `workspaces.py`; nothing outside the class assigns the
    attribute, and `_fit.py` only reads it. `_inconsistent` below does assign it
    directly, which is how these tests reach the state the three bypass paths can
    produce -- but that is the test, not a fifth production path.

    `_Diffractogram._concatenated_diffraction` is therefore the second guard, at the
    write boundary every one of those paths funnels through. Removing either leaves
    a silent route to a misaligned file, which is why both are tested here.
    """

    @staticmethod
    def _inconsistent(ws, n_rows: int, n_two_theta: int = 20):
        """Force the mismatch past the setter, as the four other paths can."""
        ws._2theta_matrix = np.tile(np.linspace(60.0, 120.0, n_two_theta), (n_rows, 1))
        ones = np.ones((n_rows, n_two_theta))
        ws._diff_data_set = {None: ones}
        ws._var_data_set = {None: ones.copy()}

    def test_setter_rejects_a_row_count_that_contradicts_the_sub_runs(
        self, minimal_HidraWorkspace: Callable[..., HidraWorkspace]
    ):
        # Arrange
        ws = minimal_HidraWorkspace(with_instrument=False, sub_runs=np.array([1, 2, 3]))
        two_theta = np.tile(np.linspace(60.0, 120.0, 20), (5, 1))  # 5 rows, 3 sub-runs
        ones = np.ones((5, 20))

        # Act / Assert
        with pytest.raises(RuntimeError, match=r"5 row\(s\) but this workspace has 3 sub-run"):
            ws.set_reduced_diffraction_data_set(two_theta, {None: ones}, {None: ones})

    def test_setter_rejects_a_mask_array_that_disagrees_with_two_theta(
        self, minimal_HidraWorkspace: Callable[..., HidraWorkspace]
    ):
        # Arrange
        ws = minimal_HidraWorkspace(with_instrument=False, sub_runs=np.array([1, 2, 3]))
        two_theta = np.tile(np.linspace(60.0, 120.0, 20), (3, 1))

        # Act / Assert -- a mask sampled on a different number of channels
        with pytest.raises(RuntimeError, match=r"sampled on the same axes and must agree"):
            ws.set_reduced_diffraction_data_set(two_theta, {None: np.ones((3, 25))}, {None: np.ones((3, 25))})

    def test_write_boundary_rejects_a_mismatch_the_setter_never_saw(
        self, minimal_HidraWorkspace: Callable[..., HidraWorkspace]
    ):
        """The case that matters: the mismatch arrived by one of the three other paths."""
        # Arrange
        ws = minimal_HidraWorkspace(with_instrument=False, sub_runs=np.array([1, 2, 3]))
        self._inconsistent(ws, n_rows=5)  # bypasses the setter, as production paths do

        # Act / Assert
        with pytest.raises(RuntimeError, match=r"input workspace \[0\] has reduced data for 5 scan"):
            _Diffractogram._concatenated_diffraction([ws], DEFAULT_TAG)

    def test_the_offending_input_is_named_when_several_are_written(
        self, minimal_HidraWorkspace: Callable[..., HidraWorkspace]
    ):
        """With N inputs a short one shifts every later one, so the message must say which."""
        # Arrange
        good = minimal_HidraWorkspace(with_instrument=False, sub_runs=np.array([1, 2, 3]))
        bad = minimal_HidraWorkspace(with_instrument=False, sub_runs=np.array([4, 5, 6]))
        self._inconsistent(bad, n_rows=2)

        # Act / Assert
        with pytest.raises(RuntimeError, match=r"input workspace \[1\] has reduced data for 2 scan"):
            _Diffractogram._concatenated_diffraction([good, bad], DEFAULT_TAG)


class TestTextureWorkspaceRoundTrip:
    """A texture workspace reduces `eta_*` only, and could not be written at all.

    Every reduction uses the default detector mask; a texture reduction
    additionally intersects it with an eta region, so its `_diff_data_set` keys are
    `eta_<angle>` and there is **no** key for the default mask alone. Two things
    rejected that:

    - `nxstress_mask_names` injected `DEFAULT_TAG` into the *diffractogram*
      keyspace, so a bare `DIFFRACTOGRAM` was demanded for data never reduced; and
    - `validateWorkspaceAndPeaksData` checked `PeakCollection.mask` against the
      *mask* namespace, where `eta_0.0` is correctly absent -- it is a
      diffractogram key, not a mask.

    Nothing in PyRS writes texture through NXstress yet, which is why neither was
    caught; this pins the shape before something does.
    """

    @staticmethod
    def _texture_workspace(minimal_HidraWorkspace, sub_runs, keys):
        ws = minimal_HidraWorkspace(with_instrument=True, with_masks=True, sub_runs=sub_runs)
        two_theta = np.tile(np.linspace(60.0, 120.0, 20), (len(sub_runs), 1))
        ones = np.ones((len(sub_runs), 20))
        ws.set_reduced_diffraction_data_set(two_theta, {k: ones for k in keys}, {k: ones.copy() for k in keys})
        return ws

    def test_eta_only_reductions_write_and_read(self, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path):
        # Arrange
        sub_runs = np.array([1, 2, 3])
        keys = ("eta_-5.0", "eta_0.0", "eta_5.0")
        ws = self._texture_workspace(minimal_HidraWorkspace, sub_runs, keys)
        # `mask` is the diffractogram this collection was fitted on, set through the
        # fixture rather than by assigning `_mask`: the argument exists for this.
        peaks = minimal_PeakCollection(N_subrun=3, sub_runs=sub_runs, mask="eta_0.0")
        path = tmp_path / "texture.nxs"

        # Act
        with NXstress(path, "w") as nx:
            nx.write([ws], [[peaks]])
        with NXstress(path, "r") as nx:
            workspaces, peakss = nx.read()

        # Assert -- one group per eta key, and NO bare DIFFRACTOGRAM invented
        with h5py.File(path, "r") as f:
            groups = sorted(k for k in f["entry/FIT"] if k.startswith("DIFFRACTOGRAM"))
        assert groups == ["DIFFRACTOGRAM_eta_-5.0", "DIFFRACTOGRAM_eta_0.0", "DIFFRACTOGRAM_eta_5.0"]
        assert sorted(str(k) for k in workspaces[0].reduction_masks) == sorted(keys)
        assert peakss[0][0].mask == "eta_0.0"

    def test_a_peak_naming_a_diffractogram_that_does_not_exist_is_refused(
        self, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path
    ):
        """The check that replaced the mask-namespace one, doing the real work."""
        # Arrange
        sub_runs = np.array([1, 2, 3])
        ws = self._texture_workspace(minimal_HidraWorkspace, sub_runs, ("eta_0.0",))
        peaks = minimal_PeakCollection(N_subrun=3, sub_runs=sub_runs, mask="eta_99.0")  # never reduced

        # Act / Assert
        with pytest.raises(ValueError, match=r"Reduced data for diffractogram 'eta_99.0'"):
            with NXstress(tmp_path / "bad.nxs", "w") as nx:
                nx.write([ws], [[peaks]])
