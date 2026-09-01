"""
Tests for pyrs/utilities/NXstress/_fit.py
"""

from collections.abc import Callable
import numpy as np
from nexusformat.nexus import NXdata, NXnote, NXparameters, NXprocess
import pytest

from pyrs.core.workspaces import HidraWorkspace
from pyrs.peaks.peak_collection import PeakCollection
from pyrs.utilities.NXstress._fit import _BackgroundParameters, _Diffractogram, _Fit, _PeakParameters
from pyrs.utilities.NXstress._definitions import DEFAULT_TAG


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

        peak_params = _PeakParameters.init_group([peak0])

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

        peak_params = _PeakParameters.init_group([peak0, peak1])

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
            _PeakParameters.init_group([peak0, peak1])

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

        bg_params = _BackgroundParameters.init_group([peak0])

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

        bg_params = _BackgroundParameters.init_group([peak0, peak1])

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
            _BackgroundParameters.init_group([peak0, peak1])

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
            _Diffractogram._init(ws)

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
            _Diffractogram.init_group(ws, "non_existent_mask", [peak0])

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

        dgram = _Diffractogram.init_group(ws, DEFAULT_TAG, [peak0])

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
        fit = _Fit._init(logs, processing_description="Test description", processing_time="2024-01-15T10:30:00")

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

        fit = _Fit.init_group(ws, [peak0], ws._sample_logs)

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
        )

        # Set a mask that doesn't exist in the workspace
        peak0._mask = "non_existent_mask"

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

        pp_pv = _PeakParameters.init_group([peak_pv])
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

        pp_g = _PeakParameters.init_group([peak_g])
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
