"""
pyrs/utilities/NXstress/_fit.py

Private service class for NeXus NXstress-compatible I/O.
This class provides I/O for the `fit` `NXprocess` subgroup:
  this subgroup includes the reduced output data as a 'diffraction_data' `NXdata` group.
"""

import logging

from datetime import datetime
from nexusformat.nexus import NXdata, NXfield, NXnote, NXparameters, NXprocess
import numpy as np
from typing import Tuple

from pyrs.peaks.peak_collection import PeakCollection
from pyrs.core.peak_profile_utility import BackgroundFunction
from pyrs.core.workspaces import HidraWorkspace
from pyrs.dataobjects.sample_logs import SampleLogs
from pyrs.utilities.pydantic_transition import validate_call_

from pyrs.utilities.convertdatatypes import to_text

from ._definitions import (
    FIELD_DTYPE,
    CHUNK_SHAPE,
    GROUP_NAME,
    group_naming_scheme,
    growable,
    tail_append,
    diffractogram_detector_mask,
    nxstress_diffractogram_keys,
    nxstress_mask_names,
    row_aligned_fields,
    UNDEFINED_PEAK_TAG,
    workspace_mask_key,
)
from ._peaks import IndexedPeaks

"""
REQUIRED PARAMETERS FOR NXstress:
---------------------------------

├─ fit                                    (NXprocess, group)
│   ├─ date                               (dataset: ISO8601 string)
│   ├─ program                            (dataset: string)
│   ├─ description                         (NXnote, group)
│   ├─ peakparameters                      (NXparameters, group)
│   └─ diffractogram                       (NXdata, group)
│        ├─ diffractogram                  (dataset)
│        ├─ diffractogram_errors           (dataset)
│        ├─ daxis/xaxis                    (dataset)
│        ├─ @axes                          (attribute: string)
│        └─ @signal                        (attribute: string)
"""


class _PeakParameters:
    @classmethod
    def _init(cls, peakss: list[PeakCollection]) -> NXparameters:
        # required 'peak_parameters' subgroup
        pp = NXparameters()
        peak_profile = str(peakss[0].peak_profile).lower() if peakss else UNDEFINED_PEAK_TAG

        # To be compliant with `NXstress` schema:
        #   this cannot be tiled: all `PeakCollection` must share the same `peak_profile`.
        pp["title"] = NXfield(peak_profile, dtype=FIELD_DTYPE.STRING.value)

        pp["center"] = NXfield(
            np.empty((0,), dtype=np.float64), maxshape=(None,), chunks=CHUNK_SHAPE(1), units="degree"
        )
        pp["center_errors"] = NXfield(
            np.empty((0,), dtype=np.float64), maxshape=(None,), chunks=CHUNK_SHAPE(1), units="degree"
        )
        pp["height"] = NXfield(
            np.empty((0,), dtype=np.float64), maxshape=(None,), chunks=CHUNK_SHAPE(1), units="counts"
        )
        pp["height_errors"] = NXfield(
            np.empty((0,), dtype=np.float64), maxshape=(None,), chunks=CHUNK_SHAPE(1), units="counts"
        )
        pp["fwhm"] = NXfield(np.empty((0,), dtype=np.float64), maxshape=(None,), chunks=CHUNK_SHAPE(1), units="degree")
        pp["fwhm_errors"] = NXfield(
            np.empty((0,), dtype=np.float64), maxshape=(None,), chunks=CHUNK_SHAPE(1), units="degree"
        )

        # Voigt or Pseudo-Voigt: Lorentzian fraction
        pp["form_factor"] = NXfield(
            np.empty((0,), dtype=np.float64), maxshape=(None,), chunks=CHUNK_SHAPE(1), units="1"
        )
        pp["form_factor_errors"] = NXfield(
            np.empty((0,), dtype=np.float64), maxshape=(None,), chunks=CHUNK_SHAPE(1), units="1"
        )

        return pp

    @classmethod
    def init_group(cls, indexed: list[IndexedPeaks], data: NXparameters | None = None) -> NXparameters:
        # required 'peak_parameters' subgroup
        #
        # Sorted with the same key as the PEAKS group and `_BackgroundParameters`:
        # these three are position-aligned, and that alignment is the only thing
        # making row `n` of each describe the same peak.
        #
        # `data` is an existing on-disk group to tail-append to (spec 04c). All three
        # position-aligned groups grow by the same row count in the same sorted order,
        # which is what preserves the alignment across an append.
        pp = data if data is not None else cls._init([item.collection for item in indexed])

        for item in sorted(indexed, key=IndexedPeaks.sort_key):
            cls._append_peak(pp, item.collection)

        return pp

    @classmethod
    @validate_call_
    def _append_peak(cls, pp: NXparameters, peaks: PeakCollection) -> NXparameters:
        # Append the peak parameters from a single `PeakCollection` instance.

        # Verify the `PeakCollection` peak-profile type.
        peak_profile = str(peaks.peak_profile).lower()
        if pp["title"] == UNDEFINED_PEAK_TAG:
            pp["title"].replace(peak_profile)
        elif peak_profile != pp["title"]:
            raise ValueError(
                f"All `PeakCollection` must share the same peak profile ''{pp['title']}'', not ''{peak_profile}''."
            )

        # Use _effective_ peak parameters here: all peaks will then have the same number of parameter,
        #   and all parameter values will be in the expected column.
        # We have one new parameter value for each of 'N_scan' subruns.

        N_scan = len(peaks.sub_runs)
        cur_rows = pp["center"].shape[0]
        new_rows = cur_rows + N_scan

        ## In the following, make sure to include _only_ the peak-function parameters.
        params_value, params_error = peaks.get_effective_params()

        pp["center"].resize((new_rows,))
        pp["center_errors"].resize((new_rows,))
        pp["height"].resize((new_rows,))
        pp["height_errors"].resize((new_rows,))
        pp["fwhm"].resize((new_rows,))
        pp["fwhm_errors"].resize((new_rows,))
        pp["form_factor"].resize((new_rows,))
        pp["form_factor_errors"].resize((new_rows,))

        pp["center"][cur_rows:] = params_value["Center"].astype(np.float64)
        pp["center_errors"][cur_rows:] = params_error["Center"].astype(np.float64)
        pp["height"][cur_rows:] = params_value["Height"].astype(np.float64)
        pp["height_errors"][cur_rows:] = params_error["Height"].astype(np.float64)
        pp["fwhm"][cur_rows:] = params_value["FWHM"].astype(np.float64)
        pp["fwhm_errors"][cur_rows:] = params_error["FWHM"].astype(np.float64)

        # Voigt or Pseudo-Voigt: Lorentzian fraction
        pp["form_factor"][cur_rows:] = (1.0 - params_value["Mixing"]).astype(np.float64)
        pp["form_factor_errors"][cur_rows:] = params_error["Mixing"].astype(np.float64)

        return pp

    @classmethod
    def peakParametersForRange(cls, pp, start: int, end: int) -> tuple:
        """Extract peak parameters for a specific range and convert to native parameters.

        Reads effective parameters from the NXparameters group, slices to the specified range,
        and converts to native parameters using the appropriate converter.

        CRITICAL: form_factor is stored as (1 - Mixing), so we invert: Mixing = 1 - form_factor

        Parameters
        ----------
        pp : NXparameters
            Peak parameters group
        start : int
            Starting index (inclusive)
        end : int
            Ending index (exclusive)

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (native_values, native_errors) structured arrays
        """
        from pyrs.core.peak_profile_utility import PeakShape, get_parameter_dtype, get_effective_parameters_converter

        # Get peak profile type and converter (needed for both Intensity reconstruction and native conversion)
        peak_shape = PeakShape.getShape(pp["title"].nxdata)
        converter = get_effective_parameters_converter(peak_shape)

        # Build effective parameter structured arrays
        N = end - start
        eff_values = np.zeros(N, dtype=get_parameter_dtype(effective=True))
        eff_errors = np.zeros(N, dtype=get_parameter_dtype(effective=True))

        # Slice datasets and populate effective arrays
        eff_values["Center"] = pp["center"].nxdata[start:end]
        eff_values["Height"] = pp["height"].nxdata[start:end]
        eff_values["FWHM"] = pp["fwhm"].nxdata[start:end]

        # CRITICAL: Invert form_factor to Mixing
        eff_values["Mixing"] = 1.0 - pp["form_factor"].nxdata[start:end]

        eff_errors["Center"] = pp["center_errors"].nxdata[start:end]
        eff_errors["Height"] = pp["height_errors"].nxdata[start:end]
        eff_errors["FWHM"] = pp["fwhm_errors"].nxdata[start:end]
        eff_errors["Mixing"] = pp["form_factor_errors"].nxdata[start:end]

        # Intensity is not stored -- reconstruct it and its error from the stored (h, fwhm, eta).
        # The reconstruction is shape-dependent because each peak profile has a different
        # native parameterisation and a different write-path error propagation.
        h = eff_values["Height"]
        fwhm = eff_values["FWHM"]
        eta = eff_values["Mixing"]
        s_h = eff_errors["Height"]
        s_fwhm = eff_errors["FWHM"]
        s_eta = eff_errors["Mixing"]

        if peak_shape == PeakShape.PSEUDOVOIGT:
            # PseudoVoigt: $I = h \pi \Gamma / (2 (1 + F \eta))$ where $F = \sqrt{\pi \ln 2} - 1$
            eff_values["Intensity"] = converter.cal_intensity(h, fwhm, eta)

            # Error propagation: exact inversion of the write-path cal_height_error formula.
            # On the write path s_h was derived from (s_I, s_fwhm, s_eta) treated as independent:
            #   $s_h^2 = (\partial h/\partial I)^2 s_I^2
            #          + (\partial h/\partial \Gamma)^2 s_\Gamma^2
            #          + (\partial h/\partial \eta)^2 s_\eta^2$
            # Solving for $s_I^2$:
            #   $s_I^2 = (\partial I/\partial h)^2 s_h^2
            #          - (\partial I/\partial \Gamma)^2 s_\Gamma^2
            #          - (\partial I/\partial \eta)^2 s_\eta^2$
            # The subtraction removes the correlation terms that would otherwise be double-counted.
            # The guard below handles the case where floating-point noise makes $s_I^2$ slightly negative.
            _F = np.sqrt(np.pi * np.log(2)) - 1.0
            I_val = eff_values["Intensity"]
            dI_dh = np.where(h != 0, I_val / h, 0.0)
            dI_dfwhm = np.where(fwhm != 0, I_val / fwhm, 0.0)
            dI_deta = -I_val * _F / (1.0 + _F * eta)
            sigma_I_sq = (dI_dh * s_h) ** 2 - (dI_dfwhm * s_fwhm) ** 2 - (dI_deta * s_eta) ** 2
            if np.any(sigma_I_sq < 0) or np.any(~np.isfinite(sigma_I_sq)):
                logging.getLogger(__name__).warning(
                    "peakParametersForRange: first-order s_Intensity approximation failed "
                    "(negative or non-finite variance); setting s_Intensity = NaN."
                )
                eff_errors["Intensity"] = np.nan
            else:
                eff_errors["Intensity"] = np.sqrt(sigma_I_sq)

        else:
            # Gaussian (and any other shape): Height is a native parameter, so s_h comes
            # directly from the fit and is independent of s_fwhm -- no double-counting.
            # $I = \sqrt{2\pi} \cdot h \cdot \sigma$ where $\sigma = \Gamma / (2\sqrt{2 \ln 2})$
            sigma_g = converter.cal_sigma(fwhm)
            s_sigma_g = converter.cal_sigma(s_fwhm)  # linear, same scale factor
            I_val = converter.cal_intensity(h, sigma_g)
            eff_values["Intensity"] = I_val
            eff_errors["Intensity"] = converter.cal_intensity_error(I_val, h, s_h, sigma_g, s_sigma_g)

        # A0, A1, A2 will be populated from backgroundParametersForRange
        # Initialize to 0.0 as they are part of the effective parameter dtype
        eff_values["A0"] = 0.0
        eff_values["A1"] = 0.0
        eff_values["A2"] = 0.0
        eff_errors["A0"] = 0.0
        eff_errors["A1"] = 0.0
        eff_errors["A2"] = 0.0

        # Convert to native parameters
        native_values, native_errors = converter.calculate_native_parameters(eff_values, eff_errors)

        return native_values, native_errors


class _BackgroundParameters:
    @classmethod
    def _init(cls, peakss: list[PeakCollection]) -> NXparameters:
        # required 'background_parameters' subgroup
        bp = NXparameters()

        # To be compliant with `NXstress` schema:
        #   this cannot be tiled: all `PeakCollection` must share the same `background_type`.
        background_function = (
            str(BackgroundFunction.getFunction(peakss[0].background_type)).lower() if peakss else UNDEFINED_PEAK_TAG
        )
        bp["title"] = NXfield(background_function, dtype=FIELD_DTYPE.STRING.value)

        bp["A0"] = NXfield(np.empty((0,), dtype=np.float64), maxshape=(None,), chunks=CHUNK_SHAPE(1), units="counts")
        bp["A0_errors"] = NXfield(
            np.empty((0,), dtype=np.float64), maxshape=(None,), chunks=CHUNK_SHAPE(1), units="counts"
        )

        bp["A1"] = NXfield(np.empty((0,), dtype=np.float64), maxshape=(None,), chunks=CHUNK_SHAPE(1), units="counts")
        bp["A1_errors"] = NXfield(
            np.empty((0,), dtype=np.float64), maxshape=(None,), chunks=CHUNK_SHAPE(1), units="counts"
        )

        bp["A2"] = NXfield(np.empty((0,), dtype=np.float64), maxshape=(None,), chunks=CHUNK_SHAPE(1), units="counts")
        bp["A2_errors"] = NXfield(
            np.empty((0,), dtype=np.float64), maxshape=(None,), chunks=CHUNK_SHAPE(1), units="counts"
        )

        return bp

    @classmethod
    def init_group(cls, indexed: list[IndexedPeaks], data: NXparameters | None = None) -> NXparameters:
        # required 'background_parameters' subgroup
        #   -- position-aligned with PEAKS and `_PeakParameters`; see that class's `init_group`.
        bp = data if data is not None else cls._init([item.collection for item in indexed])

        for item in sorted(indexed, key=IndexedPeaks.sort_key):
            cls._append_peak(bp, item.collection)

        return bp

    @classmethod
    @validate_call_
    def _append_peak(cls, bp: NXparameters, peaks: PeakCollection) -> NXparameters:
        # Append the background parameters from a single `PeakCollection` instance.

        # Verify the `PeakCollection` background type.
        background_title = str(BackgroundFunction.getFunction(peaks.background_type)).lower()
        if bp["title"] == UNDEFINED_PEAK_TAG:
            bp["title"].replace(background_title)
        elif background_title != bp["title"]:
            raise ValueError(
                f"All `PeakCollection` must share the same background type ''{bp['title']}'', not ''{background_title}''."
            )

        ## In the following, make sure to include _only_ the background parameters.
        params_value, params_error = peaks.get_effective_params()

        N_scan = len(peaks.sub_runs)
        cur_rows = bp["A0"].shape[0]
        new_rows = cur_rows + N_scan

        bp["A0"].resize((new_rows,))
        bp["A0_errors"].resize((new_rows,))
        bp["A1"].resize((new_rows,))
        bp["A1_errors"].resize((new_rows,))
        bp["A2"].resize((new_rows,))
        bp["A2_errors"].resize((new_rows,))

        bp["A0"][cur_rows:,] = params_value["A0"].astype(np.float64)
        bp["A0_errors"][cur_rows:,] = params_error["A0"].astype(np.float64)
        bp["A1"][cur_rows:,] = params_value["A1"].astype(np.float64)
        bp["A1_errors"][cur_rows:,] = params_error["A1"].astype(np.float64)
        bp["A2"][cur_rows:,] = params_value["A2"].astype(np.float64)
        bp["A2_errors"][cur_rows:,] = params_error["A2"].astype(np.float64)

        return bp

    @classmethod
    def backgroundParametersForRange(cls, bp, start: int, end: int) -> tuple:
        """Extract background parameters for a specific range.

        Reads background coefficients from the NXparameters group and slices to the specified range.

        Parameters
        ----------
        bp : NXparameters
            Background parameters group
        start : int
            Starting index (inclusive)
        end : int
            Ending index (exclusive)

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (eff_bg_values, eff_bg_errors) structured arrays with A0, A1, A2 fields
        """
        from pyrs.core.peak_profile_utility import get_parameter_dtype

        # Build effective background parameter arrays
        N = end - start
        eff_bg_values = np.zeros(N, dtype=get_parameter_dtype(effective=True))
        eff_bg_errors = np.zeros(N, dtype=get_parameter_dtype(effective=True))

        # Slice datasets and populate arrays
        eff_bg_values["A0"] = bp["A0"].nxdata[start:end]
        eff_bg_values["A1"] = bp["A1"].nxdata[start:end]
        eff_bg_values["A2"] = bp["A2"].nxdata[start:end]

        eff_bg_errors["A0"] = bp["A0_errors"].nxdata[start:end]
        eff_bg_errors["A1"] = bp["A1_errors"].nxdata[start:end]
        eff_bg_errors["A2"] = bp["A2_errors"].nxdata[start:end]

        return eff_bg_values, eff_bg_errors


class _Diffractogram:
    @classmethod
    def _get_diffraction_data(cls, ws: HidraWorkspace, mask_name: str) -> Tuple[np.ndarray, np.ndarray]:
        # Workaround for PyRS codebase use of `None` as the default key.
        data_key = cls._diffraction_data_key(mask_name)
        if data_key not in ws._diff_data_set:
            raise RuntimeError(
                f"NXstress._fit._Diffractogram: usage error: diffraction data '{data_key}' is not present in the workspace"
            )
        if data_key not in ws._var_data_set:
            raise RuntimeError(
                f"NXstress._fit._Diffractogram: variance for diffraction data '{mask_name}' is not present in the workspace:\n"
                "  how was this workspace initialized?"
            )
        return ws._diff_data_set[data_key], ws._var_data_set[data_key]

    @classmethod
    def _diffraction_data_key(cls, mask_name: str) -> str | None:
        # See `_definitions.workspace_mask_key` for why this mapping exists.
        return workspace_mask_key(mask_name)

    @classmethod
    def _init(cls, wss: list[HidraWorkspace]) -> NXdata:
        for n, ws in enumerate(wss):
            if ws._2theta_matrix is None:
                raise RuntimeError(
                    f"Usage error: cannot write NXstress file: input workspace [{n}] doesn't include any reduced data."
                )
        dg = NXdata()
        return dg

    @classmethod
    def _concatenated_diffraction(cls, wss: list[HidraWorkspace], mask_name: str) -> tuple:
        """Reduced data for one mask across every input, concatenated in workspace order.

        An input that has no reduced data for this mask still occupies its rows
        of the scan-point axis -- filled with `NaN`, rather than being dropped
        and shifting every later input onto the wrong rows.

        A mask *no* input reduced is still an error, as it was before: the
        NaN fill exists to align inputs that genuinely differ, not to turn a
        caller's bad mask name into a diffractogram of nothing.
        """
        if not any(cls._diffraction_data_key(mask_name) in ws._diff_data_set for ws in wss):
            # Raised from here rather than from `_get_diffraction_data` below,
            # which is never reached for a mask absent everywhere.
            where = "the workspace" if len(wss) == 1 else "any input workspace"
            raise RuntimeError(
                f"NXstress._fit._Diffractogram: usage error: diffraction data "
                f"'{cls._diffraction_data_key(mask_name)}' is not present in {where}"
            )

        two_theta, data, errors = [], [], []
        for n, ws in enumerate(wss):
            n_scan = len(ws.get_sub_runs())
            ws_two_theta = np.asarray(ws._2theta_matrix)

            # The write boundary, and the only chokepoint every path funnels through.
            # `HidraWorkspace.set_reduced_diffraction_data_set` guards the same
            # invariant at its own end, but it is one of four writers of
            # `_2theta_matrix` and the other three bypass it:
            # `_load_reduced_diffraction_data`, `_append_reduced_diffraction_data`
            # and the row-wise `set_reduced_diffraction_data` (called by
            # `reduction_manager` and `mantid_peakfit_calibration`). Unchecked, the
            # rows written here do not correspond to the `scan_point` axis, which is
            # built from the sub-runs; and with several inputs the damage is not
            # local, because one short input shifts every later one.
            if ws_two_theta.shape[0] != n_scan:
                raise RuntimeError(
                    f"NXstress._fit: input workspace [{n}] has reduced data for "
                    f"{ws_two_theta.shape[0]} scan point(s) but {n_scan} sub-run(s).\n"
                    "  The first axis of the reduced data is the scan-point axis; a mismatch would "
                    "write a diffractogram whose rows do not correspond to `scan_point`."
                )

            two_theta.append(ws_two_theta)
            if cls._diffraction_data_key(mask_name) in ws._diff_data_set:
                ws_data, ws_errors = cls._get_diffraction_data(ws, mask_name)
                data.append(np.asarray(ws_data))
                errors.append(np.asarray(ws_errors))
            else:
                n_two_theta = ws_two_theta.shape[-1]
                data.append(np.full((n_scan, n_two_theta), np.nan))
                errors.append(np.full((n_scan, n_two_theta), np.nan))

        widths = {arr.shape[-1] for arr in two_theta}
        if len(widths) > 1:
            raise RuntimeError(
                "NXstress._fit: input workspaces have different numbers of two-theta channels "
                f"({sorted(widths)}); they cannot share one diffractogram array."
            )
        return np.concatenate(two_theta), np.concatenate(data), np.concatenate(errors)

    @classmethod
    @validate_call_
    def init_group(
        cls, wss: list[HidraWorkspace], maskName: str, indexed: list[IndexedPeaks], data: NXdata | None = None
    ) -> NXdata:
        # required DIFFRACTOGRAM (NXdata) subgroup:

        if data is not None:
            # Append: grow the four scan-point-aligned datasets in lockstep. The
            # attributes and the `fit`/`fit_errors` placeholders are entry-wide and
            # already correct, so they are left alone.
            two_theta, values, errors = cls._concatenated_diffraction(wss, maskName)
            tail_append(data["scan_point"], np.concatenate([ws.get_sub_runs().raw_copy() for ws in wss]))
            tail_append(data[GROUP_NAME.DGRAM_TWO_THETA_NAME], two_theta)
            tail_append(data[GROUP_NAME.DGRAM_DIFFRACTOGRAM], values)
            tail_append(data[GROUP_NAME.DGRAM_DIFFRACTOGRAM_ERRORS], errors)
            return data

        dg = cls._init(wss)
        dg.attrs["signal"] = GROUP_NAME.DGRAM_DIFFRACTOGRAM
        dg.attrs["auxiliary_signals"] = [
            GROUP_NAME.DGRAM_DIFFRACTOGRAM_ERRORS,
            GROUP_NAME.DGRAM_FIT,
            GROUP_NAME.DGRAM_FIT_ERRORS,
        ]
        dg.attrs["axes"] = ["scan_point", "."]  # do _not_ specify a 2-D theta in 'axes'
        dg.attrs["two_theta_indices"] = [0, 1]  # two-theta has shape (<N scan points>, <N 2-theta, per scan-point>)
        dg["scan_point"] = NXfield(np.concatenate([ws.get_sub_runs().raw_copy() for ws in wss]), **growable(1))
        dg["scan_point"].attrs["units"] = ""

        two_theta, data, errors = cls._concatenated_diffraction(wss, maskName)
        dg[GROUP_NAME.DGRAM_TWO_THETA_NAME] = NXfield(  # *** DEBUG *** validator bug
            two_theta, units="degree", **growable(2)
        )

        dg[GROUP_NAME.DGRAM_DIFFRACTOGRAM] = NXfield(
            data, dtype=FIELD_DTYPE.FLOAT_DATA.value, interpretation="spectrum", units="counts", **growable(2)
        )

        dg[GROUP_NAME.DGRAM_DIFFRACTOGRAM_ERRORS] = NXfield(
            errors, dtype=FIELD_DTYPE.FLOAT_DATA.value, units="counts", **growable(2)
        )

        ##
        ## ENTRY/FIT/DIFFRACTOGRAM/fit, fit_errors: required datasets under `NXstress`:
        ##   these should contain the spectrum reconstructed from the fitted model.
        ##   For the moment, these will be initialized to NaN.
        ##
        dg[GROUP_NAME.DGRAM_FIT] = NXfield(
            np.empty((0, 0), dtype=np.float64), maxshape=(None, None), chunks=CHUNK_SHAPE(2), fillvalue=np.nan
        )
        dg[GROUP_NAME.DGRAM_FIT].attrs["interpretation"] = "spectrum"
        dg[GROUP_NAME.DGRAM_FIT].attrs["units"] = "counts"
        dg[GROUP_NAME.DGRAM_FIT_ERRORS] = NXfield(
            np.empty((0, 0), dtype=np.float64), maxshape=(None, None), chunks=CHUNK_SHAPE(2), fillvalue=np.nan
        )
        dg[GROUP_NAME.DGRAM_FIT_ERRORS].attrs["units"] = "counts"

        return dg

    @classmethod
    def appendableDatasets(cls, data, prefix: str) -> dict:
        """Every dataset in one DIFFRACTOGRAM group that an append grows.

        All four of them: `scan_point`, the two-theta axis, the diffractogram and
        its errors are scan-point aligned, so the whole group grows together.

        Args:
            data: One existing `DIFFRACTOGRAM_<key>` group.
            prefix: Path prefix for the returned keys.

        Returns:
            Path -> `NXfield`, for `NXstress._validateAppendableShapes`.
        """
        return row_aligned_fields(data, prefix)

    @classmethod
    def validateAppend(cls, wss: list[HidraWorkspace], maskName: str, data) -> None:
        """Check incoming diffraction against an existing group, without mutating it.

        `diffractogram` and its two-theta axis are two-dimensional: scan point by
        two-theta bin. Only the first axis grows. A batch reduced onto a
        different two-theta grid -- a routine difference between two reduction
        passes -- cannot be appended at all, and must be refused **before** any
        other group has grown, because a refusal discovered during mutation
        leaves the entry with 27 datasets at the new length and these three at
        the old one. That entry reads back without error.

        Args:
            wss: Workspaces being appended.
            maskName: Which `DIFFRACTOGRAM` group this is.
            data: The target entry's existing group for that mask.

        Raises:
            RuntimeError: If the two-theta bin count differs.
        """
        two_theta, _, _ = cls._concatenated_diffraction(wss, maskName)
        incoming = int(np.asarray(two_theta).shape[1])
        existing = int(data[GROUP_NAME.DGRAM_DIFFRACTOGRAM].shape[1])
        if incoming != existing:
            raise RuntimeError(
                f"NXstress._fit: cannot append -- the incoming reduced diffraction for mask "
                f"'{maskName}' has {incoming} two-theta bin(s), the target entry {existing}.\n"
                "  Only the scan-point axis grows on an append; the two-theta axis is fixed when "
                "the entry is written. Re-reduce onto the entry's binning, or write a new entry."
            )

    @classmethod
    @validate_call_
    def diffractogramFromNexus(cls, dg, rows=None):
        """Read diffractogram data from NXdata group.

        Parameters
        ----------
        dg : NXdata
            The DIFFRACTOGRAM NXdata group from the HDF5 file
        rows : np.ndarray, optional
            Boolean mask selecting one input workspace's rows of the
            concatenated scan-point axis.

        Returns
        -------
        tuple
            (scan_points, two_theta, diffractogram, diffractogram_errors)

        Note
        ----
        The write side stores variance in 'diffractogram_errors', so this is
        returned directly without conversion.
        """

        def selected(values):
            return values[rows] if rows is not None else values

        # Read scan_point array
        scan_points = selected(dg["scan_point"].nxdata)

        # Read two_theta array (using the correct field name from GROUP_NAME)
        two_theta = selected(dg[GROUP_NAME.DGRAM_TWO_THETA_NAME].nxdata)

        # Read diffractogram and diffractogram_errors (which stores variance)
        diffractogram = selected(dg[GROUP_NAME.DGRAM_DIFFRACTOGRAM].nxdata)
        diffractogram_errors = selected(dg[GROUP_NAME.DGRAM_DIFFRACTOGRAM_ERRORS].nxdata)

        return scan_points, two_theta, diffractogram, diffractogram_errors


class _Fit:
    ########################################
    # ALL methods must be `classmethod`.  ##
    ########################################

    ##
    ## Notes:
    ## -- Under 'NXstress', there can be multiple FIT (NXprocess) groups in the NXentry, but the results from only
    ##    one of these should be promoted to the canonical fit results in the PEAKS (NXreflections) group.
    ## -- FIT (NXprocess) contains the as-fit peak and background parameters, including any information associated
    ##    with the fitting process.  In this section, any appropriate coordinate system may be used.
    ## -- Not yet in PyRS: FIT/DIFFRACTOGRAM/fit, fit_errors: these datasets should contain the reconstructed spectrum
    #     from the fitted model.  We don't seem to have methods to do this yet, so these are initialized to NaN.
    ## -- The canonical fit results in PEAKS (NXreflections) should contain the final results, converted to the final
    ##    coordinate system (e.g. usually `d-spacing`).
    ##
    @classmethod
    @validate_call_
    def _init(cls, logss: list[SampleLogs], *, processing_description: str, processing_time) -> NXprocess:
        # Initialize the 'FIT' (NXprocess) group:

        fit = NXprocess()

        input_ = NXparameters()
        input_["description"] = "Peak fits and reduced diffractogram data"
        fit[GROUP_NAME.INPUT] = input_

        # Required information fields:
        fit["date"] = NXfield(processing_time)
        fit["program"] = NXfield("PyRS")
        # The NXstress schema allows one `raw_data_file` per FIT group, so a
        # multi-workspace entry records the first input's. Nothing is lost:
        # `Filename` is a per-scan-point log and is not one of
        # `_Sample.NXstress_logs`, so every input's filenames are retained in
        # full under SAMPLE_DESCRIPTION/logs.
        fit["raw_data_file"] = NXfield(to_text(logss[0]["Filename"][0]))

        note = NXnote(
            type="text/plain",
            description="Processing description",
            # author='',
            # date='',
            data=processing_description,
        )
        fit[GROUP_NAME.DESCRIPTION] = note

        return fit

    @classmethod
    @validate_call_
    def init_group(
        cls,
        wss: list[HidraWorkspace],
        indexed: list[IndexedPeaks],
        logss: list[SampleLogs],
        processing_description: str = "",
        processing_time: str | None = None,
        data: NXprocess | None = None,
    ):
        # Initialize a new 'FIT' (NXprocess) group:
        #   (see `_definitions.group_naming_scheme`).

        ## Under `NXstress`: `FIT` (NXprocess) groups contain peak and background-fit results, including any
        ##    information relevant to the fitting process used.
        if data is not None:
            # The NXnote description and the process timestamp record the FIRST write;
            # an append grows the data, it does not restate when the entry was made.
            fit = data
            _PeakParameters.init_group(indexed, data=fit[GROUP_NAME.PEAK_PARAMETERS])
            _BackgroundParameters.init_group(indexed, data=fit[GROUP_NAME.BACKGROUND_PARAMETERS])
        else:
            fit = cls._init(
                logss,
                processing_description=processing_description,
                processing_time=processing_time if bool(processing_time) else datetime.now().astimezone().isoformat(),
            )
            fit[GROUP_NAME.PEAK_PARAMETERS] = _PeakParameters.init_group(indexed)
            fit[GROUP_NAME.BACKGROUND_PARAMETERS] = _BackgroundParameters.init_group(indexed)

        # Add one DIFFRACTOGRAM group for each reduced diffraction dataset present in the workspaces.
        # `nxstress_mask_names` is the single definition of the default-mask
        # correspondence (`None` in the workspace, `DEFAULT_TAG` in the file), shared
        # with `_Masks.mask_keys` and with the read side.
        #
        # The union across inputs, not just the first workspace's: each group
        # spans the whole concatenated scan-point axis, so a mask any input
        # reduced needs a group even if the others did not.
        # The DIFFRACTOGRAM keyspace, which is NOT the mask namespace: it carries
        # only the reductions actually performed, and so does not force
        # `DEFAULT_TAG`. A texture workspace reduces `eta_*` only, and inventing a
        # bare `DIFFRACTOGRAM` for it would ask `_concatenated_diffraction` for
        # data that was never reduced.
        #
        # The union across inputs, not just the first workspace's: each group
        # spans the whole concatenated scan-point axis, so a key any input reduced
        # needs a group even if the others did not.
        mask_keys = set()
        for ws in wss:
            mask_keys |= nxstress_diffractogram_keys(ws._diff_data_set.keys())

        # Every key must reference a detector mask that exists. Always true in
        # practice -- the default is always written -- so this catches a malformed
        # key rather than a missing mask.
        known_masks = set()
        for ws in wss:
            known_masks |= nxstress_mask_names(ws._mask_dict.keys())
        for key in sorted(mask_keys):
            diffractogram_detector_mask(key, known_masks)

        for mask in sorted(mask_keys):
            dgram_name = group_naming_scheme(GROUP_NAME.DIFFRACTOGRAM, mask)
            if data is not None:
                # An append must find the group already there: a mask the target entry
                # has no DIFFRACTOGRAM for would need a new group with no history for
                # the scan points already on disk, which is schema restructuring rather
                # than a tail-append. `NXstress._classifyAppend` rejects that case
                # before anything here is reached; this is the backstop.
                if dgram_name not in fit:
                    raise RuntimeError(
                        f"NXstress: cannot append mask '{mask}': the target entry has no "
                        f"DIFFRACTOGRAM group '{dgram_name}'."
                    )
                _Diffractogram.init_group(wss, mask, indexed, data=fit[dgram_name])
                continue
            # `in fit`, not `in fit.NXdata`: the latter is a LIST of NXdata objects, so a
            # string is never a member of it and this guard could not fire. Found while
            # implementing 04c; see that spec's Follow-up 2.
            if dgram_name in fit:
                raise RuntimeError(
                    f"Usage error: DIFFRACTOGRAM (NXdata) group '{dgram_name}' already exists in the current (NXprocess) group."
                )
            fit[dgram_name] = _Diffractogram.init_group(wss, mask, indexed)

        return fit

    @classmethod
    def appendableDatasets(cls, fit) -> dict:
        """Every dataset under `FIT` that an append grows.

        Two families, both row-aligned and both growing on every append: the
        per-key DIFFRACTOGRAM groups (scan-point aligned) and the peak and
        background parameter groups (peak-index aligned). The parameter groups'
        field *names* come from the fitted profile, so they are enumerated rather
        than listed -- their `title` is the group's one entry-wide scalar and drops
        out for being scalar, not by being named here.

        Args:
            fit: The target entry's existing `FIT` group.

        Returns:
            Path -> `NXfield`, for `NXstress._validateAppendableShapes`.
        """
        found: dict = {}
        for name in fit:
            child = fit[name]
            prefix = f"{GROUP_NAME.FIT}/{name}"
            if isinstance(child, NXdata):
                found.update(_Diffractogram.appendableDatasets(child, prefix))
            elif name in (GROUP_NAME.PEAK_PARAMETERS, GROUP_NAME.BACKGROUND_PARAMETERS):
                found.update(row_aligned_fields(child, prefix))
        return found

    @classmethod
    def validateAppend(cls, wss: list[HidraWorkspace], fit) -> None:
        """Check incoming diffraction against every existing DIFFRACTOGRAM group.

        The mask *set* is checked by `NXstress._classifyAppend`; this checks the
        shape of what would go into each group that is already there.

        Args:
            wss: Workspaces being appended.
            fit: The target entry's existing `FIT` group.

        Raises:
            RuntimeError: If any mask's two-theta bin count differs.
        """
        mask_keys = set()
        for ws in wss:
            mask_keys |= nxstress_diffractogram_keys(ws._diff_data_set.keys())
        for mask in sorted(mask_keys):
            dgram_name = group_naming_scheme(GROUP_NAME.DIFFRACTOGRAM, mask)
            if dgram_name in fit:
                _Diffractogram.validateAppend(wss, mask, fit[dgram_name])

    @classmethod
    def validateWorkspaceAndPeaksData(cls, ws: HidraWorkspace, peakss: list[PeakCollection]):
        # VERIFY that scan_point[s] and mask[s] reference by any `PeakCollection` are present in the workspace.
        scan_point = set(ws.get_sub_runs().raw_copy())

        diff_data_keys = set(ws._diff_data_set.keys())
        var_data_keys = set(ws._var_data_set.keys())
        if diff_data_keys != var_data_keys:
            raise ValueError(
                f"Diffraction-data keys '{diff_data_keys}' and variance keys '{var_data_keys}' are not the same."
            )

        for peaks in peakss:
            # VERIFY that any <scan point> referenced by any `PeakCollection` is included in the workspace.

            # Note: `PeakCollection.get_sub_runs()` is *broken*:
            #   it does not actually return a `SubRuns` instance!
            peaks_scan_point = set(peaks._sub_run_array.raw_copy())
            if not peaks_scan_point.issubset(scan_point):
                raise ValueError(
                    f"Scan points {peaks_scan_point}, required by `PeakCollection`,\n"
                    f"  are not present in workspace scan points {scan_point}."
                )

            # VERIFY that the diffractogram a `PeakCollection` was fitted on exists.
            #
            # `PeakCollection.mask` is a DIFFRACTOGRAM key, not a mask name -- it
            # indexes `_diff_data_set`, as the lookup below does. It was previously
            # also checked against the *mask* namespace, which rejected every
            # legitimate texture peak: `eta_0.0` is a perfectly good diffractogram
            # key and is not a mask. That check is gone; this one is the correct
            # form, and was always doing the real work.
            peaks_mask = peaks.mask
            data_key = _Diffractogram._diffraction_data_key(peaks_mask)
            if data_key not in ws._diff_data_set:
                raise ValueError(
                    f"Reduced data for diffractogram '{peaks_mask}', required by `PeakCollection`,\n"
                    f"  is not present in the workspace. Available: "
                    f"{sorted(nxstress_diffractogram_keys(ws._diff_data_set.keys()))}."
                )
