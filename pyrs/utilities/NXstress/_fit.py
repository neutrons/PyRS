"""
pyrs/utilities/NXstress/_fit.py

Private service class for NeXus NXstress-compatible I/O.
This class provides I/O for the `fit` `NXprocess` subgroup:
  this subgroup includes the reduced output data as a 'diffraction_data' `NXdata` group.
"""
from nexusformat.nexus import (
    NXdata, NXfield, NXnote, NXparameters, NXprocess
)
import numpy as np
from typing import Tuple

from pyrs.peaks.peak_collection import PeakCollection
from pyrs.core.peak_profile_utility import (
    BackgroundFunction, EFFECTIVE_PEAK_PARAMETERS
)
from pyrs.core.workspaces import HidraWorkspace
from pyrs.dataobjects.sample_logs import SampleLogs
from pyrs.utilities.pydantic_transition import validate_call_

from ._definitions import (
    FIELD_DTYPE, CHUNK_SHAPE, DEFAULT_TAG,
    GROUP_NAME, EFFECTIVE_BACKGROUND_PARAMETERS,
    group_naming_scheme
)

"""
REQUIRED PARAMETERS FOR NXstress:
---------------------------------

├─ fit                                    (NXprocess, group)
│   ├─ @date                               (attribute: ISO8601 string)
│   ├─ @program                            (attribute: string)
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
    def init_group(cls, peaks: PeakCollection) -> NXparameters:
        # required 'peak_parameters' subgroup
        pp = NXparameters()
        
        N_scan = len(peaks.sub_runs)
        peak_profile = peaks.peak_profile
        background_function = BackgroundFunction.getFunction(peaks.background_type)
        background_params = background_function.native_parameters
        
        # Use _effective_ peak parameters here: all peaks will then have the same number of parameter,
        #   and all parameter values will be in the expected column.
        # For the moment, I'm assuming that we have one value for each of 'N_scan' subruns.
        
        ## Include _only_ the peak parameters, _exclude_ the background parameters.
        peak_parameters = [param for param in EFFECTIVE_PEAK_PARAMETERS if param not in EFFECTIVE_BACKGROUND_PARAMETERS]
        params_name = np.tile(np.array(peak_parameters), (N_scan, 1))
        params_value, params_error = peaks.get_effective_params()
        
        pp['title'] = NXfield(peak_profile, dtype=FIELD_DTYPE.STRING.value)
        
        # Note: fitted values use `FLOAT_DATA`, calculated values use `FLOAT_CONSTANT`.
        #   This is a premature optimization: probably both should just use `np.float64`.
        pp['center'] = params_value['Center'].astype(FIELD_DTYPE.FLOAT_DATA.value)
        pp['center'].attrs['units'] = 'degree'
        pp['center_errors'] = params_error['Center'].astype(FIELD_DTYPE.FLOAT_DATA.value)
        pp['center_errors'].attrs['units'] = 'degree'
        pp['height'] = params_value['Height'].astype(FIELD_DTYPE.FLOAT_DATA.value)
        pp['height'].attrs['units'] = 'counts'
        pp['height_errors'] = params_error['Height'].astype(FIELD_DTYPE.FLOAT_DATA.value)
        pp['height_errors'].attrs['units'] = 'counts'
        pp['fwhm'] = params_value['FWHM'].astype(FIELD_DTYPE.FLOAT_DATA.value)
        pp['fwhm'].attrs['units'] = 'degree'
        pp['fwhm_errors'] = params_error['FWHM'].astype(FIELD_DTYPE.FLOAT_DATA.value)
        pp['fwhm_errors'].attrs['units'] = 'degree'
        
        # Voigt or Pseudo-Voigt: Lorentzian fraction
        pp['form_factor'] = (1.0 - params_value['Mixing']).astype(FIELD_DTYPE.FLOAT_DATA.value)
        pp['form_factor'].attrs['units'] = '1'
        pp['form_factor_errors'] = params_error['Mixing'].astype(FIELD_DTYPE.FLOAT_DATA.value)
        pp['form_factor_errors'].attrs['units'] = '1'
        
        return pp
          

class _BackgroundParameters:
        
    @classmethod
    @validate_call_
    def init_group(cls, peaks: PeakCollection) -> NXparameters:
        # required 'background_parameters' subgroup
        bp = NXparameters()

        background_function = BackgroundFunction.getFunction(peaks.background_type)

        ## Include _only_ the background parameters.
        params_value, params_error = peaks.get_effective_params()
        bp['title'] = NXfield(background_function, dtype=FIELD_DTYPE.STRING.value) 
        bp['A'] = np.column_stack([params_value[param] for param in EFFECTIVE_BACKGROUND_PARAMETERS]) # shape: (N_scan, N_param)
        
        return bp
        
class _Diffractogram:

    @classmethod
    def _get_diffraction_data(cls, ws: HidraWorkspace, mask_name: str) -> Tuple[np.ndarray, np.ndarray]:
        # Workaround for PyRS codebase bizarre use of `None` as the default key.
        data_key, errors_key = cls._diffraction_data_keys(mask_name)
        return ws._diff_data_set[data_key], ws._var_data_set[errors_key]

    @classmethod
    def _diffraction_data_keys(cls, mask_name: str) -> Tuple[str, str]:
        # Workaround for PyRS codebase bizarre use of `None` as the default key.
        if mask_name != DEFAULT_TAG:
            data_key = mask_name
            errors_key = f"{mask_name}_var"
        else:
            data_key = None # WTF?
            errors_key = 'main_var'
        return data_key, errors_key 
        
    @classmethod
    def _init(cls, ws: HidraWorkspace) -> NXdata:
        if ws._2theta_matrix is None:
            raise RuntimeError("Usage error: cannot write NXstress file: workspace includes no reduced data.")        
        dg = NXdata()
        return dg 
        
    @classmethod
    @validate_call_
    def init_group(cls, ws: HidraWorkspace, maskName: str, peaks: PeakCollection) -> NXparameters:
        # required DIFFRACTOGRAM (NXdata) subgroup:        
        data_key, errors_key = cls._diffraction_data_keys(maskName)
        if data_key not in ws._diff_data_set or errors_key not in ws._var_data_set:
            # *** DEBUG ***
            print(f"=====> diffraction data: {ws._diff_data_set.keys()}, errors: {ws._var_data_set.keys()}")
            raise RuntimeError(f"Reduced data for mask '{maskName}' is not attached to the workspace.")
        
        dg = cls._init(ws)
        dg.attrs['signal'] = GROUP_NAME.DGRAM_DIFFRACTOGRAM
        dg.attrs['auxiliary_signals'] = [
            GROUP_NAME.DGRAM_DIFFRACTOGRAM_ERRORS,
            GROUP_NAME.DGRAM_FIT,
            GROUP_NAME.DGRAM_FIT_ERRORS
        ] 
        dg.attrs['axes'] = ['scan_point', 'two_theta']
        dg['scan_point'] = NXfield(ws.get_sub_runs())
        dg['scan_point'].attrs['units'] = ''
        
        two_theta = ws._2theta_matrix
        dg['two_theta'] = NXfield(two_theta)
        dg['two_theta'].attrs['units'] = 'degree'
        
        data, errors = cls._get_diffraction_data(ws, maskName)
        dg[GROUP_NAME.DGRAM_DIFFRACTOGRAM] = NXfield(data, dtype=FIELD_DTYPE.FLOAT_DATA.value)
        dg[GROUP_NAME.DGRAM_DIFFRACTOGRAM].attrs['interpretation'] = 'spectrum'
        dg[GROUP_NAME.DGRAM_DIFFRACTOGRAM].attrs['units'] = 'counts'
 
        dg[GROUP_NAME.DGRAM_DIFFRACTOGRAM_ERRORS] = NXfield(errors, dtype=FIELD_DTYPE.FLOAT_DATA.value)
        
        # ENTRY/FIT/DIFFRACTOGRAM/fit, fit_errors: required datasets: these should contain the spectrum reconstructed from the fitted model.
        #   For the moment, this will be initialized to NaN.
        dg[GROUP_NAME.DGRAM_FIT] = NXfield(np.empty((0, 0), dtype=np.float64),
                                           maxshape=(None, None), chunks=CHUNK_SHAPE, fillvalue=np.nan)
        dg[GROUP_NAME.DGRAM_FIT].attrs['interpretation'] = 'spectrum'
        dg[GROUP_NAME.DGRAM_FIT].attrs['units'] = 'counts'                                        
        dg[GROUP_NAME.DGRAM_FIT_ERRORS] = NXfield(np.empty((0, 0), dtype=np.float64),
                                                  maxshape=(None, None), chunks=CHUNK_SHAPE, fillvalue=np.nan)
        dg[GROUP_NAME.DGRAM_FIT_ERRORS].attrs['units'] = 'counts'                                        
        
        return dg
        

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
    def _init(cls, maskName: str, logs: SampleLogs) -> NXprocess:
        # Initialize the 'FIT' (NXprocess) group:
        #   in case of multiple 'FIT' groups: <mask name> is used as a name suffix.

        fit = NXprocess()
        
        fit['name'] = f'NXprocess group for mask {maskName}'
        # Required information fields:
        fit['raw_data_file'] = NXfield('')
        fit['date'] = NXfield('')
        fit['program'] = NXfield('PyRS')
        fit['description'] = NXnote()
        fit['description']['detector_mask'] = NXfield(maskName)

        return fit
    
    @classmethod
    @validate_call_
    def init_group(cls, maskName: str, ws: HidraWorkspace, peaks: PeakCollection, logs: SampleLogs):
        # Initialize a new 'FIT' (NXprocess) group:
        #   in case of multiple 'FIT' groups: <mask name> should be used as a name suffix
        #   (see `_definitions.group_naming_scheme`).
        
        ## Under `NXstress`: `FIT` (NXprocess) groups contain peak and background-fit results, including any
        ##    information relevant to the fitting process used.
        fit = cls._init(maskName, logs)
        fit[GROUP_NAME.PEAK_PARAMETERS] = _PeakParameters.init_group(peaks)
        fit[GROUP_NAME.BACKGROUND_PARAMETERS] = _BackgroundParameters.init_group(peaks)
        fit[GROUP_NAME.DGRAM_DIFFRACTOGRAM] = _Diffractogram.init_group(ws, maskName, peaks)
        
        return fit
