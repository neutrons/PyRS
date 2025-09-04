"""
peaks_IO

Private service class for NeXus NXstress-compatible I/O.
This class provides I/O for the `peaks` `NXreflections` subgroup:
  this subgroup includes fitted peak data, as used in reduction.
"""

import numpy as np
from nexusformat.nexus import NXentry, NXreflections, NXfield, NXroot, save


"""
REQUIRED PARAMETERS FOR NXstress:
---------------------------------

├─ peaks                                  (NXreflections, group)
│   ├─ h                                   (dataset)
│   ├─ k                                   (dataset)
│   ├─ l                                   (dataset)
│   └─ phase_name                          (dataset)
"""
class Peaks_IO:
    ########################################
    # ALL methods must be `classmethod`.  ##
    ########################################

    @classmethod
    def _init_group(cls, nx: NXFile, peaks: PeakCollection, logs: SampleLogs) -> NXreflections:
        # Initialize (or re-initialize) the 'peaks' group

        # assumes 'entry' already exists
        if 'peaks' not in nx.root['entry']:
            # Create an NXreflections (PEAKS) group under entry
            nx.root['entry']['peaks'] = NXreflections()
        peaks = nx.root['entry']['peaks']

        # Create extensible (resizable) fields for h, k, l, etc.
        chunk_shape = (100,)             # Reasonable chunk size (tunable)
        
        # Use h5py's special dtype for variable-length UTF-8 strings
        vlen_str_dtype = h5py.string_dtype(encoding='utf-8')

        peaks['scan_point'] = NXfield(np.empty(initial_shape, dtype=np.int32),
                                      maxshape=max_shape, chunks=chunk_shape)

        peaks['h'] = NXfield(np.empty((0,), dtype=np.int32),
                             maxshape=(None,), chunks=chunk_shape)
        peaks['k'] = NXfield(np.empty((0,), dtype=np.int32),
                             maxshape=(None,), chunks=chunk_shape)
        peaks['l'] = NXfield(np.empty((0,), dtype=np.int32),
                             maxshape=(None,), chunks=chunk_shape)
        
        peaks['phase_name'] = NXfield(np.empty((0,), dtype=vlen_str_dtype),
                                      maxshape=(None,), chunks=chunk_shape)
        
        # Components of the normalized scattering vector Q in the sample reference frame
        #   'qx', 'qy', and 'qz' are *required* by NXstress, but it looks as if PyRS doesn't
        #   use these -- initialize to `NaN`.
        peaks['qx'] = NXfield(np.empty((0,), dtype=np.float64),
                              maxshape=(None,), chunks=chunk_shape, fillvalue=np.nan)
        peaks['qx'].attrs['units'] = '1'        
        peaks['qy'] = NXfield(np.empty((0,), dtype=np.float64),
                              maxshape=(None,), chunks=chunk_shape, fillvalue=np.nan)        
        peaks['qy'].attrs['units'] = '1'        
        peaks['qz'] = NXfield(np.empty((0,), dtype=np.float64),
                              maxshape=(None,), chunks=chunk_shape, fillvalue=np.nan)        
        peaks['qz'].attrs['units'] = '1'        

        peaks['peak_profile'] = NXfield(np.empty((0,), dtype=vlen_str_dtype),
                                        maxshape=(None,), chunks=chunk_shape)
        peaks['background'] = NXfield(np.empty((0,), dtype=vlen_str_dtype),
                                      maxshape=(None,), chunks=chunk_shape)
        
        peaks['params_name'] = NXfield(np.empty((0, N_param), dtype=vlen_str_dtype),
                                       maxshape=(None, N_param), chunks=chunk_shape)        
        peaks['params_value'] = NXfield(np.empty((0, N_param), dtype=np.float64),
                                        maxshape=(None, N_param), chunks=chunk_shape)        
        peaks['params_error'] = NXfield(np.empty((0, N_param), dtype=np.float64),
                                        maxshape=(None, N_param), chunks=chunk_shape)        

        peaks['center'] = NXfield(np.empty((0,), dtype=np.float64),
                                  maxshape=(None,), chunks=chunk_shape)
        peaks['center'].attrs['units'] = 'degree'
        
        peaks['center_errors'] = NXfield(np.empty((0,), dtype=np.float64),
                                         maxshape=(None,), chunks=chunk_shape)
        peaks['center_errors'].attrs['units'] = 'degree'
        peaks['center_type'] = NXfield(np.empty((0,), dtype=vlen_str_dtype),
                                       maxshape=(None,), chunks=chunk_shape)        

        peaks['fit_costs'] = NXfield(np.empty((0,), dtype=np.float64),
                                     maxshape=(None,), chunks=chunk_shape)
        peaks['fit_status'] = NXfield(np.empty((0,), dtype=vlen_str_dtype),
                                      maxshape=(None,), chunks=chunk_shape)
       
        ## Wavelength doesn't go here: it should go to NXinstrument (or NXmonochromator).
        ## peaks['wavelength'] = NXfield(np.empty((0,), dtype=np.float64),
        ##                              maxshape=(None,), chunks=chunk_shape)
        
        # Sample position for each subrun -- initialize to `NaN`.
        peaks['sx'] = NXfield(np.empty((0,), dtype=np.float64),
                              maxshape=(None,), chunks=chunk_shape, fillvalue=np.nan)
        peaks['sx'].attrs['units'] = logs.units('sx')        
        peaks['sy'] = NXfield(np.empty((0,), dtype=np.float64),
                              maxshape=(None,), chunks=chunk_shape, fillvalue=np.nan)        
        peaks['sy'].attrs['units'] = logs.units('sx')        
        peaks['sz'] = NXfield(np.empty((0,), dtype=np.float64),
                              maxshape=(None,), chunks=chunk_shape, fillvalue=np.nan)        
        peaks['sx'].attrs['units'] = logs.units('sx')        


x These go in `FIT/peak_parameters` !        
x        peaks['d_reference'] = NXfield(np.empty((0,), dtype=np.float64),
                                      maxshape=(None,), chunks=chunk_shape)
         
x        peaks['d_reference_error'] = NXfield(np.empty((0,), dtype=np.float64),
                                      maxshape=(None,), chunks=chunk_shape)
        
x        peaks['strain'] = NXfield(np.empty((0,), dtype=np.float64),
                                      maxshape=(None,), chunks=chunk_shape)
        
x        peaks['strain_error'] = NXfield(np.empty((0,), dtype=np.float64),
                                      maxshape=(None,), chunks=chunk_shape)
        return peaks
    
    @classmethod
    def write(cls, nx: NXFile, peaks: PeakCollection, logs: SampleLogs):
        # Initialize and / or append to the PEAKS group:
        #   append the values for a single peak, for all of its scan_points,
        #   to the PEAKS group in the NXstress-format NXFile object.
        peaks = cls._init_group(nx)

        scan_point = peaks.get_sub_runs()
        N = len(scan_points)
        (h, k, l), phase_name = cls._parsePeakTag(peaks.peak_tag)
        h, k, l, phase_name = np.array((h,) * N), np.array((k,) * N), np.array((l,) * N), np.array((phase_name,) * N)

        peak_profile = np.array(peaks.peak_profile,) * N)
        background = np.array(peaks.background_type,) * N)
        
        # Use _effective_ peak parameters here: all peaks will then have the same number of parameter,
        #   and all parameter values will be in the expected column.
        # For the moment, I'm assuming that we have one value for each of 'N' subruns.
        params_name = np.tile(np.array(EFFECTIVE_PEAK_PARAMETERS), (N, 1))
        vs, es = peaks.get_effective_params()
        params_value = np.column_stack([vs[param] for param in EFFECTIVE_PEAK_PARAMETERS]) # shape: (N, N_param)
        params_error = np.column_stack([es[param] for param in EFFECTIVE_PEAK_PARAMETERS])
   
        center = params_value['Center']
        center_errors = params_error['Center']
        center_type = np.array(('two-theta',) * N)
        
        fit_costs = peaks.get_chisq()
        fit_status = peaks.get_fit_status()
        
        ## Wavelength doesn't go here: it should go to NXinstrument (or NXmonochromator).
        ## wavelength = np.array((peaks._wavelength,) * N)
        
        d_reference, d_reference_error = peaks.get_d_reference()
        d_reference = np.array((d_reference,) * N)
        d_reference_error = np.array((d_reference_error,) * N)
        
        strain, strain_error = peaks.get_strain()
        strain = np.array((strain,) * N)
        strain_error = np.array((strain_error,) * N)
   
        curr_len = peaks['h'].shape[0]
        new_len = curr_len + N
        
        peaks['scan_point'].resize((new_len,))
        
        peaks['h'].resize((new_len,))
        peaks['k'].resize((new_len,))
        peaks['l'].resize((new_len,))
        peaks['phase_name'].resize((new_len,))
        
        peaks['peak_profile'].resize((new_len,))
        peaks['background'].resize((new_len,))
        peaks['params_name'].resize((new_len, N_param))
        peaks['params_value'].resize((new_len, N_param))
        peaks['params_error'].resize((new_len, N_param))
        peaks['fit_costs'].resize((new_len,))
        peaks['fit_status'].resize((new_len,))
        
        peaks['wavelength'].resize((new_len,))
        peaks['d_reference'].resize((new_len,))
        peaks['d_reference_error'].resize((new_len,))
        peaks['strain'].resize((new_len,))
        peaks['strain_error'].resize((new_len,))
        
        peaks['scan_point'][curr_len:] = scan_point
        peaks['h'][curr_len:] = h
        peaks['k'][curr_len:] = k
        peaks['l'][curr_len:] = l
        peaks['phase_name'][curr_len:] = phase_name

        peaks['peak_profile'][curr_len:] = peak_profile
        peaks['background'][curr_len:] = background
        peaks['params_name'][curr_len:] = params_name
        peaks['params_value'][curr_len:] = params_value
        peaks['params_error'][curr_len:] = params_error
        peaks['fit_costs'][curr_len:] = fit_costs
        peaks['fit_status'][curr_len:] = fit_status
        
        peaks['wavelength'][curr_len:] = wavelength
        peaks['d_reference'][curr_len:] = d_reference
        peaks['d_reference_error'][curr_len:] = d_reference_error
        peaks['strain'][curr_len:] = strain
        peaks['strain_error'][curr_len:] = strain_error        
