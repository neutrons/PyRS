"""
peaks_IO

Private service class for NeXus NXstress-compatible I/O.
This class provides I/O for the `peaks` `NXreflections` subgroup:
  this subgroup includes fitted peak data, as used in reduction.
"""

"""
REQUIRED PARAMETERS FOR NXstress:
---------------------------------

├─ peaks                                  (NXreflections, group)
│   ├─ h                                   (dataset)
│   ├─ k                                   (dataset)
│   ├─ l                                   (dataset)
│   └─ phase_name                          (dataset)
"""

import numpy as np
from nexusformat.nexus import NXentry, NXreflections, NXfield
from pydantic import validate_call
import re
from typing import Tuple

class _Peaks:
    ########################################
    # ALL methods must be `classmethod`.  ##
    ########################################

    @classmethod
    def _parse_peak_tag(cls, tag: str) -> Tuple[str, Tuple[int, int, int]]:
        # Parse a peak-tag string into its <phase name> and Miller indices (h, k, l).
        maybeHKL = max(re.finditer(r"\d+", s), key=lambda m: len(m.group(0)), default=None)
        if maybeHKL is None or len(maybeHKL.group(0)) % 3 != 0:
            raise RuntimeError(
                f"Unable to parse peak tag '{tag}' into its its <phase name> and Miller indices (h, k, l)."
            )
        # Extract <phase name> as the rest of the tag.
        i, j = maybeHKL.span()
        phase = tag[:i] + tag[j:]
        
        # Extract (h, k, l)
        maybeHKL = maybeHKL.group(0)
        N_d = len(maybeHKL) // 3
        h, k, l = int(maybeHKL[0: N_d]), int(maybeHKL[N_d: 2 * N_d]), int(maybeHKL[2 * N_d: 3 * N_d])
        
        return phase, (h, k, l)
        
    @classmethod
    def _init(cls, peaks: PeakCollection, logs: SampleLogs) -> NXreflections:
        # Initialize the 'PEAKS' group
        peaks = NXreflections()

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
        
        ## Components of the normalized scattering vector Q in the sample reference frame
        ##   'qx', 'qy', and 'qz' are *required* by NXstress, but it looks as if PyRS doesn't
        ##   use these -- initialize to `NaN`.
        peaks['qx'] = NXfield(np.empty((0,), dtype=np.float64),
                              maxshape=(None,), chunks=chunk_shape, fillvalue=np.nan)
        peaks['qx'].attrs['units'] = '1'        
        peaks['qy'] = NXfield(np.empty((0,), dtype=np.float64),
                              maxshape=(None,), chunks=chunk_shape, fillvalue=np.nan)        
        peaks['qy'].attrs['units'] = '1'        
        peaks['qz'] = NXfield(np.empty((0,), dtype=np.float64),
                              maxshape=(None,), chunks=chunk_shape, fillvalue=np.nan)        
        peaks['qz'].attrs['units'] = '1'        
        ##

        peaks['center'] = NXfield(
            np.empty((0,), dtype=np.float64),
            maxshape=(None,), chunks=chunk_shape,
            units='angstrom'
        )        
        peaks['center_errors'] = NXfield(
            np.empty((0,), dtype=np.float64),
            maxshape=(None,),
            chunks=chunk_shape,
            units='angstrom')
        peaks['center_type'] = NXfield('d-spacing')  
        
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

        return peaks
    
    @classmethod
    def init_group(cls, peaks: PeakCollection, logs: SampleLogs) -> NXreflections:
        # Initialize the PEAKS group:
        #   according to the NXstress schema, this group contains the canonical reduction data,
        #   in a form usable for stress / strain calculations.
        
        # TODO: this section is implemented in a form that allows new data to be appended,
        #   however at present appending data is not yet supported.
        peaks = cls._init_group(nx)

        scan_point = peaks.get_sub_runs()
        N_scan = len(scan_points)
        phase_name, (h, k, l) = cls._parse_peak_tag(peaks.peak_tag)
        
        # Each dataset has scan point as its first index.
        phase_name = np.array((phase_name,) * N_scan) 
        h, k, l = np.array((h,) * N_scan), np.array((k,) * N_scan), np.array((l,) * N_scan)
        
        d_reference, d_reference_error = peaks.get_d_reference()
        d_reference = np.array((d_reference,) * N_scan)
        d_reference_error = np.array((d_reference_error,) * N_scan)
        
        curr_len = peaks['h'].shape[0]
        new_len = curr_len + N
        
        peaks['scan_point'].resize((new_len,))
        
        peaks['h'].resize((new_len,))
        peaks['k'].resize((new_len,))
        peaks['l'].resize((new_len,))
        peaks['phase_name'].resize((new_len,))

        # For `PEAKS` (NXreflections) group: 'center' means `d_reference`.
        peaks['center'].resize((new_len,))
        peaks['center_error'].resize((new_len,))
        
        peaks['scan_point'][curr_len:] = scan_point
        peaks['h'][curr_len:] = h
        peaks['k'][curr_len:] = k
        peaks['l'][curr_len:] = l
        peaks['phase_name'][curr_len:] = phase_name

        peaks['center'][curr_len:] = d_reference
        peaks['center_error'][curr_len:] = d_reference_error

        return peaks
