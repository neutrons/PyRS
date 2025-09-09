"""
pyrs/utilities/NXstress/_peaks.py

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
from nexusformat.nexus import (
    NXentry, NXreflections, NXfield
)
import re
from typing import Tuple

from pyrs.peaks.peak_collection import PeakCollection
from pyrs.dataobjects.sample_logs import SampleLogs
from pyrs.utilities.pydantic_transition import validate_call_

from ._definitions import CHUNK_SHAPE, FIELD_DTYPE

class _Peaks:
    ########################################
    # ALL methods must be `classmethod`.  ##
    ########################################

    @classmethod
    def _parse_peak_tag(cls, tag: str) -> Tuple[str, Tuple[int, int, int]]:
        # Parse a peak-tag string into its <phase name> and Miller indices (h, k, l).
        maybeHKL = max(re.finditer(r"\d+", tag), key=lambda m: len(m.group(0)), default=None)
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
    def _init(cls, logs: SampleLogs) -> NXreflections:
        # Initialize the 'PEAKS' group
        peaks = NXreflections()

        peaks['scan_point'] = NXfield(np.empty((0,), dtype=np.int32),
                                      maxshape=(None,), chunks=CHUNK_SHAPE)

        peaks['h'] = NXfield(np.empty((0,), dtype=np.int32),
                             maxshape=(None,), chunks=CHUNK_SHAPE)
        peaks['k'] = NXfield(np.empty((0,), dtype=np.int32),
                             maxshape=(None,), chunks=CHUNK_SHAPE)
        peaks['l'] = NXfield(np.empty((0,), dtype=np.int32),
                             maxshape=(None,), chunks=CHUNK_SHAPE)
        
        peaks['phase_name'] = NXfield(np.empty((0,), dtype=FIELD_DTYPE.STRING.value),
                                      maxshape=(None,), chunks=CHUNK_SHAPE)
        
        ## Components of the normalized scattering vector Q in the sample reference frame
        ##   'qx', 'qy', and 'qz' are *required* by NXstress, but it looks as if PyRS doesn't
        ##   use these -- initialize to `NaN`.
        peaks['qx'] = NXfield(np.empty((0,), dtype=np.float64),
                              maxshape=(None,), chunks=CHUNK_SHAPE, fillvalue=np.nan)
        peaks['qx'].attrs['units'] = '1'        
        peaks['qy'] = NXfield(np.empty((0,), dtype=np.float64),
                              maxshape=(None,), chunks=CHUNK_SHAPE, fillvalue=np.nan)        
        peaks['qy'].attrs['units'] = '1'        
        peaks['qz'] = NXfield(np.empty((0,), dtype=np.float64),
                              maxshape=(None,), chunks=CHUNK_SHAPE, fillvalue=np.nan)        
        peaks['qz'].attrs['units'] = '1'        
        ##

        peaks['center'] = NXfield(
            np.empty((0,), dtype=np.float64),
            maxshape=(None,), chunks=CHUNK_SHAPE,
            units='angstrom'
        )        
        peaks['center_errors'] = NXfield(
            np.empty((0,), dtype=np.float64),
            maxshape=(None,),
            chunks=CHUNK_SHAPE,
            units='angstrom')
        peaks['center_type'] = NXfield('d-spacing')  
        
        # Sample position for each subrun -- initialize to `NaN`.
        peaks['sx'] = NXfield(np.empty((0,), dtype=np.float64),
                              maxshape=(None,), chunks=CHUNK_SHAPE, fillvalue=np.nan)
        peaks['sx'].attrs['units'] = logs.units('sx')        
        peaks['sy'] = NXfield(np.empty((0,), dtype=np.float64),
                              maxshape=(None,), chunks=CHUNK_SHAPE, fillvalue=np.nan)        
        peaks['sy'].attrs['units'] = logs.units('sx')        
        peaks['sz'] = NXfield(np.empty((0,), dtype=np.float64),
                              maxshape=(None,), chunks=CHUNK_SHAPE, fillvalue=np.nan)        
        peaks['sx'].attrs['units'] = logs.units('sx')        

        return peaks
    
    @classmethod
    def init_group(cls, peak_collection: PeakCollection, logs: SampleLogs) -> NXreflections:
        # Initialize the PEAKS group:
        #   according to the NXstress schema, this group contains the canonical reduction data,
        #   in a form usable for stress / strain calculations.
        
        # TODO: this section is implemented in a form that allows new data to be appended,
        #   however at present appending data is not yet supported.
        peaks = cls._init(logs)

        scan_point = peak_collection.sub_runs.raw_copy()
        N_scan = len(scan_point)
        phase_name, (h, k, l) = cls._parse_peak_tag(peak_collection.peak_tag)
        
        # Each dataset has scan point as its first index.
        phase_name = np.array((phase_name,) * N_scan) 
        h, k, l = np.array((h,) * N_scan), np.array((k,) * N_scan), np.array((l,) * N_scan)
        
        d_reference, d_reference_error = peak_collection.get_d_reference()
        d_reference = np.array((d_reference,) * N_scan)
        d_reference_error = np.array((d_reference_error,) * N_scan)
        
        curr_len = peaks['h'].shape[0]
        new_len = curr_len + N_scan
        
        peaks['scan_point'].resize((new_len,))
        
        peaks['h'].resize((new_len,))
        peaks['k'].resize((new_len,))
        peaks['l'].resize((new_len,))
        peaks['phase_name'].resize((new_len,))

        # For `PEAKS` (NXreflections) group: 'center' means `d_reference`.
        peaks['center'].resize((new_len,))
        peaks['center_errors'].resize((new_len,))
        
        peaks['scan_point'][curr_len:] = scan_point
        peaks['h'][curr_len:] = h
        peaks['k'][curr_len:] = k
        peaks['l'][curr_len:] = l
        peaks['phase_name'][curr_len:] = phase_name

        peaks['center'][curr_len:] = d_reference
        peaks['center_errors'][curr_len:] = d_reference_error

        return peaks
