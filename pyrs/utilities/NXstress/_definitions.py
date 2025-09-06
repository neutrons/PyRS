"""
pyrs/utilities/NXstress/_definitions.py

Constants and definitions used by NeXus NXstress-compatible I/O.
"""
from enum import StrEnum
import h5py
from nexusformat.nexus import (
    NXdata,
    NXentry,
    NXinstrument,
    NXprocess,
    NXreflections,
    NXsample
)
import numpy as np

REQUIRED_LOGS: List[str] = []

class FIELD_DTYPE(type, Enum):
    # HDF5 dataset types for various fields
    FLOAT_CONSTANT = np.float64
    FLOAT_DATA     = np.float32
    INT_DATA       = np.int32
    STRING         = h5py.string_dtype(encoding='utf-8')

CHUNK_SHAPE = (100,)    # Reasonable chunk size (tunable)

def GROUP_NAME(StrEnum):
    # These will be the <base name> (, in case of multiple group instances),
    #   and may be modified as required.
    ENTRY = ('entry', True, NXentry)                              # `NXentry`          : one or more per HDF5 file
    PEAKS = ('reduced_peaks', False, NXreflections)               # `NXreflections`    : one per `NXentry`
    FIT   = ('diffraction_fit', True, NXprocess)                  # `NXprocess`        : one or more per `NXentry`
    DIFFRACTOGRAM = ('diffractogram', False, NXdata)              # `NXdata`           : one per `NXprocess`
    SAMPLE_DESCRIPTION = ('sample_description', False, NXsample)  # `NXsample`         : one per `NXentry`
    INSTRUMENT = ('instrument', False, NXinstrument)              # `NXinstrument`     : one per `NXentry`
    INPUT_DATA = ('input_data', True, NXdata)                     # [optional] `NXdata`: one or more per `NXentry`

    def __new__(cls, value, allowMultiple: bool, nxClass: NXgroup):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.allowMultiple = allowMultiple
        obj.nxClass = nxClass
        return obj
            
def NXSTRESS_REQUIRED_NAME(StrEnum):
    # These are *required* names, as specified in the `NXstress` schema,
    #   and should not be changed.
    PEAK_PARAMETERS = ('peak_parameters', False, NXparameters)
    BACKGROUND_PARAMETERS = ('background_parameters', False, NXparameters)
    DIFFRACTOGRAM = ('diffractogram', False, NXfield)
    DIFFRACTOGRAM_ERRORS = ('diffractogram_errors', False, NXfield)
    FIT = ('fit', False, NXfield)
    FIT_ERRORS = ('fit_errors', False, NXfield)

    def __new__(cls, value, allowMultiple: bool, nxClass: NXgroup):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.allowMultiple = allowMultiple
        obj.nxClass = nxClass
        return obj

EFFECTIVE_BACKGROUND_PARAMETERS = ['A0', 'A1', 'A2']

# Name or suffix corresponding to the default dataset:
# -- when a group name uses this as a suffix tag (e.g. multiple FIT (NXprocess) groups, one for each mask)
#    this default tag should be _omitted_ from the group name.
# -- presently this is only used for masks, to allow multiple FIT groups.
DEFAULT_TAG = '_DEFAULT_'

def group_naming_scheme(base_name: str, suffix: int | str) -> str:
    # Generate the name for an HDF5 group, allowing for multiple group instances:
    #   instance: 
    #     int: enumerated group names: '_1' is omitted;
    #     str: group names (e.g. 'FIT' (NXprocess)), delineated using a tag suffix: '__DEFAULT_' is omitted.
    tag = ''
    if isinstance(suffix, int) and suffix > 1\
        or isinstance(suffix, str) and suffix != DEFAULT_TAG:
      tag = f'_{suffix}'
    else:
      raise f'`group_naming_scheme`: not implemented for "{suffix}" suffix'
    return f'{base_name}{tag}'

