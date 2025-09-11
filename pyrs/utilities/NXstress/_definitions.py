"""
pyrs/utilities/NXstress/_definitions.py

Constants and definitions used by NeXus NXstress-compatible I/O.
"""
from enum import Enum, StrEnum
import h5py
from nexusformat.nexus import (
    NXbeam,
    NXcollection,
    NXdata,
    NXdetector,
    NXentry,
    NXfield,
    NXgroup,
    NXinstrument,
    NXmonochromator,
    NXnote,
    NXparameters,
    NXprocess,
    NXreflections,
    NXsample,
    NXsource,
    NXtransformations
)
import numpy as np
from typing import List

from pyrs.utilities.pydantic_transition import validate_call_

REQUIRED_LOGS: List[str] = []


class _TypeBehavior:
    # Avoid metaclass conflict if mixin were derived from `type` directly.
    
    def __call__(self, *args, **kwargs):
        # Allow calling the enum member to construct via the underlying type
        return self.value(*args, **kwargs)

    def is_instance(self, obj):
        return isinstance(obj, self.value)

    def is_subclass(self, cls):
        return issubclass(cls, self.value)

    def __str__(self):
        return self.value.__name__

        
class FIELD_DTYPE(_TypeBehavior, Enum):
    # HDF5 dataset types for various fields
    FLOAT_CONSTANT = np.float64
    FLOAT_DATA     = np.float32
    INT_DATA       = np.int32
    STRING         = h5py.string_dtype(encoding='utf-8')


CHUNK_SHAPE = (100,)    # Reasonable chunk size (tunable)
            
class REQUIRED_NAME(StrEnum):
    # These are *required* group or dataset names, as specified in the `NXstress` schema.
    
    # FIT/DIFFRACTOGRAM sub-fields:
    PEAK_PARAMETERS = 'peak_parameters'
    BACKGROUND_PARAMETERS = 'background_parameters'
    DGRAM_DIFFRACTOGRAM = 'diffractogram'
    DGRAM_DIFFRACTOGRAM_ERRORS = 'diffractogram_errors'
    DGRAM_FIT = 'fit'
    DGRAM_FIT_ERRORS = 'fit_errors'
    
    INSTRUMENT = 'instrument'
    BEAM = 'beam_intensity_profile'
    PEAKS = 'peaks'
            
class GROUP_NAME(StrEnum):
    # Group names: ordered by their appearance in the NXstress schema:
    #
    # -- Unless initialized from a `REQUIRED_NAME`, these may be modified as necessary.
    # -- In case of multiple group instances, the enum value here becomes the <base_name>,
    #    with the <instance number> or <tag> becoming a name suffix (see `group_naming_scheme` below).
    #
    
    # Multiple NXentry are allowed in case there are multiple reduced data sets:
    #   e.g. from the same input data set, using different optimal peak-fit combinations.
    ENTRY = ('entry', True, NXentry)

    INSTRUMENT = (REQUIRED_NAME.INSTRUMENT, False, NXinstrument)
    CALIBRATION = ('calibration', False, NXnote)
    SOURCE = ('source', False, NXsource)
    DETECTOR = ('detector', True, NXdetector)
    TRANSFORMATIONS = ('transformations', False, NXtransformations)
    BEAM = (REQUIRED_NAME.BEAM, False, NXbeam)
    MONOCHROMATOR = ('monochromator', False, NXmonochromator)

    SAMPLE_DESCRIPTION = ('sample', False, NXsample)
    
    # FIT (NXprocess) groups contain the reduced data (and associated metadata):
    #   there should be one FIT group corresponding to each detector mask.
    FIT   = ('fit', True, NXprocess)
    PEAK_PARAMETERS = (REQUIRED_NAME.PEAK_PARAMETERS, False, NXparameters)
    BACKGROUND_PARAMETERS = (REQUIRED_NAME.BACKGROUND_PARAMETERS, False, NXparameters)
    DIFFRACTOGRAM = ('diffractogram', False, NXdata)
    # DIFFRACTOGRAM sub-fields:
    DGRAM_DIFFRACTOGRAM = (REQUIRED_NAME.DGRAM_DIFFRACTOGRAM, False, NXfield)
    DGRAM_DIFFRACTOGRAM_ERRORS = (REQUIRED_NAME.DGRAM_DIFFRACTOGRAM_ERRORS, False, NXfield)
    DGRAM_FIT = (REQUIRED_NAME.DGRAM_FIT, False, NXfield)
    DGRAM_FIT_ERRORS = (REQUIRED_NAME.DGRAM_FIT_ERRORS, False, NXfield)


    # PEAKS (NXreflections) presents the canonical reduction result: there is only one per NXentry.
    PEAKS = (REQUIRED_NAME.PEAKS, False, NXreflections)
    
    ## OPTIONAL GROUPS, allowed by but not specified by the schema: ##
    
    # Including the input data allows all of the information for a reduction to be contained in one file.
    INPUT_DATA = ('input_data', False, NXdata)
    
    # Masks are added as a subgroup under the `INSTRUMENT` group:
    #   both <detector mask> and <solid-angle mask> are currently recognized,
    #   however the mask names must be distinct, because they're used as suffix tags
    #   when creating other group names.
    MASKS = ('masks', False, NXcollection)

    def __new__(cls, value, allowMultiple: bool, nxClass: NXgroup):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.allowMultiple = allowMultiple
        obj.nxClass = nxClass
        return obj

# `NXstress` records `peak_parameters` and `background_parameters` in distinct groups.
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
    if not isinstance(suffix, (int, str)):
      raise RuntimeError(f"`group_naming_scheme`: not implemented for suffix '{suffix}'")    
    
    tag = ''
    if isinstance(suffix, int) and suffix > 1\
        or isinstance(suffix, str) and suffix != DEFAULT_TAG:
      tag = f'_{suffix}'
    
    return f'{base_name}{tag}'
