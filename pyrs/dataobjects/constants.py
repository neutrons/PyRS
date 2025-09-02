import numpy as np

__all__ = ['HidraConstants', 'DEFAULT_POINT_RESOLUTION']

# two points in real space separated by less than this amount (in mili meters) are considered the same point
DEFAULT_POINT_RESOLUTION = 0.01
NOT_MEASURED_NUMPY = np.nan


class HidraConstants:
    """
    Constants used for Hidra project file, workspace and related dictionary
    """
    RAW_DATA = 'raw data'
    REDUCED_DATA = 'reduced diffraction data'
    REDUCED_MAIN = 'main'   # default reduced data
    SUB_RUNS = 'sub-runs'
    CALIBRATION = 'calibration'
    SAMPLE_LOGS = 'logs'
    INSTRUMENT = 'instrument'
    GEOMETRY_SETUP = 'geometry setup'
    DETECTOR_PARAMS = 'detector'
    TWO_THETA = '2theta'
    L2 = 'L2'

    MONO = 'monochromator setting'
    WAVELENGTH = 'wave length'

    # Efficiency
    DETECTOR_EFF = 'efficiency calibration'
    RUN = 'run number'

    # Masks
    MASK = 'mask'  # main entry name of mask
    DETECTOR_MASK = 'detector'
    DEFAULT_MASK = '_DEFAULT_'
    SOLID_ANGLE_MASK = 'solid angle'

    # constants about peak fitting
    PEAK_PROFILE = 'peak profile'
    PEAKS = 'peaks'  # main entry for fitted peaks' parameters
    PEAK_FIT_CHI2 = 'chi2'
    PEAK_PARAMS = 'parameters'  # peak parameter values
    PEAK_PARAMS_ERROR = 'fitting error'  # peak parameters' fitting error
    PEAK_PARAM_NAMES = 'parameter names'  # peak parameter names
    EFFECTIVE_PEAK_PARAMS = 'effective parameters'  # effective peak parameter values
    EFFECTIVE_PEAK_PARAMS_ERROR = 'effective fitting error'  # effective peak parameters' fitting error
    PEAK_COM = 'C.O.M'  # peak's center of mass
    BACKGROUND_TYPE = 'background type'
    EXCLUDE_PEAKS = 'exclude'

    # constants for strain
    D_REFERENCE = 'd reference'   # reference peak position in d-spacing for strain calculation
    D_REFERENCE_ERROR = 'd reference error'

    # Special sample logs
    SUB_RUN_DURATION = 'sub-run duration'  # units in seconds
    SAMPLE_COORDINATE_NAMES = ('vx', 'vy', 'vz')
    SAMPLE_NAME = 'SampleName'
    SAMPLE_DESCRIPTION = 'SampleDescription'
    CHEMICAL_FORMULA = 'chemical formula'
    TEMPERATURE = 'temperature'
    STRESS_FIELD = 'stress field'
    STRESS_FIELD_DIRECTION = 'stress field direction'
    
