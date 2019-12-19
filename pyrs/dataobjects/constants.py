__all__ = ['HidraConstants']


class HidraConstants(object):
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
    SOLID_ANGLE_MASK = 'solid angle'

    # constants about peak fitting
    PEAK_PROFILE = 'peak profile'
    PEAKS = 'peaks'  # main entry for fitted peaks' parameters
    PEAK_FIT_CHI2 = 'chi2'
    PEAK_PARAMS = 'parameters'  # peak parameter values
    PEAK_PARAMS_ERROR = 'fitting error'  # peak parameters' fitting error
    PEAK_PARAM_NAMES = 'parameter names'  # peak parameter names
    PEAK_COM = 'C.O.M'  # peak's center of mass
    BACKGROUND_TYPE = 'background type'

    # Special sample logs
    SUB_RUN_DURATION = 'sub-run duration'
