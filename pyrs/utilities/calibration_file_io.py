# Class providing a series of static methods to work with files
from pyrs.utilities import checkdatatypes
from pyrs.core.instrument_geometry import AnglerCameraDetectorShift, AnglerCameraDetectorGeometry
import json


def read_calibration_json_file(calibration_file_name):
    """Import calibration file in json format

    Example:  input JSON
    {u'Lambda': 1.452,
    u'Rot_x': 0.0,
    u'Rot_y': 0.0,
    u'Rot_z': 0.0,
    u'Shift_x': 0.0,
    u'Shift_y': 0.0,
    u'Shift_z': 0.0,
    u'Status': 3,
    u'error_Lambda': 1.0829782933282927e-07,
    u'error_Rot_x': -1.0,
    u'error_Rot_y': -1.0,
    u'error_Rot_z': -1.0,
    u'error_Shift_x': -1.0,
    u'error_Shift_y': -1.0,
    u'error_Shift_z': -1.0}

    Parameters
    ----------
    calibration_file_name

    Returns
    -------
    ~tuple
        (AnglerCameraDetectorShift, AnglerCameraDetectorShift, float, float, int)
        detector position shifts as the calibration result,detector position shifts error from fitting
        status

    """

    # Check input
    checkdatatypes.check_file_name(calibration_file_name, True, False, False, 'Calibration JSON file')

    # Parse JSON file
    with open(calibration_file_name, 'r') as calib_file:
        calib_dict = json.load(calib_file)
    if calib_dict is None:
        raise RuntimeError('Failed to load JSON calibration file {}'.format(calibration_file_name))

    # Convert dictionary to AnglerCameraDetectorShift
    try:
        shift = AnglerCameraDetectorShift(shift_x=calib_dict['Shift_x'],
                                          shift_y=calib_dict['Shift_y'],
                                          shift_z=calib_dict['Shift_z'],
                                          rotation_x=calib_dict['Rot_x'],
                                          rotation_y=calib_dict['Rot_y'],
                                          rotation_z=calib_dict['Rot_z'])
    except KeyError as key_error:
        raise RuntimeError('Missing key parameter from JSON file {}: {}'.format(calibration_file_name, key_error))

    # shift error
    try:
        shift_error = AnglerCameraDetectorShift(shift_x=calib_dict['error_Shift_x'],
                                                shift_y=calib_dict['error_Shift_y'],
                                                shift_z=calib_dict['error_Shift_z'],
                                                rotation_x=calib_dict['error_Rot_x'],
                                                rotation_y=calib_dict['error_Rot_y'],
                                                rotation_z=calib_dict['error_Rot_z'])
    except KeyError as key_error:
        raise RuntimeError('Missing key parameter from JSON file {}: {}'.format(calibration_file_name, key_error))

    # Wave length
    try:
        wave_length = calib_dict['Lambda']
        wave_length_error = calib_dict['error_Lambda']
    except KeyError as key_error:
        raise RuntimeError('Missing wave length related parameter from JSON file {}: {}'
                           ''.format(calibration_file_name, key_error))

    # Calibration status
    try:
        status = calib_dict['Status']
    except KeyError as key_error:
        raise RuntimeError('Missing status parameter from JSON file {}: {}'.format(calibration_file_name, key_error))

    return shift, shift_error, wave_length, wave_length_error, status


def import_calibration_ascii_file(geometry_file_name):
    """
    import geometry set up file
    arm = 0.95
    cal::arm = 0.
    cal::shiftx = 0.
    cal::shifty = 0.1
    :param geometry_file_name:
    :return: calibration instance
    """
    checkdatatypes.check_file_name(geometry_file_name, True, False, False, 'Geometry configuration file in ASCII')

    # init output
    calibration_setup = AnglerCameraDetectorShift(0, 0, 0, 0, 0, 0)

    calibration_file = open(geometry_file_name, 'r')
    geom_lines = calibration_file.readlines()
    calibration_file.close()

    for line in geom_lines:
        line = line.strip()

        # skip empty or comment line
        if line == '' or line.startswith('#'):
            continue

        terms = line.replace('=', ' ').split()
        config_name = terms[0].strip().lower()
        config_value = float(terms[1].strip())

        if config_name == 'cal::shift_x':
            calibration_setup.center_shift_x = config_value
        elif config_name == 'cal::shift_y':
            calibration_setup.center_shift_y = config_value
        elif config_name == 'cal::arm':
            calibration_setup.arm_calibration = config_value
        elif config_name == 'cal::rot_x':
            calibration_setup.rotation_x = config_value
        elif config_name == 'cal::rot_y':
            calibration_setup.rotation_x = config_value
        elif config_name == 'cal::rot_z':
            calibration_setup.rotation_z = config_value
        else:
            raise RuntimeError(
                'Instrument geometry setup item {} is not recognized and supported.'.format(config_name))

    return calibration_setup


def import_instrument_setup(instrument_ascii_file):
    """Import instrument file in ASCII format

    Example:
      # comment
      arm = xxx  meter
      rows = 2048
      columns = 2048
      pixel_size_x = 0.00
      pixel_size_y = 0.00

    Parameters
    ----------
    instrument_ascii_file : str
        instrument file in plain ASCII format

    Returns
    -------
    AnglerCameraDetectorGeometry
        Instrument geometry setup for HB2B

    """
    checkdatatypes.check_file_name(instrument_ascii_file, False, True, False,
                                   'Instrument definition ASCII file')

    instr_file = open(instrument_ascii_file, 'r')
    setup_lines = instr_file.readlines()
    instr_file.close()

    # Init
    arm_length = detector_rows = detector_columns = pixel_size_x = pixel_size_y = None

    # Parse each line
    for line in setup_lines:
        line = line.strip()

        # skip empty and comment
        if line == '' or line.startswith('#'):
            continue

        terms = line.replace('=', ' ').split()
        arg_name = terms[0].strip().lower()
        arg_value = terms[1]

        if arg_name == 'arm':
            arm_length = float(arg_value)
        elif arg_name == 'rows':
            detector_rows = int(arg_value)
        elif arg_name == 'columns':
            detector_columns = int(arg_value)
        elif arg_name == 'pixel_size_x':
            pixel_size_x = float(arg_value)
        elif arg_name == 'pixel_size_y':
            pixel_size_y = float(arg_value)
        else:
            raise RuntimeError('Argument {} is not recognized'.format(arg_name))
    # END-FOR

    instrument = AnglerCameraDetectorGeometry(num_rows=detector_rows,
                                              num_columns=detector_columns,
                                              pixel_size_x=pixel_size_x,
                                              pixel_size_y=pixel_size_y,
                                              arm_length=arm_length,
                                              calibrated=False)

    return instrument


def write_calibration_to_json(shifts, shifts_error, wave_length, wave_lenngth_error,
                              calibration_status, file_name=None):
    """Write geometry and wave length calibration to a JSON file

    Parameters
    ----------

    Returns
    -------
    None
    """
    # Check inputs
    checkdatatypes.check_file_name(file_name, False, True, False, 'Output JSON calibration file')
    assert isinstance(shifts, AnglerCameraDetectorShift)
    assert isinstance(shifts_error, AnglerCameraDetectorShift)

    # Create calibration dictionary
    calibration_dict = shifts.convert_to_dict()
    calibration_dict['Lambda'] = wave_length

    calibration_dict.update(shifts_error.convert_error_to_dict())
    calibration_dict['error_Lambda'] = wave_lenngth_error

    calibration_dict.update({'Status': calibration_status})

    print('DICTIONARY:\n{}'.format(calibration_dict))

    with open(file_name, 'w') as outfile:
        json.dump(calibration_dict, outfile)
    print('[INFO] Calibration file is written to {}'.format(file_name))
