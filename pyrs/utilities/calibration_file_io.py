# Class providing a series of static methods to work with files
from pyrs.utilities import checkdatatypes
from pyrs.core.instrument_geometry import AnglerCameraDetectorShift
import h5py


def import_calibration_info_file(cal_info_file):
    """
    import calibration information file
    :param cal_info_file:
    :return:
    """
    checkdatatypes.check_file_name(cal_info_file, check_exist=True, check_writable=False,
                                   is_dir=False, description='HB2B calibration information file')

    cal_info_table = dict()

    cal_file = h5py.File(cal_info_file, mdoe='r')
    for wavelength_entry in cal_file:
        # each wave length
        entry_name = wavelength_entry.name
        cal_info_table[entry_name] = dict()

        # TODO - NEED TO FIND OUT HOW TO PARSE 2 Column Value
        for cal_date, cal_file in wavelength_entry.value:
            cal_info_table[entry_name][cal_date] = cal_file
        # END-FOR
    # END-FOR

    cal_file.close()

    return cal_info_table


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
    calibration_setup = AnglerCameraDetectorShift()

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
    """
    Import instrument file in ASCII format
    Example:
      # comment
      arm = xxx  meter
      rows = 2048
      columns = 2048
      pixel_size_x = 0.00
      pixel_size_y = 0.00
    :param instrument_ascii_file:
    :return:
    """
    checkdatatypes.check_file_name(instrument_ascii_file, False, True, False,
                                   'Instrument definition ASCII file')

    instr_file = open(instrument_ascii_file, 'r')
    setup_lines = instr_file.readlines()
    instr_file.close()

    raise RuntimeError('Need method to create an "InstrumentSetup"')
    instrument = None  # TODO should be: instrument = InstrumentSetup()
    for line in setup_lines:
        line = line.strip()

        # skip empty and comment
        if line == '' or line.startswith('#'):
            continue

        terms = line.replace('=', ' ').split()
        arg_name = terms[0].strip().lower()
        arg_value = terms[1]

        if arg_name == 'arm':
            instrument.arm_length = float(arg_value)
        elif arg_name == 'rows':
            instrument.detector_rows = int(arg_value)
        elif arg_name == 'columns':
            instrument.detector_columns = int(arg_value)
        elif arg_name == 'pixel_size_x':
            instrument.pixel_size_x = float(arg_value)
        elif arg_name == 'pixel_size_y':
            instrument.pixel_size_y = float(arg_value)
        else:
            raise RuntimeError('Argument {} is not recognized'.format(arg_name))
    # END-FOR

    return instrument


def write_calibration_ascii_file(two_theta, arm_length, calib_config, note, geom_file_name):
    """ write a geometry ascii file as standard
    :param two_theta:
    :param arm_length:
    :param geom_file_name:
    :param note:
    :return:
    """
    checkdatatypes.check_file_name(geom_file_name, False, True, False, 'Output geometry configuration file in ASCII')

    wbuf = '# {}\n'.format(note)
    wbuf += '2theta = {}\n'.format(two_theta)
    wbuf += 'arm = {} meter\n'.format(arm_length)
    wbuf += 'cal::shift_x = {} meter\n'.format(calib_config.shift_x)
    wbuf += 'cal::shift_y = {} meter\n'.format(calib_config.shift_y)
    wbuf += 'cal::arm = {} meter\n'.format(calib_config.arm_calibration)
    wbuf += 'cal::rot_x = {} degree\n'.format(calib_config.tilt_x)
    wbuf += 'cal::rot_y = {} degree\n'.format(calib_config.rotation_y)
    wbuf += 'cal::rot_z = {} degree\n'.format(calib_config.spin_z)

    out_file = open(geom_file_name, 'w')
    out_file.write(wbuf)
    out_file.close()

    return


class ResidualStressCalibrationFile(object):
    """
    a dedicated file import/export
    """

    def __init__(self, cal_file_name=None, read_only=False):
        """
        initialization.
        :param cal_file_name: calibration.  If false, then in the writing out mode
        :param read_only: if True, then read only. Otherwise, read/write mode
        """
        # init some parameters
        self._h5_file = None  # HDF5 handler
        self._geometry_calibration = AnglerCameraDetectorShift()
        self._calibration_date = ''

        # check
        checkdatatypes.check_bool_variable('Calibration file read only mode', read_only)
        if cal_file_name:
            # read or read/write mode
            checkdatatypes.check_file_name(cal_file_name, check_exist=True, check_writable=not read_only,
                                           is_dir=False, description='HB2B calibration file')
            self._cal_file_name = cal_file_name
            if read_only:
                self._file_mode = 'r'
            else:
                self._file_mode = 'r+'

            self._import_h5_calibration(self._cal_file_name)
        else:
            # write mode
            self._cal_file_name = None
            self._file_mode = 'w'
        # END-IF-ELSE

        return

    def _import_h5_calibration(self, h5_name):
        """
        import calibration file in HDF5 format
        :param h5_name:
        :return:
        """
        self._h5_file = h5py.File(h5_name, self._file_mode)

        return

    def close_file(self):
        """
        close h5 file
        :return:
        """
        if self._h5_file:
            self._h5_file.close()
        self._h5_file = None

    def retrieve_calibration_date(self):
        """ get the starting date of the calibration shall be applied
        :return:
        """
        if not self._h5_file:
            raise RuntimeError('No calibration file (*.h5) has been specified and thus imported in '
                               'constructor')

        # get the calibrated geometry parameters
        calib_param_entry = self._h5_file['calibration']

        # TODO - 20181210 - Need a prototype file to continue
        self._geometry_calibration.two_theta0 = calib_param_entry['2theta0'].values[0]
        raise NotImplementedError('Functionality for setting "center_shift_x" does not exist')
        # self._geometry_calibration.center_shift_x = blabla

        # get the date from the calibration file inside?


# END-CLASS


def update_calibration_info_file(cal_info_file, cal_info_table, append):
    """ Search archive in order to keep calibration up-to-date
    if in append mode, the additional information will be written to an existing calibration information hdf5 file
    otherwise, From scratch, a calibration information file will be created
    :param cal_info_file:
    :param cal_info_table: calibration information to append to calibration information file
    :param append: flag whether the mode is append or new
    :return:
    """
    # check inputs
    if append:
        checkdatatypes.check_file_name(cal_info_file, True, True, False, 'Calibration information file to create')
    else:
        checkdatatypes.check_file_name(cal_info_file, False, True, False, 'Calibration information file to append')
    checkdatatypes.check_dict('Calibration information table', cal_info_table)
    checkdatatypes.check_bool_variable('Append mode', append)

    # open file
    if append:
        cal_info_file = h5py.File(cal_info_file, mdoe='rw')
    else:
        cal_info_file = h5py.File(cal_info_file, mode='w')

    # write to file
    for wavelength_entry in cal_info_table:
        if wavelength_entry not in cal_info_file:
            # TODO fix this
            # cal_info_file[wavelength_entry] = whatever
            raise RuntimeError('encountered unknown wavelength_entry: {}'.format(wavelength_entry))

        for cal_date in cal_info_table[wavelength_entry]:
            cal_file_name = cal_info_table[wavelength_entry][cal_date]
            cal_info_file[wavelength_entry].append((cal_date, cal_file_name))
        # END-FOR
    # END-FOR

    # close
    cal_info_file.close()

    return
