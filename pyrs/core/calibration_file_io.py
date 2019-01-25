# Class providing a series of static methods to work with files
import os
import platform
import time
import checkdatatypes
import h5py
import hb2b_setup
import numpy as np


class ResidualStressInstrumentCalibration(object):
    """
    A class to handle and save intrument geometry calibration information
    """
    def __init__(self):
        """
        initialize
        """
        self.shift_x = 0.
        self.shift_y = 0.
        self.two_theta0 = 0.
        self.r_shift = 0.
        self.tilt_x = 0.   # in X-Z plane
        self.tilt_y = 0.   # along Y axis (vertical)
        self.spin_z = 0.   # rotation from center

        # Need data from client to finish this
        self.calibrated_wave_length = {'Si001': 1.00}


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
        self._geometry_calibration = ResidualStressInstrumentCalibration()
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

        return

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
        self._geometry_calibration.shift_x = blabla
        # blabla

        # get the date from the calibration file inside?

        return


# END-CLASS


class CalibrationManager(object):
    """
    A class to handle all the calibration files
    calibration shall be related to date (run cycle), wave length and etc
    """
    def __init__(self,  calib_lookup_table_file=None):
        """
        initialization for calibration manager
        :param calib_lookup_table_file: calibration table file to in order not to scan the disk and save time
        """
        self._cal_dict = dict()  # dict[wavelength, date] = param_dict
                                 # param_dict[motor position] = calibrated value

        return

    def get_calibration(self, ipts_number, run_number):
        """ get calibration in memory
        :param ipts_number:
        :param run_number:
        :return:
        """
        return

    def locate_calibration_file(self, ipts_number, run_number):
        """

        :param ipts_number:
        :param run_number:
        :return:
        """
        return

    def show_calibration_table(self):
        """

        :return:
        """


# END-DEF-CLASS (CalibrationManager)






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
            cal_info_file[wavelength_entry] = whatever

        for cal_date in cal_info_table[wavelength_entry]:
            cal_file_name = cal_info_table[wavelength_entry][cal_date]
            cal_info_file[wavelength_entry].append((cal_date, cal_file_name))
        # END-FOR
    # END-FOR

    # close
    cal_info_file.close()

    return


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





