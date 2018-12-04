# Class providing a series of static methods to work with files
import os
import platform
import time
import checkdatatypes
import h5py
import hb2b_setup


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
        # check
        checkdatatypes.check_bool_variable('Calibration file read only mode', read_only)
        if cal_file_name:
            # read or read/write mode
            checkdatatypes.check_file_name(cal_file_name, check_exist=True, check_writable=not read_only,
                                           is_dir=False, description='HB2B calibration file')
            self._cal_file_name = True
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

    def retrieve_calibration_date(self):
        """ get the starting date of the calibration shall be applied
        :return:
        """
        if self._cal_file_name is None:
            raise RuntimeError('blabla')

        # TODO - 20181203 - NOW!

        # get the date from the calibration file inside?

        return ''


# END-CLASS


def check_creation_date(file_name):
    """
    check the create date (year, month, date) for a file
    :except RuntimeError: if the file does not exist
    :param file_name: 
    :return: 
    """
    checkdatatypes.check_file_name(file_name, check_exist=True)

    # get the creation date in float (epoch time)
    if platform.system() == 'Windows':
        # windows not tested
        epoch_time = os.path.getctime(file_name)
    else:
        # mac osx/linux
        stat = os.stat(file_name)
        try:
            epoch_time = stat.st_birthtime
        except AttributeError:
            # We're probably on Linux. No easy way to get creation dates here,
            # so we'll settle for when its content was last modified.
            epoch_time = stat.st_mtime
        # END-TRY
    # END-IF-ELSE

    # convert epoch time to a string as YYYY-MM-DD
    file_create_time = time.localtime(epoch_time)
    file_create_time_str = time.strftime('%Y-%m-%d', file_create_time)

    return file_create_time_str


def is_calibration_dir(cal_sub_dir_name):
    """
    check whether the directory name is an allowed calibration directory name for HB2B
    :param cal_sub_dir_name:
    :return:
    """
    checkdatatypes.check_file_name(cal_sub_dir_name, check_exist=True, check_writable=False,
                                   is_dir=True, description='Directory for calibration files')

    dir_base_name = os.path.basename(cal_sub_dir_name)
    return dir_base_name in hb2b_setup



def scan_calibration_in_archive():
    """
    search the archive (/HFIR/HB2B/shared/CALIBRATION/) to create a table,
    which can be used to write the calibration information file
    :return:
    """
    calib_info_table = dict()

    calib_root_dir = '/HFIR/HB2B/CALIBRATION/'

    wavelength_dir_names = os.listdir(calib_root_dir)
    for wavelength_dir in wavelength_dir_names:
        # skip non-relevant directories
        wavelength_dir = os.path.join(calib_root_dir, wavelength_dir)
        if not is_calibration_dir(wavelength_dir):
            continue

        calib_info_table[wavelength_dir] = dict()
        cal_file_names = os.listdir(wavelength_dir)
        for cal_file_name in cal_file_names:
            calib_date = ResidualStressCalibrationFile(cal_file_name).retrieve_calibration_date()
            calib_info_table[wavelength_dir][calib_date] = cal_file_name
        # END-FOR
    # END-FOR

    return calib_info_table


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



# testing
print (check_creation_date('__init__.py'))
