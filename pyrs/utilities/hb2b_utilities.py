# TODO - 20181130 - Implement this class
# A module contains a set of static methods to provide instrument geometry and data archiving knowledge of HB2B
import checkdatatypes
import file_utilities



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


def get_hb2b_raw_data(ipts_number, exp_number):
    """
    get the archived HB2B raw data
    :param ipts_number:
    :param exp_number:
    :return:
    """
    # check inputs
    checkdatatypes.check_int_variable('IPTS number', ipts_number, (1, None))
    checkdatatypes.check_int_variable('Experiment number', exp_number, (1, None))

    raw_exp_file_name = '/HFIR/HB2B/IPTS-{0}/datafiles/{1}.h5'.format(ipts_number, exp_number)

    checkdatatypes.check_file_name(raw_exp_file_name, check_exist=True, check_writable=False, is_dir=False)

    return raw_exp_file_name
