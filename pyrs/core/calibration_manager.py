class CalibrationManager(object):
    """
    A class to handle all the calibration files
    calibration shall be related to date (run cycle), wave length and etc.
    """
    def __init__(self,  calib_lookup_table_file=None):
        """
        Initialization for calibration manager
        Parameters
        ----------
        calib_lookup_table_file: str
            calibration table file to in order not to scan the disk and save time
        """
        self._cal_dict = dict()  # dict[wavelength, date] = param_dict
        # param_dict[motor position] = calibrated value

        return

    def get_calibration(self, ipts_number, run_number):
        """
        Get calibration in memory
        Parameters
        ----------
        ipts_number
        run_number

        Returns
        -------
        None
        """

        return

    def locate_calibration_file(self, ipts_number, run_number):
        """
        Locate calibration file in the /HFIR/HB2B/shared/
        Parameters
        ----------
        ipts_number
        run_number

        Returns
        -------
        None
        """
        return

    def show_calibration_table(self):
        """
        Show calibration result in a table
        Returns
        -------
        None
        """
        return
# END-DEF-CLASS (CalibrationManager)
