# TODO - 20181130 - Implement this class
# A module contains a set of static methods to provide instrument geometry and data archiving knowledge of HB2B
import checkdatatypes


def get_hb2b_raw_data(ipts_number, run_number):
    """
    get the archived HB2B raw data
    :param ipts_number:
    :param run_number:
    :return:
    """
    # check inputs
    checkdatatypes.check_int_variable('IPTS number', ipts_number, (1, None))
    checkdatatypes.check_int_variable('Run number', run_number, (1, None))

    raw_exp_file_name = '/HFIR/HB2B/IPTS-{0}/datafiles/{1}.h5'.format(ipts_number, run_number)

    checkdatatypes.check_file_name(raw_exp_file_name, check_exist=True, check_writable=False, is_dir=False)

    return raw_exp_file_name


def get_hydra_project_file(ipts_number, run_number):
    """
    get the archived HB2B raw data
    :param ipts_number: IPTS number (int)
    :param run_number: Run number (int)
    :return:
    """
    # check inputs
    checkdatatypes.check_int_variable('IPTS number', ipts_number, (1, None))
    checkdatatypes.check_int_variable('Run number', run_number, (1, None))

    hydra_file_name = '/HFIR/HB2B/IPTS-{0}/shared/reduced_files/HB2B_{1}.hdf'.format(ipts_number, run_number)

    try:
        checkdatatypes.check_file_name(hydra_file_name, check_exist=True, check_writable=False, is_dir=False)
    except RuntimeError as run_error:
        print('[ERROR] Unable to find Hydra project file {} due to {}'.format(hydra_file_name, run_error))
        return None

    return hydra_file_name


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
