# A module contains a set of static methods to provide instrument geometry and data archiving knowledge of HB2B
from . import checkdatatypes


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
