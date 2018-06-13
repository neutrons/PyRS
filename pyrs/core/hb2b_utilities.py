# A module contains a set of static methods to provide instrument geometry and data archiving knowledge of HB2B
import rshelper


def get_hb2b_raw_data(ipts_number, exp_number):
    """
    get the archived HB2B raw data
    :param ipts_number:
    :param exp_number:
    :return:
    """
    # check inputs
    rshelper.check_int_variable('IPTS number', ipts_number, (1, None))
    rshelper.check_int_variable('Experiment number', exp_number, (1, None))

    raw_exp_file_name = '/HFIR/HB2B/IPTS-{0}/datafiles/{1}.h5'.format(ipts_number, exp_number)

    rshelper.check_file_name(raw_exp_file_name, check_exist=True, check_writable=False, is_dir=False)

    return raw_exp_file_name
