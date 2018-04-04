# PyRS static helper methods
import os


def check_file_name(file_name, check_exist=True, check_writable=False):
    """
    check whether an input file name is a string and whether it is a file or a file can be written to
    :param file_name:
    :param check_exist:
    :param check_writable:
    :return:
    """
    assert isinstance(file_name, str), 'Input file name {0}  must be a string but not a {1}.' \
                                       ''.format(file_name, str(file_name))

    if check_exist and os.path.exists(file_name) is False:
        raise RuntimeError('File {0} does not exist.'.format(file_name))

    if check_writable and os.access(file_name, os.W_OK):
        raise RuntimeError('File {0} is not writable.'.format(file_name))

    return