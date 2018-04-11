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


def check_int_variable(var_name, variable, value_range):
    """
    check whether an input variable is an integer
    :except AssertionError:
    :except ValueError:
    :param var_name:
    :param variable:
    :param value_range: if not None, then must be a 2 tuple as [min, max)
    :return:
    """
    assert isinstance(var_name, int), '{0} {1} must be an integer but not a {2}'\
        .format(var_name, variable, type(variable))

    if value_range is not None:
        assert len(value_range) == 2, '{0}\' value range {1} must be either a None or have 2 elements as [min, max)' \
                                      ''.format(var_name, value_range)

        min_val = value_range[0]
        max_val = value_range[1]
        if (min_val is not None and variable < min_val) or (max_val is not None and variable >= max_val):
            raise ValueError('{0} (= {1}) is out of range [{2}, {3})'.format(var_name, variable, min_val, max_val))

    return


def check_float_variable(var_name, variable, value_range):
    """
    check whether an input variable is a float
    :except AssertionError:
    :except ValueError:
    :param var_name:
    :param variable:
    :param value_range: if not None, then must be a 2 tuple as [min, max)
    :return:
    """
    assert isinstance(var_name, float), '{0} {1} must be a float but not a {2}'\
        .format(var_name, variable, type(variable))

    if value_range is not None:
        assert len(value_range) == 2, '{0}\' value range {1} must be either a None or have 2 elements as [min, max)' \
                                      ''.format(var_name, value_range)

        min_val = value_range[0]
        max_val = value_range[1]
        if (min_val is not None and variable < min_val) or (max_val is not None and variable >= max_val):
            raise ValueError('{0} (= {1}) is out of range [{2}, {3})'.format(var_name, variable, min_val, max_val))

    return


def check_string_variable(var_name, variable):
    """
    check whether an input variable is a float
    :except AssertionError:
    :except ValueError:
    :param var_name:
    :param variable:
]    :return:
    """
    assert isinstance(var_name, str), '{0} {1} must be a string but not a {2}'\
        .format(var_name, variable, type(variable))

    return