# PyRS static helper methods
import os
import numpy


def check_bool_variable(var_name, bool_var):
    """
    check whether a variable is a bool
    :param var_name:
    :param bool_var:
    :return:
    """
    check_string_variable('Variable name', var_name)
    assert isinstance(bool_var, bool), '{0} of value {1} shall be a bool but not a {2}.' \
                                       ''.format(var_name, bool_var, type(bool_var))

    return True


def check_dict(var_name, dict_var):
    """
    check whether a variable is a dictionary
    :param var_name:
    :param dict_var:
    :return:
    """
    check_string_variable('var_name', var_name)
    assert isinstance(dict_var, dict), '{0} (= {1}) must be a dict but not a {2}' \
                                       ''.format(var_name, dict_var, type(dict_var))

    return


def check_file_name(file_name, check_exist=True, check_writable=False, is_dir=False):
    """
    check whether an input file name is a string and whether it is a file or a file can be written to
    :param file_name:
    :param check_exist:
    :param check_writable:
    :param is_dir:
    :return:
    """
    assert isinstance(file_name, str), 'Input file name {0}  must be a string but not a {1}.' \
                                       ''.format(file_name, type(file_name))

    if check_exist and os.path.exists(file_name) is False:
        raise RuntimeError('File {0} does not exist.'.format(file_name))

    if check_writable:
        if os.path.exists(file_name) and os.access(file_name, os.W_OK) is False:
            # file exists but cannot be  overwritten
            raise RuntimeError('File {0} exists but is not writable.'.format(file_name))
        elif os.path.exists(file_name) is False:
            # file does not exist and the directory is not writable
            dir_name = os.path.dirname(file_name)
            if dir_name == '':
                # current working dir
                dir_name = os.getcwd()
            if os.access(dir_name, os.W_OK) is False:
                raise RuntimeError('File {0} does not exist but directory {1} is not writable.'
                                   ''.format(file_name, dir_name))
        # END-IF-ELIF
    # END-IF

    check_bool_variable('Flag for input string is a directory', is_dir)
    assert os.path.isdir(file_name) == is_dir, 'Path {0} shall {1} be a directory and it is {2}.' \
                                               ''.format(file_name, os.path.isdir(file_name), is_dir)

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
    check_string_variable('var_name', var_name)
    assert isinstance(variable, int), '{0} {1} must be an integer but not a {2}'\
        .format(var_name, variable, type(variable))

    if value_range is not None:
        assert len(value_range) == 2, '{0}\' value range {1} must be either a None or have 2 elements as [min, max)' \
                                      ''.format(var_name, value_range)

        min_val = value_range[0]
        max_val = value_range[1]
        assert min_val is None or isinstance(min_val, int), 'Minimum value {0} of value range {1} must be either None' \
                                                            ' or integer but not {2}' \
                                                            ''.format(min_val, value_range, type(min_val))
        assert max_val is None or isinstance(max_val, int), 'Maximum value {0} of value range {1} must be either None' \
                                                            ' or integer but not {2}' \
                                                            ''.format(max_val, value_range, type(max_val))
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
    check_string_variable('var_name', var_name)
    assert isinstance(variable, float) or isinstance(variable, int), '{0} {1} must be a float but not a {2}'\
        .format(var_name, variable, type(variable))

    if value_range is not None:
        assert len(value_range) == 2, '{0}\' value range {1} must be either a None or have 2 elements as [min, max)' \
                                      ''.format(var_name, value_range)

        min_val = value_range[0]
        max_val = value_range[1]
        if (min_val is not None and variable < min_val) or (max_val is not None and variable >= max_val):
            raise ValueError('{0} (= {1}) is out of range [{2}, {3})'.format(var_name, variable, min_val, max_val))

    return


def check_list(var_name, variable, allowed_values=None):
    """
    check whether a variable is a list
    :param var_name:
    :param variable:
    :param allowed_values:
    :return:
    """
    check_string_variable('var_name', var_name)
    if allowed_values is not None:
        assert isinstance(allowed_values, list), 'Allowed values (other than None) {0} must be given in' \
                                                 'a list but not {1}'.format(allowed_values,
                                                                             type(allowed_values))
    # END-IF

    assert isinstance(variable, list), '{0} {1} must be an instance of list but not a {2}' \
                                       ''.format(var_name, variable, type(variable))

    if allowed_values is not None:
        for item in variable:
            if item not in allowed_values:
                raise RuntimeError('In {0} item {1} is not in allowed list {2}'
                                   ''.format(var_name, item, allowed_values))

    return


def check_numpy_arrays(var_name, variables, dimension, check_same_shape):
    """ check numpy array or numpy arrays
    :param var_name: 
    :param variables: 
    :param dimension: 
    :param check_same_shape:
    :return: 
    """
    check_string_variable('Variable name', var_name)
    check_bool_variable('Flag to check arrays having same shape', check_same_shape)

    if isinstance(variables, numpy.ndarray):
        variables = [variables]
        check_same_shape = False
    else:
        assert isinstance(variables, list) or isinstance(variables, tuple), 'Numpy arrays {0} must be given by either' \
                                                                            ' list or tuple but not {1}' \
                                                                            ''.format(variables, type(variables))

    for index, variable in enumerate(variables):
        # check whether each variable is a numpy array with desired dimension
        assert isinstance(variable, numpy.ndarray), '{0}-th element of variable {1} ({2}) must be an ndarray but not ' \
                                                    'a {3}'.format(index, var_name, variable, type(variable))
        if dimension is not None:
            check_int_variable('ndarray dimension for {0}'.format(var_name), dimension, [0, None])
            assert len(variable.shape) == dimension, '{0}-th ndarray of variable {1} must be of {2}-dimension but ' \
                                                     'not have a shape as {3}.'.format(index, var_name, dimension,
                                                                                       variable.shape)
        # END-IF
    # END-FOR

    if check_same_shape:
        # check whether all the arrays have the same shape
        shape0 = variables[0].shape
        for index in range(1, len(variables)):
            assert shape0 == variables[index].shape, '0-th ndarray (shpae = {0}) must have the same shape as ' \
                                                     '{1}-th ndarray (shape = {2}'.format(shape0, index,
                                                                                          variables[index].shape)
    # END-IF

    return


def check_sequence(var_name, variable, allowed_type=None):
    """ check whether the input is of type tuple or list
    :param var_name:
    :param variable:
    :param allowed_type: allowed type such as str, float, int.
    :return:
    """
    check_string_variable('Variable name', var_name)

    assert isinstance(variable, list) or isinstance(variable, tuple),\
        '{} {} must be a list or tuple but not a {}'.format(var_name, variable, type(variable))

    # skip if no type check is specified
    if allowed_type is None:
        return

    for i_var, var_i in enumerate(variable):
        assert isinstance(var_i, allowed_type), '{}-th variable {} must be a {} but not a {}' \
                                                ''.format(i_var, var_i, allowed_type, type(var_i))

    return


def check_string_variable(var_name, variable, allowed_values=None):
    """
    check whether an input variable is a float
    :except AssertionError:
    :except ValueError:
    :param var_name:
    :param variable:
    :param allowed_values: list of strings
    :return:
    """
    assert isinstance(var_name, str), 'Variable name {0} must be a string but not a {1}'\
        .format(var_name, type(var_name))

    assert isinstance(variable, str), '{0} {1} must be a string but not a {2}'\
        .format(var_name, variable, type(variable))

    if isinstance(allowed_values, list):
        if variable not in allowed_values:
            raise ValueError('{} {} is not found in allowed value list {}'
                             ''.format(var_name, variable, allowed_values))
    elif allowed_values is not None:
        raise RuntimeError('Allowed values {} must be given in a list but not a {}'
                           ''.format(allowed_values, type(allowed_values)))

    return


def check_tuple(var_name, variable, tuple_size=None):
    """
    check whether a variable is a tuple.  As an option, the tuple size can be checked too.
    :param var_name:
    :param variable:
    :param tuple_size:
    :return:
    """
    check_string_variable('Variable name', var_name)

    assert isinstance(variable, tuple), '{0} (= {1}) must be a tuple but not a {2}' \
                                        ''.format(var_name, variable, type(variable))

    if tuple_size is not None:
        check_int_variable('Tuple size', tuple_size, value_range=[0, None])
        assert len(variable) == tuple_size, 'Tuple {0}\'s size {1} must be equal to {2} as user specifies.' \
                                            ''.format(variable, len(variable), tuple_size)

    return
