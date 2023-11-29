# PyRS static helper methods
import os
import numpy
from typing import Any, Iterable, Optional, Tuple
from . convertdatatypes import to_int


def check_bool_variable(var_name: str, bool_var: bool) -> None:
    '''check whether a variable is a bool'''
    assert isinstance(bool_var, bool), '{0} of value {1} shall be a bool but not a {2}.' \
                                       ''.format(var_name, bool_var, type(bool_var))


def check_dict(var_name: str, dict_var: dict) -> None:
    '''check whether a variable is a dictionary'''
    assert isinstance(dict_var, dict), '{0} (= {1}) must be a dict but not a {2}' \
                                       ''.format(var_name, dict_var, type(dict_var))


def check_file_name(file_name: str, check_exist: bool = True, check_writable: bool = False,
                    is_dir: bool = False, description: str = '') -> None:
    '''check whether an input file name is a string and whether it is a file or a file can be written to
    :exception RuntimeError: file does not meet the requirement
    :param file_name:
    :param check_exist:
    :param check_writable:
    :param is_dir:
    :param description: a description for file name
    '''
    # set note
    if len(description) == 0:
        description = 'File'

    if check_exist and not os.path.exists(file_name):
        cur_dir = os.getcwd()
        file_dir = os.path.dirname(file_name)
        files = os.listdir(file_dir)
        message = 'DEBUG: current dir: {}; file dir: {}; Files available: {}'.format(cur_dir, file_dir, files)
        print(message)
        raise RuntimeError('{} {} does not exist. FYI\n{}.'.format(description, file_name, message))

    if check_writable:
        if (os.path.exists(file_name)) and (not os.access(file_name, os.W_OK)):
            # file exists but cannot be  overwritten
            raise RuntimeError('{} {} exists but is not writable.'.format(description, file_name))
        elif not os.path.exists(file_name):
            # file does not exist and the directory is not writable
            dir_name = os.path.dirname(file_name)
            if dir_name == '':
                # current working dir
                dir_name = os.getcwd()
            if not os.access(dir_name, os.W_OK):
                raise RuntimeError('{} {} does not exist but directory {} is not writable.'
                                   ''.format(description, file_name, dir_name))

    if is_dir:
        assert os.path.isdir(file_name), 'Path {0} shall {1} be a directory and it is {2}.' \
            ''.format(file_name, os.path.isdir(file_name), is_dir)


def check_list(var_name: str, variable: Any, allowed_values: Optional[list] = None) -> None:
    '''check whether a variable is a list'''
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


def check_numpy_arrays(var_name: str, variables: Any, dimension: Optional[int], check_same_shape: bool) -> None:
    '''check numpy array or numpy arrays
    :param dimension: None for not checking dimension; Otherwise, a tuple can be compared with numpy.ndarray.shape
    '''
    if isinstance(variables, numpy.ndarray):
        variables = [variables]
        check_same_shape = False
    else:
        assert isinstance(variables, (list, tuple)), \
            'Variable {} (shall be an numpy arrays) {} must be given in form of numpy array, ' \
            'list or tuple but not {}'.format(var_name, variables, type(variables))

    for index, variable in enumerate(variables):

        # check whether each variable is a numpy array with desired dimension
        assert isinstance(variable, numpy.ndarray), '{0}-th element of variable {1} ({2}) must be an ndarray but not' \
                                                    ' a {3}'.format(index, var_name, variable, type(variable))
        if dimension is not None:
            dimension = to_int('ndarray dimension for {0}'.format(var_name), dimension, min_value=0)
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


def check_series(var_name: str, variable: Any, allowed_type: Any = None, size: Any = None) -> None:
    '''check whether the input is of type tuple or list or numpy array
    :param allowed_type: allowed type such as str, float, int.
    :param size: allowed size. None, list of integers or an integer
    '''
    check_string_variable('Variable name', var_name)

    assert isinstance(variable, (list, tuple, numpy.ndarray)), \
        '{} {} must be a list or tuple but not a {}'.format(var_name, variable, type(variable))

    # check size
    if size is not None:
        if isinstance(size, list):
            pass
        elif isinstance(size, int):
            size = [size]
        else:
            raise RuntimeError('check_sequence cannot accept size ({}) of type {}'
                               ''.format(size, type(size)))
        if len(variable) not in size:
            raise RuntimeError('Variable {} ({})has {} items not allowed by required {}'
                               ''.format(var_name, variable, len(variable), size))

    # skip if no type check is specified
    if allowed_type is not None:
        for i_var, var_i in enumerate(variable):
            assert isinstance(var_i, allowed_type), '{}-th variable {} must be a {} but not a {}' \
                                                    ''.format(i_var, var_i, allowed_type, type(var_i))
    # END-IF


def check_string_variable(var_name: str, variable: str, allowed_values: Optional[Iterable] = None,
                          allow_empty: bool = True) -> None:
    '''check whether an input variable is a float
    :except AssertionError:
    :except ValueError:
    :param allowed_values: list of strings
    :param allow_empty: Flag to allow empty string
    '''
    if isinstance(allowed_values, list):
        if variable not in allowed_values:
            if len(variable) == 0:
                err_msg = '{} (as an EMPTY STRING) is not found in allowed value list {}' \
                          ''.format(var_name, allowed_values)
            else:
                err_msg = '{} {} is not found in allowed value list {}' \
                          ''.format(var_name, variable, allowed_values)
            raise ValueError(err_msg)
    elif allowed_values is not None:
        raise RuntimeError('Allowed values {} must be given in a list but not a {}'
                           ''.format(allowed_values, type(allowed_values)))

    # Not allow empty
    if not allow_empty and len(variable) == 0:
        raise RuntimeError('Variable "{}" is not allowed to be an empty string'.format(var_name))


def check_type(var_name: str, variable: Any, var_type: type) -> None:
    '''Check a variable against an arbitrary type'''
    assert isinstance(variable, var_type), '{} {} must be of type {} but not a {}' \
                                           ''.format(var_name, variable, var_type, type(variable))


def check_tuple(var_name: str, variable: Tuple, tuple_size: Optional[int] = None) -> None:
    '''check whether a variable is a tuple.  As an option, the tuple size can be checked too.'''
    if tuple_size is not None:
        tuple_size = to_int('Tuple size', tuple_size, min_value=0)
        assert len(variable) == tuple_size, 'Tuple {0}\'s size {1} must be equal to {2} as user specifies.' \
                                            ''.format(variable, len(variable), tuple_size)
