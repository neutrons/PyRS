from typing import Any, Optional, Union

__all__ = ['to_float', 'to_int']


def __check_range(name: str, value: Union[float, int],
                  min_value: Optional[Union[float, int]] = None,
                  max_value: Optional[Union[float, int]] = None,
                  min_inclusive: bool = True,
                  max_inclusive: bool = False) -> None:
    # verify valid range
    if min_value is not None and max_value is not None and min_value > max_value:
        raise ValueError('Invalid range ({}, {}) specified for "{}"'.format(min_value, max_value, name))

    # validate the value is within range
    errors = []

    # check minimum
    if min_value is not None:
        if min_inclusive:
            if value < min_value:
                errors.append('below minimum')
        else:
            if value <= min_value:
                errors.append('below minimum')

    # check maximum
    if max_value is not None:
        if max_inclusive:
            if value > max_value:
                errors.append('above maximum')
        else:
            if value >= max_value:
                errors.append('above maximum')

    # having any error message indicates bad value
    if errors:
        err_msg = 'Variable "{}" value={} not in range ({}, {}): {}'.format(name, value, min_value, max_value,
                                                                            ' '.join(errors))
        raise ValueError(err_msg)


def to_int(name: str, value: Any,
           min_value: Optional[int] = None,
           max_value: Optional[int] = None,
           min_inclusive: bool = True,
           max_inclusive: bool = False) -> int:
    # first convert the value to an integer or give a better exception
    try:
        value = int(value)
    except ValueError as e:
        raise TypeError('Variable "{}"'.format(name)) from e
    except TypeError as e:
        raise TypeError('Variable "{}"'.format(name)) from e

    # convert the range to integers
    min_value = int(min_value) if (min_value is not None) else None
    max_value = int(max_value) if (max_value is not None) else None

    # verify valid range
    __check_range(name, value, min_value, max_value, min_inclusive, max_inclusive)
    return value


def to_float(name: str, value: Any,
             min_value: Optional[float] = None,
             max_value: Optional[float] = None,
             min_inclusive: bool = True,
             max_inclusive: bool = False) -> float:
    # first convert the value to a float or give a better exception
    try:
        value = float(value)
    except ValueError as e:
        raise TypeError('Variable "{}"'.format(name)) from e
    except TypeError as e:
        raise TypeError('Variable "{}"'.format(name)) from e

    # convert the range to floats
    min_value = float(min_value) if (min_value is not None) else None
    max_value = float(max_value) if (max_value is not None) else None

    # verify valid range
    __check_range(name, value, min_value, max_value, min_inclusive, max_inclusive)
    return value
