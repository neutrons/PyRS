from typing import Optional

__all__ = ['to_float']


def to_float(name: str, value,
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
    if min_value is not None and max_value is not None and min_value > max_value:
        raise ValueError('Invalid range ({}, {}) specified for "{}"'.format(min_value, max_value, name))

    # validate the value is within range
    err_msg = []

    # check minimum
    if min_value is not None:
        if min_inclusive:
            if value < min_value:
                err_msg.append('below minimum')
        else:
            if value <= min_value:
                err_msg.append('below minimum')

    # check maximum
    if max_value is not None:
        if max_inclusive:
            if value > max_value:
                err_msg.append('above maximum')
        else:
            if value >= max_value:
                err_msg.append('above maximum')

    # having any error message indicates bad value
    if err_msg:
        err_msg = 'Variable "{}" value={} not in range ({}, {}): {}'.format(name, value, min_value, max_value,
                                                                            ' '.join(err_msg))
        raise ValueError(err_msg)
    else:  # return the value converted to a float
        return value
