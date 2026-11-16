from typing import Any, Optional, Union

import numpy as np

__all__ = ["is_text_array", "to_float", "to_int", "to_text", "to_text_array"]


def __check_range(
    name: str,
    value: Union[float, int],
    min_value: Optional[Union[float, int]] = None,
    max_value: Optional[Union[float, int]] = None,
    min_inclusive: bool = True,
    max_inclusive: bool = False,
) -> None:
    # verify valid range
    if min_value is not None and max_value is not None and min_value > max_value:
        raise ValueError('Invalid range ({}, {}) specified for "{}"'.format(min_value, max_value, name))

    # validate the value is within range
    errors = []

    # check minimum
    if min_value is not None:
        if min_inclusive:
            if value < min_value:
                errors.append("below minimum")
        else:
            if value <= min_value:
                errors.append("below minimum")

    # check maximum
    if max_value is not None:
        if max_inclusive:
            if value > max_value:
                errors.append("above maximum")
        else:
            if value >= max_value:
                errors.append("above maximum")

    # having any error message indicates bad value
    if errors:
        err_msg = 'Variable "{}" value={} not in range ({}, {}): {}'.format(
            name, value, min_value, max_value, " ".join(errors)
        )
        raise ValueError(err_msg)


def to_int(
    name: str,
    value: Any,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
    min_inclusive: bool = True,
    max_inclusive: bool = False,
) -> int:
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


def to_float(
    name: str,
    value: Any,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    min_inclusive: bool = True,
    max_inclusive: bool = False,
) -> float:
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


def to_text(value: Any) -> str:
    """A string value as `str`, whether it arrives as `str` or as `bytes`.

    The single conversion for PyRS's "`bytes` on disk, `str` in memory" rule: a
    caller that hand-rolls `isinstance(v, bytes)` drifts from the next one, and
    nine such sites had drifted into four different behaviours -- some handling
    `numpy.bytes_` and some not, some coercing the non-bytes branch to `str` and
    some returning it untouched.

    Args:
        value: A `str`, `bytes`, `numpy.str_` or `numpy.bytes_`.

    Returns:
        The value as `str`.

    Example:
        >>> to_text(b"11"), to_text("11")
        ('11', '11')
    """
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    return str(value)


def is_text_array(values: Any) -> bool:
    """Whether `values` is a numpy array holding only string values.

    The single definition of "array of strings", so that a caller handling a
    mixture of log types asks one question rather than reimplementing the dtype
    rules. Three dtypes qualify, and the third is the one that matters:

    - `<U`, numpy's own text dtype.
    - `|S`, fixed-width bytes. What `np.array([b"a", b"b"])` produces.
    - `object`, holding only `bytes` or `str`. **What HDF5 actually yields** for
      the variable-length UTF-8 dtype, through both h5py and `nexusformat`.

    An `object` array has to be inspected element-wise, because its dtype says
    nothing. One holding a mix of text and numbers is **not** a text array.

    An **empty** `object` array *is* -- it has no values to misclassify, so the
    only thing at stake is its dtype, and in this codebase it can have come from
    nowhere but an empty variable-length string dataset: every other empty HDF5
    dataset keeps its own dtype on read, and only vlen strings come back as
    `object`. An empty string log is legitimate both on disk and in memory, and
    refusing it would push emptiness back out to every caller as a special case.

    An empty array whose dtype still says *numbers* is a different matter and
    stays False: `float64` is not ambiguous just because it is empty.

    Args:
        values: Anything.

    Returns:
        True when `to_text_array` would convert it.

    Example:
        >>> is_text_array(np.array([b"11"])), is_text_array(np.array([1.5]))
        (True, False)
        >>> is_text_array(np.empty(0, dtype=object))   # an empty string column
        True
    """
    if not isinstance(values, np.ndarray):
        return False
    if values.dtype.kind in ("U", "S"):
        return True
    if values.dtype.kind == "O":
        # `all` over an empty array is True, which is the intended answer here.
        return all(isinstance(v, (bytes, np.bytes_, str, np.str_)) for v in values.ravel())
    return False


def to_text_array(values: np.ndarray) -> np.ndarray:
    """A string array as numpy text (`<U`), whatever spelling it arrives in.

    See `is_text_array` for which dtypes count as strings and why the `object`
    case is the one that matters.

    Args:
        values: An array of string values. Passing anything else is a usage
            error, not a pass-through: a caller that may hold either text or
            numbers asks `is_text_array` first, which is the whole reason that
            predicate is public.

    Returns:
        A `<U` array with the same shape.

    Raises:
        TypeError: If `values` is not an array of strings. Returning it unchanged
            instead would make a misdirected call look like a working one, and
            the symptom would surface later as a `bytes` value in whichever
            consumer did not expect one -- which is the exact failure this
            conversion exists to end. An **empty** `object` array is accepted;
            see `is_text_array`.

    Example:
        >>> to_text_array(np.array([b"11", b"22"])).dtype.kind
        'U'
        >>> obj = np.empty(2, dtype=object); obj[:] = [b"11", b"22"]
        >>> to_text_array(obj).tolist()
        ['11', '22']
        >>> to_text_array(np.empty(0, dtype=object)).dtype.kind
        'U'
    """
    if not is_text_array(values):
        kind = f"{type(values).__name__}" if not isinstance(values, np.ndarray) else f"dtype {values.dtype!r}"
        raise TypeError(
            f"to_text_array expects an array of strings, got {kind}.\n"
            "  Guard the call with `is_text_array(...)` if the value may be numeric, "
            "heterogeneous, or an empty object array."
        )
    if values.dtype.kind == "U":
        return values
    if values.dtype.kind == "S":
        return np.char.decode(values, "utf-8")
    flat = values.ravel()
    # `dtype=np.str_` is load-bearing for the empty case: `np.array([])` is
    # `float64`, so an empty string column would come back as an empty array of
    # numbers -- which is what the hand-rolled conversion this replaced did.
    return np.array([to_text(v) for v in flat], dtype=np.str_).reshape(values.shape)
