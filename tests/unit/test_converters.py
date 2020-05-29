import pytest

from pyrs.utilities import to_float


def test_convert_to_float():
    GOOD = 'good'
    BAD = 'bad'

    # simple checks
    assert to_float(GOOD, 42.) == 42.
    assert to_float(GOOD, 42) == 42.
    assert to_float(GOOD, '42') == 42.
    assert to_float(GOOD, 42., 41, 43) == 42.
    assert to_float(BAD, 42., 42, 43, min_inclusive=True, max_inclusive=False) == 42.
    assert to_float(BAD, 42., 41, 42, min_inclusive=False, max_inclusive=True) == 42.

    # check invalid range
    with pytest.raises(ValueError) as err:
        to_float(BAD, 42., 43, 41)
        assert BAD in str(err.value)

    # check value outside of range
    with pytest.raises(ValueError) as err:
        assert not to_float(BAD, 42., 42, 43, min_inclusive=False, max_inclusive=True)
        assert BAD in str(err.value)
    with pytest.raises(ValueError) as err:
        assert not to_float(BAD, 42., 41, 42, min_inclusive=True, max_inclusive=False)
        assert BAD in str(err.value)

    # check with non-convertable values
    for value in [None, 'strings cannot be converted', (1, 2)]:
        with pytest.raises(TypeError) as err:
            assert not to_float(BAD, value)
            assert BAD in str(err.value)


if __name__ == '__main__':
    pytest.main([__file__])
