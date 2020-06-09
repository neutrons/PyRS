import numpy as np
import pytest
from pyrs.core.strain_stress_calculator import _to_md


def test_bad_extent():
    wksp_name = 'test_bad_extent'
    # check decreasing extent
    try:
        _ = _to_md(wksp_name, (10, 0, 1), (10, 20, 1), (20, 30, 1),
                   np.arange(0, 1000, 1, dtype=float),
                   np.zeros(1000, dtype=float),
                   units='meter,meter,meter')
        assert False, 'Should generate exception'
    except ValueError as err:
        assert 'Extents must be increasing' in str(err)

    # check negative step size
    try:
        _ = _to_md(wksp_name, (0, 10, -1), (10, 20, 1), (20, 30, 1),
                   np.arange(0, 1000, 1, dtype=float),
                   np.zeros(1000, dtype=float),
                   units='meter,meter,meter')
        assert False, 'Should generate exception'
    except ValueError as err:
        assert 'Negative step size' in str(err)


def test_simple():
    wksp_name = 'test_simple'
    # label, min, max, delta
    x_descr = ('x', 0, 10, 1)
    y_descr = ('y', 10, 20, 1)
    z_descr = ('z', 20, 30, 1)

    signal = np.arange(0, 1000, 1, dtype=float)

    # create MDhisto
    histo = _to_md(wksp_name, x_descr[1:], y_descr[1:], z_descr[1:],
                   signal, np.zeros(1000, dtype=float), units='meter,meter,meter')

    # simple checks
    assert histo, 'Workspace was not returned'
    assert histo.id() == 'MDHistoWorkspace'
    # dimensions
    for i, (label, min_value, max_value, delta) in enumerate([x_descr, y_descr, z_descr]):
        dimension = histo.getDimension(i)
        assert dimension.getName() == label
        assert dimension.getUnits() == 'meter'
        assert dimension.getMinimum() == min_value
        assert dimension.getMaximum() == max_value
        assert dimension.getX(1) - dimension.getX(0) == delta
    # check the signal and error
    np.testing.assert_equal(histo.getSignalArray().ravel(), signal, err_msg='Signal')
    np.testing.assert_equal(histo.getErrorSquaredArray().ravel(), 0., err_msg='Errors')

    # clean up
    histo.delete()


if __name__ == '__main__':
    pytest.main[__file__]
