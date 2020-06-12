import numpy as np
import pytest
from pyrs.dataobjects.sample_logs import DirectionExtents
from pyrs.core.strain_stress_calculator import _to_md


def test_simple():
    wksp_name = 'test_simple'

    # sampled coordinates 10 x 10 x 10 = 10^3
    # each (x, y, z) coordinate is the centerpoint in a 3D bin
    xyz = [list(range(0, 10)), list(range(10, 20)), list(range(20, 30))]
    extents = tuple([DirectionExtents(coordinates) for coordinates in xyz])

    # we have one signal and one error for each of the 10 x 10 x 10 xyz coordinates
    signal, errors = np.arange(0, 1000, 1, dtype=float), np.zeros(1000, dtype=float)

    histo = _to_md(wksp_name, extents, signal, errors, units='meter,meter,meter')
    assert histo, 'Workspace was not returned'
    assert histo.id() == 'MDHistoWorkspace'

    extents_binmd = [[float(x) for x in extent.to_binmd.split(',')] for extent in extents]
    for i, (min_value, max_value, number_bins) in enumerate(extents_binmd):
        dimension = histo.getDimension(i)
        assert dimension.getUnits() == 'meter'
        assert dimension.getMinimum() == min_value
        assert dimension.getMaximum() == max_value
        assert dimension.getNBins() == number_bins
    # check the signal and error
    np.testing.assert_equal(histo.getSignalArray().ravel(), signal, err_msg='Signal')
    np.testing.assert_equal(histo.getErrorSquaredArray().ravel(), 0., err_msg='Errors')

    # clean up
    histo.delete()


if __name__ == '__main__':
    pytest.main[__file__]
