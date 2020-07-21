# This is a test of the model component of the strain/stress viewer
from pyrs.interface.strainstressviewer.model import Model
import numpy as np
import pytest


def test_model():
    model = Model()

    assert model.selectedPeak is None
    assert model.e11 is None
    assert model.e22 is None
    assert model.e33 is None
    assert model._stress is None

    model.e11 = 'tests/data/HB2B_1320.h5'
    assert model.e11.name == '11'
    assert model.peakTags == ['peak0']
    assert 'peak0' in model.e11_peaks

    model.selectedPeak = 'peak0'
    assert model.selectedPeak == 'peak0'

    assert model.subruns == range(1, 313)

    assert model.validate_selection('11') is None
    assert model.validate_selection('22') == "e22 file hasn't been loaded"
    assert model.validate_selection('33') == "e33 file hasn't been loaded"

    d0, d0_e = model.d0
    np.testing.assert_equal(d0, np.ones(312))
    np.testing.assert_equal(d0_e, np.zeros(312))

    model.d0 = np.linspace(1, 1.05, 312)
    d0, d0_e = model.d0
    np.testing.assert_equal(d0, np.linspace(1, 1.05, 312))
    np.testing.assert_equal(d0_e, np.zeros(312))

    for plot_param in ("dspacing_center",
                       "d_reference",
                       "Center",
                       "Height",
                       "FWHM",
                       "Mixing",
                       "Intensity",
                       "strain"):
        e11_md = model.get_field_md('11', plot_param)
        assert e11_md.name() == f'e11 {plot_param}'
        assert e11_md.getNumDims() == 3
        assert [e11_md.getDimension(n).getNBins() for n in range(3)] == [18, 6, 3]
        assert model.get_field_md('22', plot_param) is None
        assert model.get_field_md('33', plot_param) is None

    # Need to load ε22 so it should fail
    with pytest.raises(KeyError):
        model.calculate_stress('in-plane-stress', 200, 0.3)

    model.e22 = 'tests/data/HB2B_1320.h5'
    assert model.e22.name == '22'

    model.calculate_stress('in-plane-stress', 200, 0.3)

    for direction in ('11', '22', '33'):
        stress_md = model.get_field_md(direction, 'stress')
        assert stress_md.name() == f'e{direction} stress'
        assert stress_md.getNumDims() == 3
        assert [stress_md.getDimension(n).getNBins() for n in range(3)] == [18, 6, 3]

    # Should be all zero for in-plane stress case
    assert np.count_nonzero(model.get_field_md('33', 'stress').getSignalArray()) == 0

    model.calculate_stress('in-plane-strain', 200, 0.3)

    for direction in ('11', '22', '33'):
        stress_md = model.get_field_md(direction, 'stress')
        assert stress_md.name() == f'e{direction} stress'
        assert stress_md.getNumDims() == 3
        assert [stress_md.getDimension(n).getNBins() for n in range(3)] == [18, 6, 3]

    # Should be all non-zero for in-plane strain case
    assert np.count_nonzero(model.get_field_md('33', 'stress').getSignalArray()) == 18*6*3

    model.e33 = 'tests/data/HB2B_1320.h5'
    assert model.e33.name == '33'

    model.calculate_stress('diagonal', 200, 0.3)

    for direction in ('11', '22', '33'):
        stress_md = model.get_field_md(direction, 'stress')
        assert stress_md.name() == f'e{direction} stress'
        assert stress_md.getNumDims() == 3
        assert [stress_md.getDimension(n).getNBins() for n in range(3)] == [18, 6, 3]

    # Should be all non-zero for diagonal stress case
    assert np.count_nonzero(model.get_field_md('33', 'stress').getSignalArray()) == 18*6*3
