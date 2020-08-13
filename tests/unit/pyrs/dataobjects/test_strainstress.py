import pytest
import numpy as np

from pyrs.peaks.peak_collection import PeakCollection
from pyrs.dataobjects.fields import StressField
NAN = np.nan

@pytest.fixture(scope='module')
def facade_mock():
    r"""Mock of a facade object"""

    class PeakCollectionMock(PeakCollection):
        r"""Duck typing for PeakCollection"""
        def __init__(self, runnumber=None)
            self._runnumber = runnumber

    peaks = {n: PeakCollectionMock(runnumber=n) for n in (1234, 1235, 1236, 1237, 1238) }


def assert_workspace():
    return True


class TestFacade:

    @pytest.mark.skip(reason='Not yet implemented')
    def test_runs(self, facade_mock):
        r"""Find runs for a particular direction"""
        assert facade_mock.runs('11') == ['1234', '1235']
        assert facade_mock.runs('22') == ['1236']
        assert facade_mock.runs('33') == ['1237', '1238']

    @pytest.mark.skip(reason='Not yet implemented')
    def test_youngs_modulus(self, facade_mock):
        assert facade_mock.youngs_modulus == 1.0

    @pytest.mark.skip(reason='Not yet implemented')
    def test_poisson_ratio(self, facade_mock):
        assert facade_mock.poisson_ratio == 1.0

    @pytest.mark.skip(reason='Not yet implemented')
    def test_d0(self, facade_mock):
        r"""Get the reference lattice spacing"""
        assert facade_mock.d0.values
        assert facade_mock.d0.errors

    @pytest.mark.skip(reason='Not yet implemented')
    def test_d0(self, facade_mock):
        r"""Export the reference spacing to a MDHistoWorkspace"""
        assert facade_mock.workspace('d0')

    @pytest.mark.skip(reason='Not yet implemented')
    def test_select(self, facade_mock):
        r"""Select run numbers or directions"""

        facade_mock.select('11')
        assert facade_mock.direction == '11'
        facade_mock.runs == ['1234', '1235']
        assert facade_mock.selection == '11'
        facade_mock.select('1234')
        assert facade_mock.direction == '11'
        facade_mock.runs == '1234'
        assert facade_mock.selection == '11'
        with pytest.raises(TypeError) as exception_info:
            facade_mock.select(1234)
        assert 'Expected format for run numbers is a strin' in str(exception_info.value)

        facade_mock.select(1236)
        assert facade_mock.selection == 1236
        assert facade_mock.direction == '22'
        assert facade_mock.runs == [1236]

    @pytest.mark.skip(reason='Not yet implemented')
    def test_xyz(self, facade_mock):
        r"""Coordinates of the sample points"""
        assert facade_mock.x == np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0])
        assert facade_mock.y == np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        assert facade_mock.z == np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

    @pytest.mark.skip(reason='Not yet implemented')
    def test_strain_array(self, facade_mock):
        r"""strains along a particular direction or run number"""
        facade_mock.select(1234)
        assert facade_mock.strain.values == np.array([100, 110, 120, 130, 140, 150, NAN, NAN, NAN, NAN])
        assert facade_mock.strain.errors == np.array([10, 11, 12, 13, 14, 15, NAN, NAN, NAN, NAN])
        facade_mock.select(1235)
        assert facade_mock.strain.values == np.array([NAN, NAN, NAN, NAN, NAN, 150, 160, 170, 180, 190])
        assert facade_mock.strain.errors == np.array([NAN, NAN, NAN, NAN, NAN, 15, 16, 17, 18, 19])
        facade_mock.select('11')
        assert facade_mock.strain.values == np.array([100, 110, 120, 130, 140, 150, 160, 170, 180, 190])
        assert facade_mock.strain.values == np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])

    @pytest.mark.skip(reason='Not yet implemented')
    def test_strain_workspace(self, facade_mock):
        r"""Export the strains to a MDHistoWorkspace"""
        facade_mock.select(1234)
        assert facade_mock.workspace('strain')
        facade_mock.select('11')
        assert facade_mock.workspace('strain')

    @pytest.mark.skip(reason='Not yet implemented')
    def test_stress(self, facade_mock):
        r"""strains along a particular direction. Also for a run number, when a direction contains only one run
        number"""
        facade_mock.select(1234)
        with pytest.raises(ValueError) as exception_info:
            facade_mock.stress
        assert 'The selection (run 1234) does not define a direction' in str(exception_info.value)
        facade_mock.select('11')
        assert facade_mock.stress.values == [200, 210, 220, 230, 240, 250, 260, 270, 280, 290]
        assert facade_mock.stress.errors == [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        facade_mock.select('22')
        assert facade_mock.stress.values == [300, 310, 320, 330, 340, 350, 360, 370, 380, 390]
        assert facade_mock.stress.errors == [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]

    @pytest.mark.skip(reason='Not yet implemented')
    def test_stress_workspace(self, facade_mock):
        r"""Export the stress components to a MDHistoWorkspace"""
        facade_mock.select(1234)
        with pytest.raises(ValueError) as exception_info:
            facade_mock.to_worksapce('stress')
        assert 'The selection (run 1234) does not define a direction' in str(exception_info.value)
        facade_mock.select('11')
        assert facade_mock.workspace('stress')
        facade_mock.select('1236')
        assert facade_mock.workspace('stress')

    @pytest.mark.skip(reason='Not yet implemented')
    def test_fitting_parameters(self, facade_mock):
        r"""Retrieve the titting parameters for a particular run, or for a particular direction"""
        parameters = ['Center', 'Height', 'FWHM', 'Mixing', 'A0', 'A1', 'Intensity']
        facade_mock.select(1234)
        for parameter in parameters:
            assert facade_mock.get_parameter(parameter).values
            assert facade_mock.get_parameter(parameter).errors
        facade_mock.select('11')
        for parameter in parameters:
            assert facade_mock.get_parameter(parameter).values
            assert facade_mock.get_parameter(parameter).errors

    @pytest.mark.skip(reason='Not yet implemented')
    def test_fitting_parameters_workspace(self, facade_mock):
        r"""Export the fitting parameters to MDHistoWorkspace"""
        parameters = ['Center', 'Height', 'FWHM', 'Mixing', 'A0', 'A1', 'Intensity']
        facade_mock.select(1234)
        for parameter in parameters:
            assert facade_mock.workspace(parameter)
        facade_mock.select('11')
        for parameter in parameters:
            assert facade_mock.workspace(parameter)
