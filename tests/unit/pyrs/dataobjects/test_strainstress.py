import pytest
import numpy as np

from pyrs.core.stress_facade import StressFacade
from pyrs.dataobjects.fields import StrainField, StressField
from pyrs.dataobjects.sample_logs import PointList, SubRuns
from pyrs.peaks.peak_collection import PeakCollection

NAN = np.nan

@pytest.fixture(scope='module')
def stress_mock2():
    r"""Mock of a facade object"""

    def serve_list(begin, end):
        vx = 0.1 * np.arange(10)  # from 0.0 to 0.9
        return PointList([vx[begin: end], np.zeros(end - begin), np.zeros(end - begin)])


    class PeakCollectionMock(PeakCollection):
        r"""Duck typing for PeakCollection"""

        def __init__(self,  runnumber=None):
            self._runnumber = runnumber
            self._sub_run_array = SubRuns()

    r"""We create four single strain instances, below is how they overlap over the vx's extent
        0  1  2  3  4  5  6  7  8  9
    s0  ***********************
    s1    ************
    s2             ***************
    s3       ***********************
    """
    point_lists = [serve_list(begin, end) for begin, end in [(0, 8), (1, 5), (4, 9), (2, 10)]]
    peak_collections = [PeakCollectionMock(runnumber=runnumber) for runnumber in [1234, 1235, 1236, 1237]]

    strains = list()
    for point_list, peak_collection in zip(point_lists, peak_collections):
        strains.append(StrainField(point_list=point_list, peak_collection=peak_collection))
    strain11, strain22, strain33 = strains[0], strains[1] + strains[2], strains[3]

    # values of Young's modulus and Poisson's ratio to render simpler strain-to-stress formulae
    return {'diagonal': StressField(strain11, strain22, strain33, 1./3, 4./3, 'diagonal'),
            'in-plane-strain': StressField(strain11, strain22, None, 1./3, 4./3, 'in-plane-strain'),
            'in-plane-stress': StressField(strain11, strain22, None, 1./2, 3./2, 'in-plane-stress')}


def assert_workspace():
    return True


class TestFacade:

    def test_init(self, stress_mock):
        assert StressFacade(stress_mock['diagonal'])
        assert StressFacade(stress_mock['in-plane'])

    def test_runs(self, stress_mock):
        r"""Find runs for a particular direction"""
        assert stress_mock.runs('11') == ['1234', '1235']
        assert stress_mock.runs('22') == ['1236']
        assert stress_mock.runs('33') == ['1237', '1238']

    @pytest.mark.skip(reason='Not yet implemented')
    def test_youngs_modulus(self, stress_mock):
        assert stress_mock.youngs_modulus == 1.0

    @pytest.mark.skip(reason='Not yet implemented')
    def test_poisson_ratio(self, stress_mock):
        assert stress_mock.poisson_ratio == 1.0

    @pytest.mark.skip(reason='Not yet implemented')
    def test_d0(self, stress_mock):
        r"""Get the reference lattice spacing"""
        assert stress_mock.d0.values
        assert stress_mock.d0.errors

    @pytest.mark.skip(reason='Not yet implemented')
    def test_d0(self, stress_mock):
        r"""Export the reference spacing to a MDHistoWorkspace"""
        assert stress_mock.workspace('d0')

    @pytest.mark.skip(reason='Not yet implemented')
    def test_select(self, stress_mock):
        r"""Select run numbers or directions"""

        stress_mock.select('11')
        assert stress_mock.direction == '11'
        stress_mock.runs == ['1234', '1235']
        assert stress_mock.selection == '11'
        stress_mock.select('1234')
        assert stress_mock.direction == '11'
        stress_mock.runs == '1234'
        assert stress_mock.selection == '11'
        with pytest.raises(TypeError) as exception_info:
            stress_mock.select(1234)
        assert 'Expected format for run numbers is a strin' in str(exception_info.value)

        stress_mock.select(1236)
        assert stress_mock.selection == 1236
        assert stress_mock.direction == '22'
        assert stress_mock.runs == [1236]

    @pytest.mark.skip(reason='Not yet implemented')
    def test_xyz(self, stress_mock):
        r"""Coordinates of the sample points"""
        assert stress_mock.x == np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0])
        assert stress_mock.y == np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        assert stress_mock.z == np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

    @pytest.mark.skip(reason='Not yet implemented')
    def test_strain_array(self, stress_mock):
        r"""strains along a particular direction or run number"""
        stress_mock.select(1234)
        assert stress_mock.strain.values == np.array([100, 110, 120, 130, 140, 150, NAN, NAN, NAN, NAN])
        assert stress_mock.strain.errors == np.array([10, 11, 12, 13, 14, 15, NAN, NAN, NAN, NAN])
        stress_mock.select(1235)
        assert stress_mock.strain.values == np.array([NAN, NAN, NAN, NAN, NAN, 150, 160, 170, 180, 190])
        assert stress_mock.strain.errors == np.array([NAN, NAN, NAN, NAN, NAN, 15, 16, 17, 18, 19])
        stress_mock.select('11')
        assert stress_mock.strain.values == np.array([100, 110, 120, 130, 140, 150, 160, 170, 180, 190])
        assert stress_mock.strain.values == np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])

    @pytest.mark.skip(reason='Not yet implemented')
    def test_strain_workspace(self, stress_mock):
        r"""Export the strains to a MDHistoWorkspace"""
        stress_mock.select(1234)
        assert stress_mock.workspace('strain')
        stress_mock.select('11')
        assert stress_mock.workspace('strain')

    @pytest.mark.skip(reason='Not yet implemented')
    def test_stress(self, stress_mock):
        r"""strains along a particular direction. Also for a run number, when a direction contains only one run
        number"""
        stress_mock.select(1234)
        with pytest.raises(ValueError) as exception_info:
            stress_mock.stress
        assert 'The selection (run 1234) does not define a direction' in str(exception_info.value)
        stress_mock.select('11')
        assert stress_mock.stress.values == [200, 210, 220, 230, 240, 250, 260, 270, 280, 290]
        assert stress_mock.stress.errors == [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        stress_mock.select('22')
        assert stress_mock.stress.values == [300, 310, 320, 330, 340, 350, 360, 370, 380, 390]
        assert stress_mock.stress.errors == [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]

    @pytest.mark.skip(reason='Not yet implemented')
    def test_stress_workspace(self, stress_mock):
        r"""Export the stress components to a MDHistoWorkspace"""
        stress_mock.select(1234)
        with pytest.raises(ValueError) as exception_info:
            stress_mock.to_worksapce('stress')
        assert 'The selection (run 1234) does not define a direction' in str(exception_info.value)
        stress_mock.select('11')
        assert stress_mock.workspace('stress')
        stress_mock.select('1236')
        assert stress_mock.workspace('stress')

    @pytest.mark.skip(reason='Not yet implemented')
    def test_fitting_parameters(self, stress_mock):
        r"""Retrieve the titting parameters for a particular run, or for a particular direction"""
        parameters = ['Center', 'Height', 'FWHM', 'Mixing', 'A0', 'A1', 'Intensity']
        stress_mock.select(1234)
        for parameter in parameters:
            assert stress_mock.get_parameter(parameter).values
            assert stress_mock.get_parameter(parameter).errors
        stress_mock.select('11')
        for parameter in parameters:
            assert stress_mock.get_parameter(parameter).values
            assert stress_mock.get_parameter(parameter).errors

    @pytest.mark.skip(reason='Not yet implemented')
    def test_fitting_parameters_workspace(self, stress_mock):
        r"""Export the fitting parameters to MDHistoWorkspace"""
        parameters = ['Center', 'Height', 'FWHM', 'Mixing', 'A0', 'A1', 'Intensity']
        stress_mock.select(1234)
        for parameter in parameters:
            assert stress_mock.workspace(parameter)
        stress_mock.select('11')
        for parameter in parameters:
            assert stress_mock.workspace(parameter)
