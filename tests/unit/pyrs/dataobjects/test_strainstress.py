import pytest
import numpy as np

from copy import deepcopy
from pyrs.core.stress_facade import StressFacade
from pyrs.dataobjects.fields import StrainField, StressField
from pyrs.dataobjects.sample_logs import PointList, SubRuns
from pyrs.peaks.peak_collection import PeakCollection

NAN = np.nan

def assert_workspace():
    return True


class TestStressFacade:

    def test_init(self, strain_stress_object_1):
        for stress in strain_stress_object_1['stresses'].values():
            facade = StressFacade(stress)
            assert facade
            assert facade._stress_cache['11'] == facade._stress.stress11
            assert facade._strain_cache['11'] == facade._stress.strain11
            assert facade._strain_cache['1234'] == facade._stress.strain11.strains[0]
            assert facade._strain_cache['1236'] == facade._stress.strain22.strains[1]
            assert facade._stress_cache['33'] == facade._stress.stress33
            assert facade._strain_cache['33'] == facade._stress.strain33

    def test_runs(self, strain_stress_object_1):
        r"""Find runs for a particular direction"""
        facade = StressFacade(strain_stress_object_1['stresses']['diagonal'])
        assert facade.runs('11') == ['1234']
        assert facade.runs('22') == ['1235', '1236']
        assert facade.runs('33') == ['1237']
        assert facade._all_runs() == ['1234', '1235', '1236', '1237']

        facade = StressFacade(strain_stress_object_1['stresses']['in-plane-strain'])
        assert facade.runs('33') == []
        facade = StressFacade(strain_stress_object_1['stresses']['in-plane-stress'])
        assert facade.runs('33') == []
        assert facade._all_runs() == ['1234', '1235', '1236']

    def test_youngs_modulus(self, strain_stress_object_1):
        facade = StressFacade(strain_stress_object_1['stresses']['diagonal'])
        assert facade.youngs_modulus == pytest.approx(4. / 3)

    def test_poisson_ratio(self, strain_stress_object_1):
        facade = StressFacade(strain_stress_object_1['stresses']['diagonal'])
        assert facade.poisson_ratio == pytest.approx(1. / 3)

    @pytest.mark.skip(reason='Not yet implemented')
    def test_d_refence(self, strain_stress_object_1):
        r"""Get the reference lattice spacing"""
        facade = StressFacade(strain_stress_object_1['stresses']['diagonal'])
        facade.selection
        assert facade.d_reference.values
        assert facade.d_reference.errors
        assert facade.workspace('d_reference')

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
