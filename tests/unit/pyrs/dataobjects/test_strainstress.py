import pytest
import numpy as np
from numpy.testing import assert_allclose

from pyrs.core.stress_facade import StressFacade


NAN = np.nan


def assert_workspace(workspace, signal_array):
    r"""
    Set of assertions for data related to fixture strain_stress_object_1
    """
    assert workspace.id() == 'MDHistoWorkspace'
    dimension = workspace.getDimension(0)
    assert dimension.getUnits() == 'mm'
    # adding half a bin each direction since values from mdhisto are boundaries and constructor uses centers
    min_value, max_value = 0.0, 9.0
    assert dimension.getMinimum() == pytest.approx(min_value - 0.5)
    assert dimension.getMaximum() == pytest.approx(max_value + 0.5)
    assert dimension.getNBins() == 10
    assert_allclose(workspace.getSignalArray().ravel(), signal_array, atol=1.e-6)


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

    def test_x_y_z(self, strain_stress_object_1):
        facade = StressFacade(strain_stress_object_1['stresses']['diagonal'])
        assert_allclose(facade.x, np.arange(10))
        assert_allclose(facade.y, np.zeros(10))
        assert_allclose(facade.z, np.zeros(10))

        facade = StressFacade(strain_stress_object_1['stresses']['in-plane-strain'])
        assert_allclose(facade.x, np.arange(9))
        assert_allclose(facade.y, np.zeros(9))
        assert_allclose(facade.z, np.zeros(9))

    def test_direction(self, strain_stress_object_1):
        r"""Select run numbers or directions"""
        facade = StressFacade(strain_stress_object_1['stresses']['in-plane-stress'])
        facade.selection = '11'
        assert facade.direction == '11'
        facade.selection = '1234'
        assert facade.direction == '11'
        facade.selection = '1235'
        assert facade.direction == '22'
        facade.selection = '33'
        assert facade.direction == '33'

    def test_strain_field(self, strain_stress_object_1):
        r"""strains along for a particular direction or run number"""
        nanf = float('nan')
        facade = StressFacade(strain_stress_object_1['stresses']['diagonal'])
        # direction 11 and components
        facade.selection = '11'
        expected = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, nanf, nanf]
        r"""
        assert_allclose(facade.strain.values, expected, equal_nan=True, atol=1e-6)
        facade.selection = '1234'
        assert_allclose(facade.strain.values, expected, equal_nan=True, atol=1e-6)
        # direction 22 and components
        facade.selection = '22'
        expected = [nanf, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, nanf]
        assert_allclose(facade.strain.values, expected, equal_nan=True, atol=1e-6)
        facade.selection = '1235'
        expected = [nanf, 0.01, 0.02, 0.03, 0.04, nanf, nanf, nanf, nanf, nanf]
        facade.selection = '1236'
        expected = [nanf, nanf, nanf, nanf, 0.04, 0.05, 0.06, 0.07, 0.08, nanf]
        assert_allclose(facade.strain.values, expected, equal_nan=True, atol=1e-6)
        # direction 33 and components
        facade.selection = '33'
        expected = [nanf, nanf, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
        assert_allclose(facade.strain.values, expected, equal_nan=True, atol=1e-6)
        facade.selection = '1237'
        assert_allclose(facade.strain.values, expected, equal_nan=True, atol=1e-6)
        """

        facade = StressFacade(strain_stress_object_1['stresses']['in-plane-strain'])
        r"""
        # direction 11 and components
        facade.selection = '11'
        expected = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, nanf]
        assert_allclose(facade.strain.values, expected, equal_nan=True, atol=1e-6)
        facade.selection = '1234'
        assert_allclose(facade.strain.values, expected, equal_nan=True, atol=1e-6)
        # direction 22 and components
        facade.selection = '22'
        expected = [nanf, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
        assert_allclose(facade.strain.values, expected, equanlnan=True, atol=1e-6)
        facade.selection = '1235'
        expected = [nanf, 0.01, 0.02, 0.03, 0.04, nanf, nanf, nanf, nanf]
        facade.selection = '1236'
        expected = [nanf, nanf, nanf, nanf, 0.04, 0.05, 0.06, 0.07, 0.08]
        assert_allclose(facade.strain.values, expected, equal_nan=True, atol=1e-6)
        """
        # direction 33 and components
        facade.selection = '33'
        expected = [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]

        facade = StressFacade(strain_stress_object_1['stresses']['in-plane-stress'])
        r"""
        # direction 11 and components
        facade.selection = '11'
        expected = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, nanf]
        assert_allclose(facade.strain.values, expected, equal_nan=True, atol=1e-6)
        facade.selection = '1234'
        assert_allclose(facade.strain.values, expected, equal_nan=True, atol=1e-6)
        # direction 22 and components
        facade.selection = '22'
        expected = [nanf, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
        assert_allclose(facade.strain.values, expected, equal_nan=True, atol=1e-6)
        facade.selection = '1235'
        expected = [nanf, 0.01, 0.02, 0.03, 0.04, nanf, nanf, nanf, nanf]
        facade.selection = '1236'
        expected = [nanf, nanf, nanf, nanf, 0.04, 0.05, 0.06, 0.07, 0.08]
        assert_allclose(facade.strain.values, expected, equal_nan=True, atol=1e-6)
        # direction 33 and components
        facade.selection = '33'
        expected = [nanf, -0.02, -0.04, -0.06, -0.08, -0.10, -0.12, -0.14, nanf]
        """

    def test_strain_workspace(self, strain_stress_object_1):
        r"""Export the strains to a MDHistoWorkspace"""
        facade = StressFacade(strain_stress_object_1['stresses']['diagonal'])
        facade.selection = '11'
        assert_workspace(facade.workspace('strain'), facade.strain.values)

        r"""
        facade = StressFacade(strain_stress_object_1['stresses']['in-plane-strain'])
        facade.selection = '33'
        assert_workspace(facade.workspace('strain'), facade.strain.values)
        """

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
    def test_d_reference_field(self, strain_stress_object_1):
        r"""Get the reference lattice spacing"""
        facade = StressFacade(strain_stress_object_1['stresses']['diagonal'])
        assert_allclose(facade.d_reference, np.ones(10))

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
