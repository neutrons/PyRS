import pytest
import numpy as np
from numpy.testing import assert_allclose

from pyrs.core.stress_facade import StressFacade
from pyrs.dataobjects.fields import ScalarFieldSample

nanf = float('nan')
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

    def test_youngs_modulus_setter(self, strain_stress_object_0):
        facade = StressFacade(strain_stress_object_0['stresses']['diagonal'])
        facade.youngs_modulus *= 2
        assert facade.youngs_modulus == pytest.approx(8. / 3)
        facade.selection = '11'
        assert_allclose(facade.stress.values, 2 * np.array([0.30, 0.34, 0.38, 0.42, 0.46]), atol=0.01)
        facade.selection = '22'
        assert_allclose(facade.stress.values, 2 * np.array([0.40, 0.44, 0.48, 0.52, 0.56]), atol=0.01)
        facade.selection = '33'
        assert_allclose(facade.stress.values, 2 * np.array([0.50, 0.54, 0.58, 0.62, 0.66]), atol=0.01)

    def test_poisson_ratio(self, strain_stress_object_1):
        facade = StressFacade(strain_stress_object_1['stresses']['diagonal'])
        assert facade.poisson_ratio == pytest.approx(1. / 3)

    def test_poisson_ratio_setter(self, strain_stress_object_0):
        facade = StressFacade(strain_stress_object_0['stresses']['diagonal'])
        facade.poisson_ratio = 0.0
        assert facade.poisson_ratio == pytest.approx(0.0)
        for selection in ('11', '22', '33'):
            facade.selection = selection
            assert_allclose(facade.stress.values, facade.youngs_modulus * facade.strain.values, atol=0.001)

    def test_x_y_z(self, strain_stress_object_1):
        facade = StressFacade(strain_stress_object_1['stresses']['diagonal'])
        assert_allclose(facade.x, np.arange(10))
        assert_allclose(facade.y, np.zeros(10))
        assert_allclose(facade.z, np.zeros(10))

        facade = StressFacade(strain_stress_object_1['stresses']['in-plane-strain'])
        assert_allclose(facade.x, np.arange(9))
        assert_allclose(facade.y, np.zeros(9))
        assert_allclose(facade.z, np.zeros(9))

    def test_point_list(self, strain_stress_object_1):
        stress = strain_stress_object_1['stresses']['diagonal']
        facade = StressFacade(stress)
        assert facade.point_list == stress.point_list

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

    def test_extend_to_stacked_point_list(self, strain_stress_object_1):
        stress = strain_stress_object_1['stresses']['diagonal']
        facade = StressFacade(stress)
        facade.selection == '1235'
        field = stress.strain22.strains[0].field  # scalar field sample for run 1235
        assert_allclose(field.values, [0.01, 0.02, 0.03, 0.04], atol=0.001)
        field_extended = facade._extend_to_stacked_point_list(field)
        nan = float('nan')
        assert_allclose(field_extended.values, [nan, 0.01, 0.02, 0.03, 0.04, nan, nan, nan, nan, nan, ], atol=0.001)

    def test_strain_field(self, strain_stress_object_1):
        r"""strains along for a particular direction or run number"""
        facade = StressFacade(strain_stress_object_1['stresses']['diagonal'])
        # direction 11 and components
        facade.selection = '11'
        expected = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, nanf, nanf]
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
        expected = [nanf, nanf, nanf, nanf, 0.045, 0.05, 0.06, 0.07, 0.08, nanf]
        assert_allclose(facade.strain.values, expected, equal_nan=True, atol=1e-6)
        # direction 33 and components
        facade.selection = '33'
        expected = [nanf, nanf, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
        assert_allclose(facade.strain.values, expected, equal_nan=True, atol=1e-6)
        facade.selection = '1237'
        assert_allclose(facade.strain.values, expected, equal_nan=True, atol=1e-6)

        facade = StressFacade(strain_stress_object_1['stresses']['in-plane-strain'])
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
        expected = [nanf, nanf, nanf, nanf, 0.045, 0.05, 0.06, 0.07, 0.08]
        assert_allclose(facade.strain.values, expected, equal_nan=True, atol=1e-6)
        # direction 33 and components
        facade.selection = '33'
        expected = [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
        assert_allclose(facade.strain.values, expected, equal_nan=True, atol=1e-6)

        facade = StressFacade(strain_stress_object_1['stresses']['in-plane-stress'])
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
        expected = [nanf, nanf, nanf, nanf, 0.045, 0.05, 0.06, 0.07, 0.08]
        assert_allclose(facade.strain.values, expected, equal_nan=True, atol=1e-6)
        # direction 33 and components
        facade.selection = '33'
        expected = [nanf, -0.02, -0.04, -0.06, -0.08, -0.10, -0.12, -0.14, nanf]

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

    def test_stress_field(self, strain_stress_object_1):
        r"""Stresses along a particular direction. Also for a run number, when a direction contains only one run
        number"""
        facade = StressFacade(strain_stress_object_1['stresses']['diagonal'])
        facade.selection = '1234'
        with pytest.raises(ValueError) as exception_info:
            facade.stress.values
        assert 'Stress can only be computed for directions' in str(exception_info.value)
        # TODO assert stresses for the remaining selections

    def test_stress_workspace(self, strain_stress_object_1):
        r"""Stresses along a particular direction. Also for a run number, when a direction contains only one run
        number"""
        facade = StressFacade(strain_stress_object_1['stresses']['diagonal'])
        facade.selection = '1234'
        with pytest.raises(ValueError) as exception_info:
            facade.workspace('stress')
        assert 'Stress can only be computed for directions' in str(exception_info.value)
        # TODO assert stresses for the remaining selections

    def test_d_reference_field(self, strain_stress_object_1):
        r"""Get the reference lattice spacing"""
        stress = strain_stress_object_1['stresses']['diagonal']
        facade = StressFacade(stress)
        assert_allclose(facade.d_reference.values, np.ones(facade.size))
        # "pollute" the reference spacing of run 1235
        strain_single_1235 = stress.strain22.strains[0]
        strain_single_1235.set_d_reference([1.001, 0.1])
        facade = StressFacade(stress)
        with pytest.raises(AssertionError) as exception_info:
            facade.d_reference
        assert 'reference spacings are different on different directions' in str(exception_info.value)

    def test_set_d_reference(self, strain_stress_object_0, strain_stress_object_1):
        r"""
        strain_stress_object_0: strains stacked, all have the same set of sample points
        strain_stress_object_1: strains stacked, having different set of sample points
        """
        #
        # Using strain_stress_object_0
        #
        facade = StressFacade(strain_stress_object_0['stresses']['diagonal'])
        #
        # Case single value (errors assumed 0.0)
        facade.d_reference = 2.0
        assert_allclose(facade.d_reference.values, 2.0 * np.ones(facade.size))
        assert_allclose(facade.d_reference.errors, np.zeros(facade.size))
        #
        # Case single value and error, passing different types of objects
        for d_update in [(2.0, 0.20), [2.1, 0.21], np.array([2.2, 0.22])]:
            value, error = d_update
            facade.d_reference = d_update
            assert_allclose(facade.d_reference.values, value * np.ones(facade.size))
            assert_allclose(facade.d_reference.errors, error * np.ones(facade.size))
        #
        # Case scalar field with d_reference update for all sample points
        values = 1.0 + 0.1 * np.arange(facade.size)  # (1.0, 1.1, 1.2, 1.3, 1.4)
        d_update = ScalarFieldSample('d_reference', values, 0.1 * values, facade.x, facade.y, facade.z)
        facade.d_reference = d_update
        assert_allclose(facade.d_reference.values, values)
        assert_allclose(facade.d_reference.errors, 0.1 * values)
        #
        # Case scalar field with d_reference update for some sample points
        facade.d_reference = (1.0, 0.0)  # "reset" d_reference
        assert_allclose(facade.d_reference.values, [1.0, 1.0, 1.0, 1.0, 1.0])
        assert_allclose(facade.d_reference.errors, [0.0, 0.0, 0.0, 0.0, 0.0])
        indexes = [0, 2, 4]  # indexes of the sample points whose d_reference will be updated
        d_update = ScalarFieldSample('d_reference', values[indexes], 0.1 * values[indexes],
                                     facade.x[indexes], facade.y[indexes], facade.z[indexes])
        facade.d_reference = d_update
        assert_allclose(facade.d_reference.values, [1.0, 1.0, 1.2, 1.0, 1.4])
        assert_allclose(facade.d_reference.errors, [0.10, 0.00, 0.12, 0.00, 0.14])
        #
        # Using strain_stress_object_1
        #
        facade = StressFacade(strain_stress_object_1['stresses']['diagonal'])
        #
        # Case single value (errors assumed 0.0)
        facade.d_reference = 2.0
        assert_allclose(facade.d_reference.values, 2.0 * np.ones(facade.size))
        assert_allclose(facade.d_reference.errors, np.zeros(facade.size))
        #
        # Case single value and error, passing different types of objects
        for d_update in [(2.0, 0.20), [2.1, 0.21], np.array([2.2, 0.22])]:
            value, error = d_update
            facade.d_reference = d_update
            assert_allclose(facade.d_reference.values, value * np.ones(facade.size))
            assert_allclose(facade.d_reference.errors, error * np.ones(facade.size))
        #
        # Case scalar field with d_reference update for all sample points
        values = 1.0 + 0.1 * np.arange(facade.size)  # (1.0, 1.1,.., 1.9)
        d_update = ScalarFieldSample('d_reference', values, 0.1 * values, facade.x, facade.y, facade.z)
        facade.d_reference = d_update
        assert_allclose(facade.d_reference.values, values)
        assert_allclose(facade.d_reference.errors, 0.1 * values)
        #
        # Case scalar field with d_reference update for some sample points
        values = 1.0 + 0.1 * np.arange(facade.size)  # [1.0, 1.1,.., 1.9]
        errors = 0.1 * values  # [0.1, 0.11,..,0.19]
        facade.d_reference = (1.0, 0.0)  # "reset" d_reference
        assert_allclose(facade.d_reference.values, np.ones(facade.size))
        assert_allclose(facade.d_reference.errors, np.zeros(facade.size))
        indexes = [0, 2, 4, 6, 8]  # indexes of the sample points whose d_reference will be updated
        d_update = ScalarFieldSample('d_reference', values[indexes], errors,
                                     facade.x[indexes], facade.y[indexes], facade.z[indexes])
        facade.d_reference = d_update
        assert_allclose(facade.d_reference.values, [1.00, 1.00, 1.20, 1.00, 1.40, 1.00, 1.60, 1.00, 1.80, 1.00])
        assert_allclose(facade.d_reference.errors, [0.10, 0.00, 0.12, 0.00, 0.14, 0.00, 0.16, 0.00, 0.18, 0.00])

        # What about d_reference for each strain direction and single strain?
        # d_reference along the 11 direction
        d_reference_11 = facade._stress.strain11.get_d_reference().values
        assert_allclose(d_reference_11, [1.00, 1.00, 1.20, 1.00, 1.40, 1.00, 1.60, 1.00, nanf, nanf])
        # d_reference for the single strain field with run 1234
        d_reference_1234 = facade._stress.strain11.strains[0].get_d_reference().values
        assert_allclose(d_reference_1234, [1.00, 1.00, 1.20, 1.00, 1.40, 1.00, 1.60, 1.00])

        d_reference_22 = facade._stress.strain22.get_d_reference().values
        assert_allclose(d_reference_22, [nanf, 1.00, 1.20, 1.00, 1.40, 1.00, 1.60, 1.00, 1.80, nanf])
        d_reference_1235 = facade._stress.strain22.strains[0].get_d_reference().values
        assert_allclose(d_reference_1235, [1.00, 1.20, 1.00, 1.40])
        d_reference_1236 = facade._stress.strain22.strains[1].get_d_reference().values
        assert_allclose(d_reference_1236, [1.40, 1.00, 1.60, 1.00, 1.80])

        d_reference_33 = facade._stress.strain33.get_d_reference().values
        assert_allclose(d_reference_33, [nanf, nanf, 1.20, 1.00, 1.40, 1.00, 1.60, 1.00, 1.80, 1.00])
        d_reference_1237 = facade._stress.strain33.strains[0].get_d_reference().values
        assert_allclose(d_reference_1237, [1.20, 1.00, 1.40, 1.00, 1.60, 1.00, 1.80, 1.00])

    def test_peak_parameters(self, strain_stress_object_1):
        facade = StressFacade(strain_stress_object_1['stresses']['diagonal'])
        assert set(facade.peak_parameters) == {'Center', 'Height', 'FWHM', 'Mixing', 'A0', 'A1', 'Intensity'}

    def test_peak_parameter_field(self, strain_stress_object_1):
        r"""Retrieve the effective peak parameters for a particular run, or for a particular direction"""
        facade = StressFacade(strain_stress_object_1['stresses']['diagonal'])
        facade.selection = '11'
        with pytest.raises(ValueError) as exception_info:
            facade.peak_parameter('Center')
        assert 'Peak parameters can only be retrieved for run numbers' in str(exception_info.value)
        facade.selection = '1234'
        with pytest.raises(AssertionError) as exception_info:
            facade.peak_parameter('center')
        assert 'Peak parameter must be one of' in str(exception_info.value)
        facade.selection = '1234'
        expected = [100, 110, 120, 130, 140, 150, 160, 170, nanf, nanf]
        assert_allclose(facade.peak_parameter('Intensity').values, expected, equal_nan=True)
        facade.selection = '1235'
        expected = [nanf, 1.1, 1.2, 1.3, 1.4, nanf, nanf, nanf, nanf, nanf]
        assert_allclose(facade.peak_parameter('FWHM').values, expected, equal_nan=True)
        facade.selection = '1236'
        expected = [nanf, nanf, nanf, nanf, 14., 15., 16., 17., 18., nanf]
        assert_allclose(facade.peak_parameter('A0').values, expected, equal_nan=True)
        facade.selection = '1237'
        expected = [nanf, nanf, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
        assert_allclose(facade.peak_parameter('A1').values, expected, equal_nan=True)

    def test_peak_parameter_workspace(self, strain_stress_object_1):
        r"""Retrieve the effective peak parameters for a particular run, or for a particular direction"""
        facade = StressFacade(strain_stress_object_1['stresses']['diagonal'])
        facade.selection = '11'
        with pytest.raises(ValueError) as exception_info:
            facade.workspace('Center')
        assert 'Peak parameters can only be retrieved for run numbers' in str(exception_info.value)
        facade.selection = '1234'
        with pytest.raises(AssertionError) as exception_info:
            facade.workspace('center')
        assert 'Peak parameter must be one of' in str(exception_info.value)
        facade.workspace('Center')
        r"""
        assert_workspace(facade.workspace('Center'), [])
        """
        # TODO assertions for remaining selections
