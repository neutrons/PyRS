import pytest
import numpy as np
from numpy.testing import assert_allclose

from pyrs.core.stress_facade import StressFacade
from pyrs.dataobjects.fields import ScalarFieldSample, StressField
from pyrs.peaks.peak_collection import to_microstrain

to_megapascal = StressField.to_megapascal
nanf = float('nan')
NAN = np.nan


def assert_workspace(facade, query, signal_array):
    r"""
    Set of assertions for data related to fixture strain_stress_object_1

    Parameters
    ----------
    facade: StressFacade
    query: str
        Data to be exported to MDHistoWorkspace (e.g. 'strain', 'FWHM')
    signal_array: list
        List of expected values
    """
    workspace = facade.workspace(query)
    assert workspace.id() == 'MDHistoWorkspace'
    dimension = workspace.getDimension(0)
    assert dimension.getUnits() == 'mm'
    # adding half a bin each direction since values from mdhisto are boundaries and constructor uses centers
    min_value, max_value = min(facade.x), max(facade.x)
    half_bin_width = (max_value - min_value) / (2 * (facade.size - 1))
    assert dimension.getMinimum() == pytest.approx(min_value - half_bin_width)
    assert dimension.getMaximum() == pytest.approx(max_value + half_bin_width)
    assert dimension.getNBins() == facade.size
    atols = {'strain': 1.0, 'stress': 1.0}  # different queries have different decimal accuracies
    assert_allclose(workspace.getSignalArray().ravel(), signal_array, equal_nan=True, atol=atols.get(query, 1.e-5))


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
        assert_allclose(facade.stress.values, 2 * np.array([300, 340, 380, 420, 460]), atol=1)
        facade.selection = '22'
        assert_allclose(facade.stress.values, 2 * np.array([400, 440, 480, 520, 560]), atol=1)
        facade.selection = '33'
        assert_allclose(facade.stress.values, 2 * np.array([500, 540, 580, 620, 660]), atol=1)

    def test_poisson_ratio(self, strain_stress_object_1):
        facade = StressFacade(strain_stress_object_1['stresses']['diagonal'])
        assert facade.poisson_ratio == pytest.approx(1. / 3)

    def test_poisson_ratio_setter(self, strain_stress_object_0):
        facade = StressFacade(strain_stress_object_0['stresses']['diagonal'])
        facade.poisson_ratio = 0.0
        assert facade.poisson_ratio == pytest.approx(0.0)
        for selection in ('11', '22', '33'):
            facade.selection = selection
            assert_allclose(facade.stress.values, to_megapascal(facade.youngs_modulus * facade.strain.values), atol=1)

    def test_stress_type(self, strain_stress_object_1):
        for stress_type in ('diagonal', 'in-plane-strain', 'in-plane-stress'):
            facade = StressFacade(strain_stress_object_1['stresses'][stress_type])
            assert facade.stress_type == stress_type

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
        assert_allclose(field.values, to_microstrain([0.01, 0.02, 0.03, 0.04]), atol=1)
        field_extended = facade._extend_to_stacked_point_list(field)
        nan = float('nan')
        assert_allclose(field_extended.values,
                        to_microstrain([nan, 0.01, 0.02, 0.03, 0.04, nan, nan, nan, nan, nan]), atol=1)

    def test_strain_field(self, strain_stress_object_1):
        r"""strains along for a particular direction or run number"""
        facade = StressFacade(strain_stress_object_1['stresses']['diagonal'])
        # direction 11 and components
        facade.selection = '11'
        expected = to_microstrain([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, nanf, nanf])
        assert_allclose(facade.strain.values, expected, equal_nan=True, atol=1)
        facade.selection = '1234'
        assert_allclose(facade.strain.values, expected, equal_nan=True, atol=1)
        # direction 22 and components
        facade.selection = '22'
        expected = to_microstrain([nanf, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, nanf])
        assert_allclose(facade.strain.values, expected, equal_nan=True, atol=1)
        facade.selection = '1235'
        expected = to_microstrain([nanf, 0.01, 0.02, 0.03, 0.04, nanf, nanf, nanf, nanf, nanf])
        facade.selection = '1236'
        expected = to_microstrain([nanf, nanf, nanf, nanf, 0.045, 0.05, 0.06, 0.07, 0.08, nanf])
        assert_allclose(facade.strain.values, expected, equal_nan=True, atol=1)
        # direction 33 and components
        facade.selection = '33'
        expected = to_microstrain([nanf, nanf, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09])
        assert_allclose(facade.strain.values, expected, equal_nan=True, atol=1)
        facade.selection = '1237'
        assert_allclose(facade.strain.values, expected, equal_nan=True, atol=1)

        facade = StressFacade(strain_stress_object_1['stresses']['in-plane-strain'])
        # direction 11 and components
        facade.selection = '11'
        expected = to_microstrain([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, nanf])
        assert_allclose(facade.strain.values, expected, equal_nan=True, atol=1)
        facade.selection = '1234'
        assert_allclose(facade.strain.values, expected, equal_nan=True, atol=1)
        # direction 22 and components
        facade.selection = '22'
        expected = to_microstrain([nanf, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
        assert_allclose(facade.strain.values, expected, equal_nan=True, atol=1)
        facade.selection = '1235'
        expected = to_microstrain([nanf, 0.01, 0.02, 0.03, 0.04, nanf, nanf, nanf, nanf])
        facade.selection = '1236'
        expected = to_microstrain([nanf, nanf, nanf, nanf, 0.045, 0.05, 0.06, 0.07, 0.08])
        assert_allclose(facade.strain.values, expected, equal_nan=True, atol=1)
        # direction 33 and components
        facade.selection = '33'
        expected = to_microstrain([0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00])
        assert_allclose(facade.strain.values, expected, equal_nan=True, atol=1)

        facade = StressFacade(strain_stress_object_1['stresses']['in-plane-stress'])
        # direction 11 and components
        facade.selection = '11'
        expected = to_microstrain([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, nanf])
        assert_allclose(facade.strain.values, expected, equal_nan=True, atol=1)
        facade.selection = '1234'
        assert_allclose(facade.strain.values, expected, equal_nan=True, atol=1)
        # direction 22 and components
        facade.selection = '22'
        expected = to_microstrain([nanf, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
        assert_allclose(facade.strain.values, expected, equal_nan=True, atol=1)
        facade.selection = '1235'
        expected = to_microstrain([nanf, 0.01, 0.02, 0.03, 0.04, nanf, nanf, nanf, nanf])
        facade.selection = '1236'
        expected = to_microstrain([nanf, nanf, nanf, nanf, 0.045, 0.05, 0.06, 0.07, 0.08])
        assert_allclose(facade.strain.values, expected, equal_nan=True, atol=1)
        # direction 33 and components
        facade.selection = '33'
        expected = to_microstrain([nanf, -0.02, -0.04, -0.06, -0.08, -0.10, -0.12, -0.14, nanf])

    def test_strain_workspace(self, strain_stress_object_1):
        r"""Export the strains to a MDHistoWorkspace"""
        facade = StressFacade(strain_stress_object_1['stresses']['diagonal'])
        for selection, expected in [('11', [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, nanf, nanf]),
                                    ('1234', [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, nanf, nanf]),
                                    ('22', [nanf, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, nanf]),
                                    ('1235', [nanf, 0.01, 0.02, 0.03, 0.04, nanf, nanf, nanf, nanf, nanf]),
                                    ('1236', [nanf, nanf, nanf, nanf, 0.045, 0.05, 0.06, 0.07, 0.08, nanf]),
                                    ('33', [nanf, nanf, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]),
                                    ('1237', [nanf, nanf, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09])]:
            facade.selection = selection
            assert_workspace(facade, 'strain', to_microstrain(expected))

        facade = StressFacade(strain_stress_object_1['stresses']['in-plane-strain'])
        for selection, expected in [('11', [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, nanf]),
                                    ('1234', [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, nanf]),
                                    ('22', [nanf, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]),
                                    ('1235', [nanf, 0.01, 0.02, 0.03, 0.04, nanf, nanf, nanf, nanf]),
                                    ('1236', [nanf, nanf, nanf, nanf, 0.045, 0.05, 0.06, 0.07, 0.08]),
                                    ('33', [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00])]:
            facade.selection = selection
            assert_workspace(facade, 'strain', to_microstrain(expected))

        facade = StressFacade(strain_stress_object_1['stresses']['in-plane-stress'])
        for selection, expected in [('11', [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, nanf]),
                                    ('1234', [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, nanf]),
                                    ('22', [nanf, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]),
                                    ('1235', [nanf, 0.01, 0.02, 0.03, 0.04, nanf, nanf, nanf, nanf]),
                                    ('1236', [nanf, nanf, nanf, nanf, 0.045, 0.05, 0.06, 0.07, 0.08]),
                                    ('33', [nanf, -0.02, -0.04, -0.06, -0.08, -0.10, -0.12, -0.14, nanf])]:
            facade.selection = selection
            assert_workspace(facade, 'strain', to_microstrain(expected))

    def test_stress_field(self, strain_stress_object_1):
        r"""Stresses along a particular direction. Also for a run number, when a direction contains only one run
        number"""
        facade = StressFacade(strain_stress_object_1['stresses']['diagonal'])
        facade.selection = '1234'
        with pytest.raises(ValueError) as exception_info:
            facade.stress.values
        assert 'Stress can only be computed for directions' in str(exception_info.value)

        for direction, expected in [('11', [nanf, nanf, 80, 120, 160, 200, 240, 280, nanf, nanf]),
                                    ('22', [nanf, nanf, 80, 120, 160, 200, 240, 280, nanf, nanf]),
                                    ('33', [nanf, nanf, 80, 120, 160, 200, 240, 280, nanf, nanf])]:
            facade.selection = direction
            assert_allclose(facade.stress.values, expected, atol=1)

        facade = StressFacade(strain_stress_object_1['stresses']['in-plane-strain'])
        for direction, expected in [('11', [nanf, 30, 60, 90, 120, 150, 180, 210, nanf]),
                                    ('22', [nanf, 30, 60, 90, 120, 150, 180, 210, nanf]),
                                    ('33', [nanf, 20, 40, 60, 80, 100, 120, 140, nanf])]:
            facade.selection = direction
            assert_allclose(facade.stress.values, expected, atol=1)

        facade = StressFacade(strain_stress_object_1['stresses']['in-plane-stress'])
        for direction, expected in [('11', [nanf, 30, 60, 90, 120, 150, 180, 210, nanf]),
                                    ('22', [nanf, 30, 60, 90, 120, 150, 180, 210, nanf]),
                                    ('33', [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00])]:
            facade.selection = direction
            assert_allclose(facade.stress.values, expected, atol=1)

    def test_stress_workspace(self, strain_stress_object_1):
        r"""Stresses along a particular direction. Also for a run number, when a direction contains only one run
        number"""
        facade = StressFacade(strain_stress_object_1['stresses']['diagonal'])

        facade.selection = '1234'
        with pytest.raises(ValueError) as exception_info:
            facade.workspace('stress')
        assert 'Stress can only be computed for directions' in str(exception_info.value)

        for direction, expected in [('11', [nanf, nanf, 80, 120, 160, 200, 240, 280, nanf, nanf]),
                                    ('22', [nanf, nanf, 80, 120, 160, 200, 240, 280, nanf, nanf]),
                                    ('33', [nanf, nanf, 80, 120, 160, 200, 240, 280, nanf, nanf])]:
            facade.selection = direction
            assert_workspace(facade, 'stress', expected)

        facade = StressFacade(strain_stress_object_1['stresses']['in-plane-strain'])
        for direction, expected in [('11', [nanf, 30, 60, 90, 120, 150, 180, 210, nanf]),
                                    ('22', [nanf, 30, 60, 90, 120, 150, 180, 210, nanf]),
                                    ('33', [nanf, 20, 40, 60, 80, 100, 120, 140, nanf])]:
            facade.selection = direction
            assert_workspace(facade, 'stress', expected)

        facade = StressFacade(strain_stress_object_1['stresses']['in-plane-stress'])
        for direction, expected in [('11', [nanf, 30, 60, 90, 120, 150, 180, 210, nanf]),
                                    ('22', [nanf, 30, 60, 90, 120, 150, 180, 210, nanf]),
                                    ('33', [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00])]:
            facade.selection = direction
            assert_workspace(facade, 'stress', expected)

    def test_d_reference_field(self, strain_stress_object_1):
        r"""Get the reference lattice spacing"""

        for stress_type in ('diagonal', 'in-plane-strain', 'in-plane-stress'):
            facade = StressFacade(strain_stress_object_1['stresses'][stress_type])
            assert_allclose(facade.d_reference.values, np.ones(facade.size))

        # "pollute" the reference spacing of run 1235
        stress = strain_stress_object_1['stresses']['diagonal']
        strain_single_1235 = stress.strain22.strains[0]
        strain_single_1235.set_d_reference([1.001, 0.1])
        facade = StressFacade(stress)
        with pytest.raises(AssertionError) as exception_info:
            facade.d_reference
        assert 'reference spacings are different on different directions' in str(exception_info.value)

    def test_d_reference_workspace(self, strain_stress_object_1):
        for stress_type in ('diagonal', 'in-plane-strain', 'in-plane-stress'):
            facade = StressFacade(strain_stress_object_1['stresses'][stress_type])
            assert_workspace(facade, 'd_reference', np.ones(facade.size))

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
        d_update = ScalarFieldSample('d_reference', values[indexes], errors[indexes],
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
        assert set(facade.peak_parameters) == {'d', 'Center', 'Height', 'FWHM', 'Mixing',
                                               'A0', 'A1', 'Intensity'}

    def test_peak_parameter_field(self, strain_stress_object_1):
        r"""Retrieve the effective peak parameters for a particular run, or for a particular direction"""
        facade = StressFacade(strain_stress_object_1['stresses']['diagonal'])

        facade.selection = '11'
        expected = [100, 110, 120, 130, 140, 150, 160, 170, nanf, nanf]
        assert_allclose(facade.peak_parameter('Intensity').values, expected, equal_nan=True)
        facade.selection = '1234'
        with pytest.raises(AssertionError) as exception_info:
            facade.peak_parameter('center')
        assert 'Peak parameter must be one of' in str(exception_info.value)
        facade.selection = '1234'
        expected = [100, 110, 120, 130, 140, 150, 160, 170, nanf, nanf]
        assert_allclose(facade.peak_parameter('Intensity').values, expected, equal_nan=True)

        facade.selection = '22'
        expected = [nanf, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, nanf]
        assert_allclose(facade.peak_parameter('FWHM').values, expected, equal_nan=True)
        facade.selection = '1235'
        expected = [nanf, 1.1, 1.2, 1.3, 1.4, nanf, nanf, nanf, nanf, nanf]
        assert_allclose(facade.peak_parameter('FWHM').values, expected, equal_nan=True)
        facade.selection = '1236'
        expected = [nanf, nanf, nanf, nanf, 14., 15., 16., 17., 18., nanf]
        assert_allclose(facade.peak_parameter('A0').values, expected, equal_nan=True)

        facade.selection = '33'
        expected = [nanf, nanf, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
        assert_allclose(facade.peak_parameter('A1').values, expected, equal_nan=True)
        facade.selection = '1237'
        expected = [nanf, nanf, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
        assert_allclose(facade.peak_parameter('A1').values, expected, equal_nan=True)

        facade = StressFacade(strain_stress_object_1['stresses']['in-plane-strain'])
        facade.selection = '33'
        with pytest.raises(ValueError) as exception_info:
            facade.peak_parameter('Intensity')
        assert 'Intensity not measured along 33 when in in-plane-strain'

        facade = StressFacade(strain_stress_object_1['stresses']['in-plane-stress'])
        facade.selection = '33'
        with pytest.raises(ValueError) as exception_info:
            facade.peak_parameter('FWHM')
        assert 'FWHM not measured along 33 when in in-plane-stress'

    def test_peak_parameter_workspace(self, strain_stress_object_1):
        r"""Retrieve the effective peak parameters for a particular run, or for a particular direction"""
        facade = StressFacade(strain_stress_object_1['stresses']['diagonal'])

        facade.selection = '11'
        expected = [100, 110, 120, 130, 140, 150, 160, 170, nanf, nanf]
        assert_workspace(facade, 'Intensity', expected)
        facade.selection = '1234'
        expected = [100, 110, 120, 130, 140, 150, 160, 170, nanf, nanf]
        assert_workspace(facade, 'Intensity', expected)

        facade.selection = '22'
        expected = [nanf, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, nanf]
        assert_workspace(facade, 'FWHM', expected)
        facade.selection = '1235'
        expected = [nanf, 1.1, 1.2, 1.3, 1.4, nanf, nanf, nanf, nanf, nanf]
        assert_workspace(facade, 'FWHM', expected)
        facade.selection = '1236'
        expected = [nanf, nanf, nanf, nanf, 14., 15., 16., 17., 18., nanf]
        assert_workspace(facade, 'A0', expected)

        facade.selection = '33'
        expected = [nanf, nanf, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
        assert_workspace(facade, 'A1', expected)
        facade.selection = '1237'
        expected = [nanf, nanf, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
        assert_workspace(facade, 'A1', expected)

        with pytest.raises(AssertionError) as exception_info:
            facade.workspace('center')
        assert 'Peak parameter must be one of' in str(exception_info.value)

    def test_d_field(self, strain_stress_object_1):
        r"""Retrieve the d spacing for a particular direction and for a particular run"""
        facade = StressFacade(strain_stress_object_1['stresses']['in-plane-strain'])
        # Peak parameters can only be retrieved for run numbers
        facade.selection = '11'
        expected = [1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, nanf]
        assert_allclose(facade.peak_parameter('d').values, expected, equal_nan=True)
        facade.selection = '1234'
        assert_allclose(facade.peak_parameter('d').values, expected, equal_nan=True)
        facade.selection = '22'
        expected = [nanf, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08]
        assert_allclose(facade.peak_parameter('d').values, expected, equal_nan=True)
        facade.selection = '1235'
        expected = [nanf, 1.01, 1.02, 1.03, 1.04, nanf, nanf, nanf, nanf]
        assert_allclose(facade.peak_parameter('d').values, expected, equal_nan=True)
        facade.selection = '1236'
        expected = [nanf, nanf, nanf, nanf, 1.045, 1.05, 1.06, 1.07, 1.08]
        assert_allclose(facade.peak_parameter('d').values, expected, equal_nan=True)
        facade.selection = '33'
        with pytest.raises(ValueError) as exception_info:
            facade.peak_parameter('d')
        assert 'd-spacing not measured along 33 when in in-plane-strain' in str(exception_info.value)

    def test_d_workspace(self, strain_stress_object_1):
        facade = StressFacade(strain_stress_object_1['stresses']['in-plane-strain'])

        for selection, expected in [('11', [1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, nanf]),
                                    ('1234', [1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, nanf]),
                                    ('22', [nanf, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08]),
                                    ('1235', [nanf, 1.01, 1.02, 1.03, 1.04, nanf, nanf, nanf, nanf]),
                                    ('1236', [nanf, nanf, nanf, nanf, 1.045, 1.05, 1.06, 1.07, 1.08])]:
            facade.selection = selection
            assert_workspace(facade, 'd', expected)

        facade.selection = '33'
        with pytest.raises(ValueError) as exception_info:
            facade.workspace('d')
        assert 'd-spacing not measured along 33 when in in-plane-strain' in str(exception_info.value)
