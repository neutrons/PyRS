
import pytest
from filecmp import cmp
from os import remove

from pyrs.dataobjects.fields import ScalarFieldSample
from pyrs.dataobjects.fields import StressField
from pyrs.dataobjects.fields import StrainField
from pyrs.core.summary_generator_stress import SummaryGeneratorStress


def test_write_csv_empty_strain_filenames():

    with pytest.raises(RuntimeError):
        # strain that doesn't come from a project file
        X = [0.000, 1.000, 2.000, 3.000, 4.000, 5.000, 6.000, 7.000, 8.000, 9.000]
        Y = [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
        Z = [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]

        def strain_instantiator(name, values, errors, x, y, z):
            strain = StrainField()
            strain._field = ScalarFieldSample(name, values, errors, x, y, z)
            return strain

        strain11 = strain_instantiator('strain',
                                       [0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.080, 0.009],
                                       [0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009],
                                       X, Y, Z)
        strain22 = strain_instantiator('strain',
                                       [0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.080, 0.009],
                                       [0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009],
                                       X, Y, Z)
        strain33 = strain_instantiator('strain',
                                       [0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.080, 0.009],
                                       [0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009],
                                       X, Y, Z)

        stress = StressField(strain11, strain22, strain33, 200, 0.3)
        SummaryGeneratorStress('dummy.csv', stress)


def test_write_csv_none_stress():

    with pytest.raises(RuntimeError):
        SummaryGeneratorStress('dummy.csv',  None)


def test_write_csv_incorrect_filename():

    with pytest.raises(RuntimeError):
        sample11 = StrainField('tests/data/HB2B_1320.h5')
        sample22 = StrainField('tests/data/HB2B_1320.h5')
        sample33 = StrainField('tests/data/HB2B_1320.h5')
        stress = StressField(sample11, sample22, sample33, 200, 0.3)
        SummaryGeneratorStress('not_csv',  stress)


EXPECTED_FILE_SUMMARY_CSV_1320 = 'tests/data/HB2B_StressStrain_peak0_Summary_expected_1320.csv'


@pytest.mark.parametrize('project_tags, expected_file',
                         [([1320, 1320, 1320], EXPECTED_FILE_SUMMARY_CSV_1320)],
                         ids=['HB2B_1320_SUMMARY_CSV'])
def test_write_summary_csv(project_tags: str, expected_file: str):

    sample11 = StrainField('tests/data/HB2B_' + str(project_tags[0]) + '.h5')
    sample22 = StrainField('tests/data/HB2B_' + str(project_tags[1]) + '.h5')
    sample33 = StrainField('tests/data/HB2B_' + str(project_tags[2]) + '.h5')

    stress = StressField(sample11, sample22, sample33, 200, 0.3)

    stress_csv_filename = 'HB2B_StressStrain_peak0_Summary.csv'
    stress_csv = SummaryGeneratorStress(stress_csv_filename, stress)
    stress_csv.write_summary_csv()

    assert(cmp(stress_csv_filename, expected_file))
    # cleanup
    remove(stress_csv_filename)
