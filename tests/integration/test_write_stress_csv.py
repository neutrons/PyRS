
import pytest
from filecmp import cmp
from os import remove

from pyrs.peaks import PeakCollectionLite  # type: ignore
from pyrs.dataobjects.sample_logs import PointList
from pyrs.dataobjects.fields import StressField
from pyrs.dataobjects.fields import StrainField
from pyrs.core.summary_generator_stress import SummaryGeneratorStress


def strain_instantiator(name, values, errors, x, y, z):
    return StrainField(name,
                       peak_collection=PeakCollectionLite(name, strain=values, strain_error=errors),
                       point_list=PointList([x, y, z]))


def test_write_csv_empty_strain_filenames():

    with pytest.raises(RuntimeError) as exception_info:
        # strain that doesn't come from a project file
        X = [0.000, 1.000, 2.000, 3.000, 4.000, 5.000, 6.000, 7.000, 8.000, 9.000]
        Y = [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
        Z = [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]

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
        assert 'StrainField filenames in direction ' in str(exception_info.value)


def test_write_csv_none_stress():

    with pytest.raises(RuntimeError) as exception_info:
        SummaryGeneratorStress('dummy.csv',  None)
        assert 'Error: stress input must be of type StressField' in str(exception_info.value)


def test_write_csv_incorrect_filename(test_data_dir: str):

    with pytest.raises(RuntimeError) as exception_info:
        sample11 = StrainField(test_data_dir + '/HB2B_1320.h5')
        sample22 = StrainField(test_data_dir + '/HB2B_1320.h5')
        sample33 = StrainField(test_data_dir + '/HB2B_1320.h5')
        stress = StressField(sample11, sample22, sample33, 200, 0.3)
        SummaryGeneratorStress('not_csv',  stress)
        assert 'File name must end with extension ' in str(exception_info.value)


EXPECTED_FILE_SUMMARY_CSV_1320 = 'tests/data/HB2B_StressStrain_peak0_Summary_expected_1320.csv'


@pytest.mark.parametrize('project_tags, expected_file',
                         [([1320, 1320, 1320], EXPECTED_FILE_SUMMARY_CSV_1320)],
                         ids=['HB2B_1320_SUMMARY_CSV'])
def test_write_summary_csv(test_data_dir: str, project_tags: str, expected_file: str):

    sample11 = StrainField(test_data_dir + '/HB2B_{}.h5'.format(project_tags[0]))
    sample22 = StrainField(test_data_dir + '/HB2B_{}.h5'.format(project_tags[1]))
    sample33 = StrainField(test_data_dir + '/HB2B_{}.h5'.format(project_tags[2]))

    stress = StressField(sample11, sample22, sample33, 200, 0.3)

    stress_csv_filename = 'HB2B_StressStrain_peak0_Summary.csv'
    stress_csv = SummaryGeneratorStress(stress_csv_filename, stress)
    stress_csv.write_summary_csv()

    assert(cmp(stress_csv_filename, expected_file))
    # cleanup
    remove(stress_csv_filename)


EXPECTED_FILE_SUMMARY_CSV_1320_33Calculated =\
    'tests/data/HB2B_StressStrain_peak0_Summary_expected_1320_33Calculated.csv'


@pytest.mark.parametrize('project_tags, expected_file',
                         [([1320, 1320], EXPECTED_FILE_SUMMARY_CSV_1320_33Calculated)],
                         ids=['HB2B_1320_SUMMARY_CSV_33Calculated'])
def test_write_summary_33calculated_csv(test_data_dir: str, project_tags: str, expected_file: str):

    sample11 = StrainField(test_data_dir + '/HB2B_' + str(project_tags[0]) + '.h5')
    sample22 = StrainField(test_data_dir + '/HB2B_' + str(project_tags[1]) + '.h5')

    stress = StressField(sample11, sample22, None, 200, 0.3, stress_type='in-plane-strain')  # type: ignore

    stress_csv_filename = 'HB2B_StressStrain_peak0_Summary_33Calculated.csv'
    stress_csv = SummaryGeneratorStress(stress_csv_filename, stress)
    stress_csv.write_summary_csv()

    assert(cmp(stress_csv_filename, expected_file))
    # cleanup
    remove(stress_csv_filename)


def test_write_summary_33calculated_nan_csv(test_data_dir: str):

    sample11 = StrainField(test_data_dir + '/HB2B_1331.h5', peak_tag='peak0')
    sample22 = StrainField(test_data_dir + '/HB2B_1332.h5', peak_tag='peak0')

    stress = StressField(sample11, sample22, None, 200, 0.3, stress_type='in-plane-strain')  # type: ignore

    stress_csv_filename = 'HB2B_StressStrain_peak0_Summary_33Calculated_1331_1332.csv'
    stress_csv = SummaryGeneratorStress(stress_csv_filename, stress)
    stress_csv.write_summary_csv()
