
import pytest
from pyrs.core.summary_generator_stress import SummaryGeneratorStress
from pyrs.dataobjects.fields import StressField
from pyrs.dataobjects.fields import StrainField

EXPECTED_HEADER_STRESS_1320 = '''# Hidra Project Names = 1320
# Peak Tags = peak0
# Young's Modulus (E)[GPa] = 200
# Poisson Ratio = 0.3'''.split('\n')


@pytest.mark.parametrize('project_tags, expected_header',
                         [([1320, 1320, 1320], EXPECTED_HEADER_STRESS_1320)],
                         ids=['HB2B_STRESS_1320_CSV'])
def test_write_stress_csv(project_tags: str, expected_header: str):

    sample11 = StrainField('tests/data/HB2B_' + str(project_tags[0]) + '.h5')
    sample22 = StrainField('tests/data/HB2B_' + str(project_tags[1]) + '.h5')
    sample33 = StrainField('tests/data/HB2B_' + str(project_tags[2]) + '.h5')

    stress = StressField(sample11, sample22, sample33, 200, 0.3)

    stress_csv_filename = 'HB2B_'
    for project_tag in project_tags:
        stress_csv_filename += str(project_tag) + '_'
    stress_csv_filename = stress_csv_filename[:-1] + '.csv'

    stress_csv = SummaryGeneratorStress(stress_csv_filename, [stress])
    stress_csv.write_csv()

    with open(stress_csv_filename, 'r') as fcsv:
        lines_in = fcsv.read().splitlines()

    for i, expected_header_line in enumerate(EXPECTED_HEADER_STRESS_1320):
        assert(lines_in[i] == expected_header_line)
