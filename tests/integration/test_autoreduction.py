import os
from pyrs.core.nexus_conversion import NeXusConvertingApp
from pyrs.core.powder_pattern import ReductionApp
import pytest

nexusfile = 'tests/testdata/LaB6_10kev_35deg-00004_Rotated.h5'
nexusfile = '/HFIR/HB2B/IPTS-22731/nexus/HB2B_439.nxs.h5'


@pytest.mark.skipif((not os.path.exists(nexusfile)),
                    reason='File "{}" does not exist'.format(nexusfile))
def test_nexus_to_project():
    projectfile = '/tmp/HB2B_439.h5'

    # remove the project file if it currently exists
    if os.path.exists(projectfile):
        os.remove(projectfile)

    # do the conversion to sub-runs
    converter = NeXusConvertingApp(nexusfile)
    converter.convert()
    converter.save(projectfile)

    # tests for the created file
    assert os.path.exists(projectfile)
    # TODO add more tests

    # extract the powder patterns and add them to the project file
    reducer = ReductionApp(use_mantid_engine=False)
    # TODO should add versions for testing arguments:
    # instrument_file, calibration_file, mask, sub_runs
    reducer.load_project_file(projectfile)
    reducer.reduce_data()
    reducer.save_diffraction_data(projectfile)

    # tests for the created file
    assert os.path.exists(projectfile)
    # TODO add more tests

if __name__ == '__main__':
    pytest.main([__file__])
