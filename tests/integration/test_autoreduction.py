import os
from pyrs.core.nexus_conversion import NeXusConvertingApp
from pyrs.core.powder_pattern import ReductionApp
import pytest


@pytest.mark.parametrize('nexusfile, projectfile',
                         [('/HFIR/HB2B/IPTS-22731/nexus/HB2B_439.nxs.h5', 'HB2B_439.h5'),
                          # A good peak
                          ('/HFIR/HB2B/IPTS-22731/nexus/HB2B_938.nxs.h5', 'HB2B_938.h5'),
                          # Vanadium
                          ('/HFIR/HB2B/IPTS-22731/nexus/HB2B_931.nxs.h5', 'HB2B_931.h5')],
                         ids=('HB2B_439', 'HB2B_938', 'HB2B_931'))
def test_nexus_to_project(nexusfile, projectfile):
    if not os.path.exists(nexusfile):
        pytest.skip('File "{}" does not exist'.format(nexusfile))

    # remove the project file if it currently exists
    if os.path.exists(projectfile):
        os.remove(projectfile)

    # do the conversion to sub-runs
    converter = NeXusConvertingApp(nexusfile)
    converter.convert()
    converter.save(projectfile)

    # tests for the created file
    assert os.path.exists(projectfile), 'Project file {} does not exist'.format(projectfile)
    # TODO add more tests

    # extract the powder patterns and add them to the project file
    reducer = ReductionApp(use_mantid_engine=False)
    # TODO should add versions for testing arguments:
    # instrument_file, calibration_file, mask, sub_runs
    reducer.load_project_file(projectfile)
    reducer.reduce_data(sub_runs=None, instrument_file=None, calibration_file=None, mask=None)
    reducer.save_diffraction_data(projectfile)

    # tests for the created file
    assert os.path.exists(projectfile)
    # TODO add more tests


@pytest.mark.parametrize('project_file, van_project_file, target_project_file',
                         [('data/HB2B_938.h5', 'data/HB2B_931.h5', 'HB2B_938_van.h5')],
                         ids=['HB2B_930V'])
def test_apply_vanadium(project_file, van_project_file, target_project_file):
    """Test applying vanadium to the raw data in project file

    Parameters
    ----------
    project_file : str
        raw HiDRA project file to convert to 2theta pattern
    van_project_file : str
        raw HiDra vanadium file
    target_project_file : str
        target HiDRA

    Returns
    -------

    """
    # Check files' existence
    assert os.path.exists(project_file)
    assert os.path.exists(van_project_file)

    # Load data
    # extract the powder patterns and add them to the project file
    reducer = ReductionApp(use_mantid_engine=False)
    # TODO should add versions for testing arguments:
    # instrument_file, calibration_file, mask, sub_runs
    reducer.load_project_file(project_file)
    reducer.reduce_data(sub_runs=None, instrument_file=None, calibration_file=None, mask=None,
                        van_file=van_project_file)
    reducer.save_diffraction_data(target_project_file)


if __name__ == '__main__':
    pytest.main([__file__])
