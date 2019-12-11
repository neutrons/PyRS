import os
from pyrs.core.nexus_conversion import NeXusConvertingApp
from pyrs.core.powder_pattern import ReductionApp
import pytest


def checkFileExists(filename, feedback):
    '''``feedback`` should be 'skip' to skip the test if it doesn't exist,
    or 'assert' to throw an AssertionError if the file doesn't exist
    '''
    if os.path.exists(filename):
        return

    message = 'File "{}" does not exist'.format(filename)
    if feedback == 'skip':
        pytest.skip(message)
    elif feedback == 'assert':
        raise AssertionError(message)
    else:
        raise ValueError('Do not know how to give feedback={}'.format(feedback))


def convertNeXusToProject(nexusfile, projectfile, skippable):
    if skippable:
        checkFileExists(nexusfile, feedback='skip')
    else:
        checkFileExists(nexusfile, feedback='assert')

    # remove the project file if it currently exists
    if os.path.exists(projectfile):
        os.remove(projectfile)

    converter = NeXusConvertingApp(nexusfile)
    converter.convert(start_time=0)
    converter.save(projectfile)

    # tests for the created file
    assert os.path.exists(projectfile), 'Project file {} does not exist'.format(projectfile)


def addPowderToProject(projectfile, use_mantid_engine=False):
    checkFileExists(projectfile, feedback='assert')

    # extract the powder patterns and add them to the project file
    reducer = ReductionApp(use_mantid_engine=use_mantid_engine)
    # TODO should add versions for testing arguments: instrument_file, calibration_file, mask, sub_runs
    reducer.load_project_file(projectfile)
    reducer.reduce_data(sub_runs=None, instrument_file=None, calibration_file=None, mask=None)
    reducer.save_diffraction_data(projectfile)

    # tests for the created file
    assert os.path.exists(projectfile)


@pytest.mark.parametrize('nexusfile, projectfile',
                         [('/HFIR/HB2B/IPTS-22731/nexus/HB2B_439.nxs.h5', 'HB2B_439.h5'),  # file when reactor was off
                          ('/HFIR/HB2B/IPTS-22731/nexus/HB2B_931.nxs.h5', 'HB2B_931.h5'),  # Vanadium
                          ('data/HB2B_938.nxs.h5', 'HB2B_938.h5')],  # A good peak
                         ids=('HB2B_439', 'HB2B_931', 'RW_938'))
def test_nexus_to_project(nexusfile, projectfile):
    """Test converting NeXus to project and convert to diffraction pattern

    Note: project file cannot be the same as NeXus file as the output file will be
    removed by pytest

    Parameters
    ----------
    nexusfile
    projectfile

    Returns
    -------

    """
    # convert the nexus file to a project file and do the "simple" checks
    convertNeXusToProject(nexusfile, projectfile, skippable=True)

    # extract the powder patterns and add them to the project file
    addPowderToProject(projectfile, use_mantid_engine=False)

    # cleanup
    os.remove(projectfile)


def test_split_log_time_average():
    """(Integration) test on doing proper time average on split sample logs

    Run-1086 was measured with moving detector (changing 2theta value) along sub runs.

    Returns
    -------

    """
    # Convert the NeXus to project
    nexus_file = '/HFIR/HB2B/IPTS-22731/nexus/HB2B_1086.nxs.h5'
    project_file = 'HB2B_1086.h5'
    convertNeXusToProject(nexus_file, project_file, skippable=True)


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
    checkFileExists(project_file, feedback='assert')
    checkFileExists(van_project_file, feedback='assert')

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
