from . import checkdatatypes
from contextlib import contextmanager
from mantid import ConfigService
from mantid.api import FileFinder
from mantid.simpleapi import mtd, GetIPTS, SaveNexusProcessed
import os
from pathlib import Path
from subprocess import check_output
from typing import Union

__all__ = ['get_ipts_dir', 'get_default_output_dir', 'get_input_project_file', 'get_nexus_file']


def save_mantid_nexus(workspace_name, file_name, title=''):
    """
    save workspace to NeXus for Mantid to import
    :param workspace_name:
    :param file_name:
    :param title:
    :return:
    """
    # check input
    checkdatatypes.check_file_name(file_name, check_exist=False,
                                   check_writable=True, is_dir=False)
    checkdatatypes.check_string_variable('Workspace title', title)

    # check workspace
    checkdatatypes.check_string_variable('Workspace name', workspace_name)
    if mtd.doesExist(workspace_name):
        SaveNexusProcessed(InputWorkspace=workspace_name,
                           Filename=file_name,
                           Title=title)
    else:
        raise RuntimeError('Workspace {0} does not exist in Analysis data service. Available '
                           'workspaces are {1}.'
                           ''.format(workspace_name, mtd.getObjectNames()))


def to_filepath(filename: Union[str, Path], check_exists: bool = True) -> Path:
    '''Asserts that a file exists and is a file.
    Raises an exception if anything is wrongither assumption is incorrect'''
    if not filename:  # empty value
        raise ValueError('Encountered empty filename')

    filepath = Path(filename)

    if check_exists:
        if not filepath.exists():
            raise IOError('File "{}" not found'.format(filename))
        if not filepath.is_file():
            raise IOError('Path "{}" is not a file'.format(filename))

    return filepath


def get_temp_directory():
    """
    get a temporary directory to write files
    :return:
    """
    # current workspace first
    temp_dir = os.getcwd()
    if os.access(temp_dir, os.W_OK):
        return temp_dir

    # /tmp/ second
    temp_dir = '/tmp/'
    if os.path.exists(temp_dir):
        return temp_dir

    # last solution: home directory
    temp_dir = os.path.expanduser('~')

    return temp_dir


# testing
# print (check_creation_date('__init__.py'))
@contextmanager
def archive_search():
    DEFAULT_FACILITY = 'default.facility'
    DEFAULT_INSTRUMENT = 'default.instrument'
    SEARCH_ARCHIVE = 'datasearch.searcharchive'
    HFIR = 'HFIR'
    HB2B = 'HB2B'

    # get the old values
    config = ConfigService.Instance()
    old_config = {}
    for property_name in [DEFAULT_FACILITY, DEFAULT_INSTRUMENT, SEARCH_ARCHIVE]:
        old_config[property_name] = config[property_name]

    # don't update things that are already set correctly
    if config[DEFAULT_FACILITY] == HFIR:
        del old_config[DEFAULT_FACILITY]
    else:
        config[DEFAULT_FACILITY] = HFIR

    if config[DEFAULT_INSTRUMENT] == HB2B:
        del old_config[DEFAULT_INSTRUMENT]
    else:
        config[DEFAULT_INSTRUMENT] = HB2B

    if HFIR in config[SEARCH_ARCHIVE]:
        del old_config[SEARCH_ARCHIVE]
    else:
        config[SEARCH_ARCHIVE] = HFIR

    try:
        # give back context
        yield

    finally:
        # set properties back to original values
        for property_name in old_config.keys():
            config[property_name] = old_config[property_name]


def __run_finddata(runnumber):
    '''This is a backup solution while the ...ORIG.nxs.h5 files are floating around'''
    result = check_output(['finddata', 'hb2b', str(runnumber)]).decode('utf-8').strip()
    if (not result) or (result == 'None'):
        raise RuntimeError('Failed to find HB2B_{} using "finddata"'.format(runnumber))
    return result


def get_ipts_dir(hint: Union[int, str, Path]) -> Path:
    """Get IPTS directory from run number. Throws an exception if the file wasn't found.

    Parameters
    ----------
    hint : int, str, Path
        run number or path to nexus file

    Returns
    -------
    str
        IPTS path: example '/HFIR/HB2B/IPTS-22731/', None for not supported IPTS
    """
    filepath = Path(str(hint))
    if filepath.exists() and filepath.parts[1] == 'HFIR':
        ipts = Path(*filepath.parts[:4])
    else:
        # try with GetIPTS
        try:
            with archive_search():
                ipts = Path(GetIPTS(RunNumber=hint, Instrument='HB2B'))
        except RuntimeError as e:
            print('GetIPTS failed:', e)
            # get the information from the nexus file
            nexusfile = get_nexus_file(hint)
            # take the first 3 directories
            ipts = get_ipts_dir(nexusfile)

    return ipts


def get_default_output_dir(run_number):
    """Get NeXus directory

    Parameters
    ----------
    run_number : int
        run number

    Returns
    -------
    str
        path to Nexus files ``/HFIR/IPTS-####/shared/manualreduce``

    """
    # this can generate an exception
    ipts_dir = get_ipts_dir(run_number)

    return os.path.join(ipts_dir, 'shared', 'manualreduce')


def get_input_project_file(run_number, preferredType='manual'):
    # this can generate an exception
    shared_dir = os.path.join(get_ipts_dir(run_number), 'shared')

    if not os.path.exists(shared_dir):
        raise RuntimeError('Shared directory "{}" does not exist'.format(shared_dir))

    # generate places to look for
    auto_dir = os.path.join(shared_dir, 'autoreduce')
    manual_dir = os.path.join(shared_dir, 'manualreduce')
    err_msg = 'Failed to find project file for run "{}" in "{}"'.format(run_number, shared_dir)

    preferredType = preferredType.lower()
    if preferredType.startswith('manual'):
        if os.path.exists(manual_dir):
            return manual_dir
        elif os.path.exists(auto_dir):
            return auto_dir
        else:
            raise RuntimeError(err_msg)
    elif preferredType.startswith('auto'):
        if os.path.exists(auto_dir):
            return auto_dir
        else:
            raise RuntimeError(err_msg)
    else:
        raise ValueError('Do not understand preferred type "{}"'.format(preferredType))


def get_nexus_file(run_number):
    try:
        with archive_search():
            nexus_file = FileFinder.findRuns('HB2B{}'.format(run_number))[0]
    except RuntimeError as e:
        print('ArchiveSearch failed:', e)
        nexus_file = __run_finddata(run_number)

    # return after `with` scope so cleanup is run
    return nexus_file


