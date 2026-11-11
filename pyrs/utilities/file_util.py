from . import checkdatatypes
from mantid.api import FileFinder
from mantid.simpleapi import mtd, GetIPTS, SaveNexusProcessed
from mantid.kernel import amend_config
import configparser
import os
from pathlib import Path
from subprocess import CalledProcessError, check_output
from typing import Optional, Union

__all__ = [
    "get_ipts_dir",
    "get_default_output_dir",
    "get_input_project_file",
    "get_nexus_file",
    "parse_eta_mask_angle",
]


def parse_eta_mask_angle(config: configparser.ConfigParser) -> Optional[float]:
    """Parse ``REDUCTION/ETA_MASK_ANGLE`` from an ``hb2b_autoreduction.ini`` config.

    ``configparser`` always returns strings, so the value must be coerced
    explicitly. This is the single source of truth shared by the autoreduction
    and live-reduction pipelines; both import this function so their config
    semantics cannot diverge.

    Args:
        config: Parsed configuration containing a ``REDUCTION`` section.

    Returns:
        The eta step in degrees, or ``None`` when the value is blank
        (meaning no out-of-plane/texture masking — in-plane reduction only).

    Raises:
        ValueError: If the value is non-empty but does not parse as a number.
            Failing loudly is intentional — the previous behaviour silently
            disabled texture reduction on a typo.

    Example:
        >>> import configparser
        >>> cfg = configparser.ConfigParser()
        >>> cfg.add_section("REDUCTION")
        >>> cfg["REDUCTION"]["ETA_MASK_ANGLE"] = "5"
        >>> parse_eta_mask_angle(cfg)
        5.0
    """
    raw = (config["REDUCTION"].get("ETA_MASK_ANGLE", "") or "").strip()
    if raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"ETA_MASK_ANGLE must be blank or a number, got {raw!r}")


def save_mantid_nexus(workspace_name, file_name, title=""):
    """
    save workspace to NeXus for Mantid to import
    :param workspace_name:
    :param file_name:
    :param title:
    :return:
    """
    # check input
    checkdatatypes.check_file_name(file_name, check_exist=False, check_writable=True, is_dir=False)
    checkdatatypes.check_string_variable("Workspace title", title)

    # check workspace
    checkdatatypes.check_string_variable("Workspace name", workspace_name)
    if mtd.doesExist(workspace_name):
        SaveNexusProcessed(InputWorkspace=workspace_name, Filename=file_name, Title=title)
    else:
        raise RuntimeError(
            "Workspace {0} does not exist in Analysis data service. Available workspaces are {1}.".format(
                workspace_name, mtd.getObjectNames()
            )
        )


def to_filepath(filename: Union[str, Path], check_exists: bool = True) -> Path:
    """Asserts that a file exists and is a file.
    Raises an exception if anything is wrongither assumption is incorrect"""
    if not filename:  # empty value
        raise ValueError("Encountered empty filename")

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
    temp_dir = "/tmp/"
    if os.path.exists(temp_dir):
        return temp_dir

    # last solution: home directory
    temp_dir = os.path.expanduser("~")

    return temp_dir


def __run_finddata(runnumber):
    """This is a backup solution while the ...ORIG.nxs.h5 files are floating around"""
    try:
        result = check_output(["finddata", "hb2b", str(runnumber)]).decode("utf-8").strip()
    except CalledProcessError as error:
        # not a RuntimeError, so the windows' error pop-ups would not catch it (#931)
        raise RuntimeError('Failed to find HB2B_{} using "finddata": {}'.format(runnumber, error)) from error
    if (not result) or (result == "None"):
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
    if filepath.exists() and filepath.parts[1] == "HFIR":
        ipts = Path(*filepath.parts[:4])
    else:
        # try with GetIPTS
        try:
            with amend_config(instrument="HB2B", facility="HFIR"):
                ipts = Path(GetIPTS(RunNumber=hint, Instrument="HB2B"))
        except RuntimeError as e:
            print("GetIPTS failed:", e)
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

    return os.path.join(ipts_dir, "shared", "manualreduce")


def get_input_project_file(run_number: Union[int, str], preferredType: str = "manual") -> str:
    """Return the reduction folder that holds the project file ``HB2B_<run_number>.h5``.

    Args:
        run_number: Run number.
        preferredType: ``"manual"`` looks in ``shared/manualreduce`` and then
            ``shared/autoreduce``; ``"auto"`` looks only in ``shared/autoreduce``.

    Returns:
        The folder (``str``) containing the run's project file.

    Raises:
        RuntimeError: If the IPTS ``shared`` folder does not exist, or no searched folder holds
            the run's project file (e.g. the run has not been reduced yet, #931).
        ValueError: If ``preferredType`` is neither manual nor auto.
    """
    preferredType = preferredType.lower()
    if preferredType.startswith("manual"):
        reductions = ["manualreduce", "autoreduce"]
    elif preferredType.startswith("auto"):
        reductions = ["autoreduce"]
    else:
        raise ValueError('Do not understand preferred type "{}"'.format(preferredType))

    # this can generate an exception
    shared_dir = os.path.join(get_ipts_dir(run_number), "shared")

    if not os.path.exists(shared_dir):
        raise RuntimeError('Shared directory "{}" does not exist'.format(shared_dir))

    # a folder that exists but lacks this run's file is not a match: its load would fail later (#931)
    project_name = "HB2B_{}.h5".format(run_number)
    folders = [os.path.join(shared_dir, reduction) for reduction in reductions]
    for folder in folders:
        if os.path.isfile(os.path.join(folder, project_name)):
            return folder

    raise RuntimeError(
        "No project file {} for run {} in {} (has the run been reduced yet?)".format(
            project_name, run_number, " or ".join(folders)
        )
    )


def get_nexus_file(run_number):
    try:
        with amend_config(instrument="HB2B", facility="HFIR"):
            found = FileFinder.findRuns("HB2B{}".format(run_number))
        if not found:
            # an empty result used to escape as IndexError, which the windows do not catch (#931)
            raise RuntimeError("No NeXus file found for HB2B_{}".format(run_number))
        nexus_file = found[0]
    except RuntimeError as e:
        print("ArchiveSearch failed:", e)
        nexus_file = __run_finddata(run_number)

    # return after `with` scope so cleanup is run
    return nexus_file
