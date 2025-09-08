from pathlib import Path

import pytest

# from util.peak_collection_helpers import createPeakCollection
# from util.workspace_helpers import loadHidraWorkspace
from util.file_object_helpers import hidraProjectFile

PROJECT_FILE_PATH = "/mnt/data0/workspaces/ORNL-work/PyRS/example/HB2B_2251.h5"

def test_createPeakCollection(createPeakCollection):
    peaks = createPeakCollection(
        peak_tag="Al 251540",
        peak_profile="Gaussian",
        background_type="Quadratic",
        wavelength=25.4,
        projectfilename=PROJECT_FILE_PATH,
        runnumber=12345,
        N_subrun=25    
    )
    
def test_loadHidraWorkspace(loadHidraWorkspace):
    ws = loadHidraWorkspace(file_path=PROJECT_FILE_PATH, name="test_workspace")
    assert ws.name == "test_workspace"
    
def test_hidraProjectFile():
    with hidraProjectFile(file_path=PROJECT_FILE_PATH) as projectFile:
        assert projectFile.name == PROJECT_FILE_PATH
