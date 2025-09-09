from pathlib import Path

from pyrs.projectfile import HidraProjectFile

import pytest

PROJECT_FILE = "HB2B_1628.h5"

def test_createPeakCollection(createPeakCollection):
    peaks = createPeakCollection(
        peak_tag="Al 251540",
        peak_profile="Gaussian",
        background_type="Quadratic",
        wavelength=25.4,
        projectfilename="/does/not/exist.h5",
        runnumber=12345,
        N_subrun=25    
    )
    
def test_load_HidraWorkspace(load_HidraWorkspace):
    ws = load_HidraWorkspace(
        file_name=PROJECT_FILE,
        name="test_workspace",
        load_raw_counts=False,
        load_reduced_diffraction=True
    )
    assert ws.name == "test_workspace"

def test_HidraProjectFile_context_manager(test_data_dir):
    project_file_path = Path(test_data_dir) / PROJECT_FILE
    with HidraProjectFile(project_file_path) as project_file:
        assert project_file.name == str(project_file_path)
