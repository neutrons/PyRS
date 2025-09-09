
from pathlib import Path
import numpy as np

from pyrs.core.workspaces import HidraWorkspace
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode

import pytest
from util.peak_collection_helpers import createPeakCollection


@pytest.fixture
def load_HidraWorkspace(test_data_dir):
    # This fixture loads a `HidraWorkspace` instance from a `HidraProject`-format file.

    def _init(*, 
    file_name: str,
    name: str,
    load_raw_counts=True,
    load_reduced_diffraction=True
    ) -> HidraWorkspace:
        file_path = Path(test_data_dir) / file_name
        ws = HidraWorkspace(name)
        with HidraProjectFile(file_path, mode=HidraProjectFileMode.READONLY) as project_file:
            ws.load_hidra_project(
                project_file,
                load_raw_counts=load_raw_counts,
                load_reduced_diffraction=load_reduced_diffraction
            )
        return ws
        
    yield _init
    
    # teardown follows
    pass

