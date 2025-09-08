
from pathlib import Path
import numpy as np

from pyrs.core.workspaces import HidraWorkspace
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode

import pytest
from util.file_object_helpers import hidraProjectFile


@pytest.fixture
def loadHidraWorkspace(*, 
    file_path: Path,
    name: str
    ):
    # This fixture loads a `HidraWorkspace` instance from a `HidraProject`-format file.

    def _init() -> HidraWorkspace:
        ws = HidraWorkspace(name)
        with hidraProjectFile(file_path, mode=HidraProjectFileMode.READONLY) as project_file:
            ws.load_hidra_project(project_file, load_raw_counts=True, load_reduced_diffraction=True)
        return ws
        
    yield _init
    
    # teardown follows
    pass
