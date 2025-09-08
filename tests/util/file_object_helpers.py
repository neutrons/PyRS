from contextlib import contextmanager
from pathlib import Path

from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode

@contextmanager
def hidraProjectFile(file_path: Path, mode: HidraProjectFileMode = HidraProjectFileMode.READONLY) -> HidraProjectFile:
    # A context manager for the `HidraProjectFile` class.

    ## TODO: this really should be placed into the `HidraProjectFile` class itself, as a `@classmethod`.

    # __enter__:
    project_file = HidraProjectFile(file_path, mode)
    yield project_file
    
    # __exit__:
    project_file.close()
    
