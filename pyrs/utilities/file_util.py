from pathlib import Path
from typing import Union

__all__ = ['get_ipts_dir', 'get_default_output_dir', 'get_input_project_file', 'get_nexus_file']


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
