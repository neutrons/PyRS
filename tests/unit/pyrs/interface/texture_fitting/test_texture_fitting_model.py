"""`_whole_number`: a `run_number` log value, in the forms project files store it.

Seen in `tests/data`: bytes holding an integer (`b'1327'`), and float32 holding a whole
number in newer reductions (`HB2B_1017.h5`: 1017.0). Mantid writes the log as text.
"""

import numpy as np
import pytest

from pyrs.interface.texture_fitting.texture_fitting_model import _whole_number


@pytest.mark.parametrize(
    "value",
    [np.bytes_(b"1327"), "1327", np.float32(1327.0), 1327],
    ids=["bytes", "str", "float32", "int"],
)
def test_stored_forms_of_a_run_number_are_read(value):
    assert _whole_number(value) == 1327


@pytest.mark.parametrize("value", [np.float32(1327.5), b"HB2B", ""], ids=["fraction", "word", "empty"])
def test_anything_else_is_no_run_number(value):
    assert _whole_number(value) is None
