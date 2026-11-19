"""
`Unset` stays the one sentinel wherever it travels.

Every call site tests `x is Unset`, so the guarantee that matters is identity:
a copied, deep-copied or pickled `Unset` must still be *the* `Unset`, or a
default would silently stop being applied. `deepcopy` in particular is reached
whenever an object holding an unresolved argument is cloned.
"""

import copy
import pickle

import pytest

from pyrs.utilities.unset import Unset


@pytest.mark.parametrize(
    "transform",
    [copy.copy, copy.deepcopy, lambda value: pickle.loads(pickle.dumps(value))],
    ids=["copy", "deepcopy", "pickle"],
)
def test_identity_survives(transform):
    # Act
    result = transform(Unset)

    # Assert
    assert result is Unset


def test_repr_names_it():
    assert repr(Unset) == "Unset"


@pytest.mark.parametrize("value", [None, 0, -1, "", [], False])
def test_is_distinct_from_falsey_values(value):
    """`Unset` is not any value a caller might legitimately pass."""
    assert value != Unset
