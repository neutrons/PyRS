"""
`WorkspaceKey`: the identity of one input workspace within an NXstress entry.

The key replaced a bare tuple of `(name, value)` pairs with the run folded in as
one more pair. What must not change with it is the on-disk row order, which
`sort_values` decides: the run is ordered among the discriminator names under
its column's name, so a field sorting after `run_number` comes after the run.

Unit tier: no file I/O; `of` uses a synthetic workspace.
"""

import warnings
from collections.abc import Callable

import numpy as np
import pytest
from pydantic import ValidationError

from pyrs.core.workspaces import HidraWorkspace
from pyrs.utilities.NXstress._peaks import IndexedPeaks
from pyrs.utilities.NXstress._workspace_key import WorkspaceKey
from pyrs.utilities.pydantic_transition import validate_call_
from pyrs.utilities.unset import Unset


DISCRIMINATORS = (("direction", "11"), ("zone", "A"))


class TestOrder:
    def test_run_sorts_among_names_by_its_column_name(self):
        """'direction' < 'run_number' < 'zone': the order the fold always produced."""
        # Arrange
        key = WorkspaceKey(DISCRIMINATORS, run=1327)

        # Act
        pairs = key.as_pairs()

        # Assert
        assert pairs == (("direction", "11"), ("run_number", 1327), ("zone", "A"))
        assert key.sort_values() == ("11", 1327, "A")

    def test_without_a_run_only_discriminators_count(self):
        key = WorkspaceKey(DISCRIMINATORS)

        assert key.run is Unset
        assert key.as_pairs() == DISCRIMINATORS
        assert key.sort_values() == ("11", "A")

    def test_empty_key_sorts_first(self):
        assert WorkspaceKey().sort_values() == ()


class TestIdentity:
    def test_equal_keys_share_a_dict_slot(self):
        # Arrange
        groups = {WorkspaceKey(DISCRIMINATORS, run=1327): "first"}

        # Act
        groups[WorkspaceKey(DISCRIMINATORS, run=1327)] = "second"

        # Assert
        assert list(groups.values()) == ["second"]

    @pytest.mark.parametrize("run", [-1, 0, 1327])
    def test_a_run_never_equals_no_run(self, run):
        """`Unset` -- no run supplied to the key -- equals no run number."""
        assert WorkspaceKey(DISCRIMINATORS, run=run) != WorkspaceKey(DISCRIMINATORS)

    def test_run_is_keyword_only(self):
        """A positional second argument would read as a run where a later field was meant."""
        with pytest.raises(TypeError):
            WorkspaceKey(DISCRIMINATORS, 1327)

    def test_runs_tell_otherwise_equal_keys_apart(self):
        assert WorkspaceKey(DISCRIMINATORS, run=1327) != WorkspaceKey(DISCRIMINATORS, run=1328)


class TestDescribe:
    def test_names_every_pair_the_run_included(self):
        assert WorkspaceKey(DISCRIMINATORS, run=1327).describe() == (
            " for {'direction': '11', 'run_number': 1327, 'zone': 'A'}"
        )

    def test_an_empty_key_adds_nothing(self):
        assert WorkspaceKey().describe() == ""


class TestOf:
    @pytest.fixture
    def workspace(self, minimal_HidraWorkspace: Callable[..., HidraWorkspace]) -> HidraWorkspace:
        ws = minimal_HidraWorkspace(sub_runs=[1, 2, 3], with_instrument=True)
        points = ws.get_sub_runs().raw_copy()
        ws.set_sample_log("direction", points, np.array(["22"] * len(points)))
        ws.set_sample_log("run_number", points, np.full(len(points), 1331))
        return ws

    def test_carries_the_run_when_the_entry_does(self, workspace):
        assert WorkspaceKey.of(workspace, ("direction",), carries_run=True) == WorkspaceKey(
            (("direction", "22"),), run=1331
        )

    def test_omits_the_run_when_the_entry_does_not(self, workspace):
        assert WorkspaceKey.of(workspace, ("direction",), carries_run=False) == WorkspaceKey((("direction", "22"),))

    def test_equals_a_hand_built_key_whatever_the_name_order(self, workspace):
        """`of` sorts by name, so a key built in name order is the same key."""
        # Arrange
        points = workspace.get_sub_runs().raw_copy()
        workspace.set_sample_log("zone", points, np.array(["A"] * len(points)))

        # Act
        keys = {
            WorkspaceKey.of(workspace, names, carries_run=True)
            for names in (("zone", "direction"), ("direction", "zone"))
        }

        # Assert
        assert keys == {WorkspaceKey((("direction", "22"), ("zone", "A")), run=1331)}

    def test_a_workspace_of_several_runs_raises(self, workspace):
        points = workspace.get_sub_runs().raw_copy()
        workspace.set_sample_log("run_number", points, np.arange(len(points)) + 1331)

        with pytest.raises(ValueError, match="holds .* runs"):
            WorkspaceKey.of(workspace, ("direction",), carries_run=True)


class TestRefusals:
    """A malformed key would compare unequal to the same key built by `of`, silently."""

    def test_unsorted_discriminators_raise(self):
        with pytest.raises(ValueError, match="sorted by name"):
            WorkspaceKey((("zone", "A"), ("direction", "11")))

    def test_a_repeated_name_raises(self):
        with pytest.raises(ValueError, match="distinct"):
            WorkspaceKey((("direction", "11"), ("direction", "22")))

    @pytest.mark.parametrize("run", [np.int64(1327), True, 1327.0])
    def test_a_run_that_is_not_an_int_raises(self, run):
        with pytest.raises(TypeError, match="must be an `int` or `Unset`"):
            WorkspaceKey(DISCRIMINATORS, run=run)


def test_validated_signatures_accept_it_without_warning():
    """`validate_call_` reaches the key through `IndexedPeaks`, and pydantic cannot validate `Unset` itself."""
    # Recorded rather than escalated with "error": pydantic catches an escalated
    # warning while building the schema, so "error" would pass with or without the fix.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        @validate_call_
        def take(indexed: list[IndexedPeaks]) -> int:
            return len(indexed)

    assert [str(warning.message) for warning in caught] == []
    assert take([]) == 0


def test_validated_signatures_refuse_a_bare_tuple_key(minimal_PeakCollection):
    """The isinstance schema is a real check: the pre-`WorkspaceKey` tuple form is refused."""

    @validate_call_
    def take(indexed: list[IndexedPeaks]) -> int:
        return len(indexed)

    with pytest.raises(ValidationError):
        take([IndexedPeaks((("direction", "11"),), minimal_PeakCollection(N_subrun=3))])
