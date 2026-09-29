"""NXstress append mode: growing an `NXentry` that is already on disk.

A write targets an entry by number, and whether that entry already exists is
what decides between creating it and growing it. Growing is a **tail-append**:
the incoming batch is sorted among itself and added after each dataset's current
end, and nothing already on disk is read back, reordered or rewritten. The
resulting file is "locally sorted, globally segmented" rather than globally
sorted, which is all `_Peaks.peakCollectionRanges` has ever required.

These tests cover the round trip, entry targeting, the three-way
Case A / Case B / duplicate classification, every precondition that makes a
rejected append a true no-op, and the one property the whole design rests on --
that each per-scan-point dataset was written resizable in the first place.

Spec: `plans/NXstress-prod/04c-nxstress-append.md`.
"""

from collections.abc import Callable
import hashlib
from pathlib import Path

import h5py
import numpy as np
import yaml
import pytest
from nexusformat.nexus import NXentry, NXfield, nxopen

from pyrs.core.workspaces import HidraWorkspace
from pyrs.peaks.peak_collection import PeakCollection
from pyrs.utilities.NXstress._definitions import DEFAULT_TAG, appendable, tail_append
from pyrs.utilities.NXstress.NXstress import NXstress


def configure(config_override, yaml_text: str) -> None:
    """Apply an `nxstress` config override that production modules can actually see.

    Goes through `config_override` rather than `default_config`: the latter swaps
    the `Config` singleton, leaving every module that bound it the documented way
    reading the previous instance. See that fixture's docstring.
    """
    config_override(yaml.safe_load(yaml_text))


def workspace(
    minimal_HidraWorkspace: Callable[..., HidraWorkspace],
    minimal_PeakCollection: Callable[..., PeakCollection],
    *,
    direction: str,
    sub_runs: tuple,
    peak_tag: str = "Fe110",
    with_raw_counts: bool = False,
    **kwargs,
) -> tuple[HidraWorkspace, list[PeakCollection]]:
    """One workspace with a `direction` discriminator and its peak collection."""
    points = np.array(sub_runs)
    ws = minimal_HidraWorkspace(
        with_instrument=True, with_masks=True, sub_runs=points, with_raw_counts=with_raw_counts, **kwargs
    )
    ws.set_sample_log("direction", points, np.array([direction] * len(points)))
    return ws, [minimal_PeakCollection(N_subrun=len(points), peak_tag=peak_tag, sub_runs=points)]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scan_points(path: Path, group: str = "SAMPLE_DESCRIPTION", entry: str = "entry") -> list:
    with h5py.File(path, "r") as f:
        return f[f"{entry}/{group}/scan_point"][()].tolist()


@pytest.fixture
def discriminated(config_override):
    """Config with `direction` as the discriminator -- the precondition for any append."""
    configure(config_override, "nxstress:\n  discriminator_fields: ['direction']\n")
    return config_override


@pytest.fixture
def written(discriminated, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path):
    """A one-workspace entry on disk, ready to be appended to."""
    path = tmp_path / "entry.nxs"
    ws, peaks = workspace(minimal_HidraWorkspace, minimal_PeakCollection, direction="11", sub_runs=(1, 2, 3))
    with NXstress(path, "w") as nx:
        nx.write([ws], [peaks])
    return path


class TestRoundTrip:
    def test_appended_workspace_reads_back_with_its_own_scan_points(
        self, written, minimal_HidraWorkspace, minimal_PeakCollection
    ):
        """A Case-A append produces an entry that splits back into two workspaces."""
        # Arrange
        ws, peaks = workspace(minimal_HidraWorkspace, minimal_PeakCollection, direction="22", sub_runs=(4, 5, 6))

        # Act
        with NXstress(written, "a") as nx:
            nx.write([ws], [peaks])
        with NXstress(written, "r") as nx:
            workspaces, peakss = nx.read()

        # Assert
        assert len(workspaces) == 2
        assert [w.get_sub_runs().raw_copy().tolist() for w in workspaces] == [[1, 2, 3], [4, 5, 6]]
        assert [w.direction for w in workspaces] == ["11", "22"]
        assert [len(collections) for collections in peakss] == [1, 1]

    def test_every_position_aligned_group_grows_by_the_same_count(
        self, discriminated, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path
    ):
        """A partial append would desynchronize the entry with no error to signal it.

        The family is *discovered* from the pre-append file rather than listed
        here: a hand-maintained list of paths is exactly what leaves a
        newly-added field uncovered, and this is the test where that matters
        most. Raw counts on both sides, so `input_data` is in the set.
        """
        # Arrange
        path = tmp_path / "counts.nxs"
        first, first_peaks = workspace(
            minimal_HidraWorkspace,
            minimal_PeakCollection,
            direction="11",
            sub_runs=(1, 2, 3),
            with_raw_counts=True,
        )
        with NXstress(path, "w") as nx:
            nx.write([first], [first_peaks])
        family = sorted(TestWriterEmitsResizableDatasets.per_scan_point_datasets(path, n_scan=3))
        assert family, "the sweep matched no datasets; its scan-point heuristic has drifted"
        ws, peaks = workspace(
            minimal_HidraWorkspace,
            minimal_PeakCollection,
            direction="22",
            sub_runs=(4, 5, 6),
            with_raw_counts=True,
        )

        # Act
        with NXstress(path, "a") as nx:
            nx.write([ws], [peaks])

        # Assert -- every member of the family discovered above is now 6 long
        with h5py.File(path, "r") as f:
            lengths = {name: f[name].shape[0] for name in family}
        assert set(lengths.values()) == {6}, lengths
        # And the sweep really did reach all three families, not just one.
        assert any(name.startswith("entry/peaks/") for name in family)
        assert any("DIFFRACTOGRAM" in name for name in family)
        assert any(name.startswith("entry/input_data/") for name in family)
        assert "entry/start_time" in family

    def test_file_is_locally_sorted_globally_segmented(self, written, minimal_HidraWorkspace, minimal_PeakCollection):
        """The appended block FOLLOWS the first; it is not interleaved into it."""
        # Arrange -- scan points chosen so a global sort would interleave the two blocks
        ws, peaks = workspace(minimal_HidraWorkspace, minimal_PeakCollection, direction="02", sub_runs=(4, 5, 6))

        # Act
        with NXstress(written, "a") as nx:
            nx.write([ws], [peaks])

        # Assert -- '02' sorts BEFORE '11', so a re-sorted file would put it first
        with h5py.File(written, "r") as f:
            directions = [v.decode() for v in f["entry/peaks/direction"][()]]
            points = f["entry/peaks/scan_point"][()].tolist()
        assert directions == ["11", "11", "11", "02", "02", "02"]
        assert points == [1, 2, 3, 4, 5, 6]

    def test_existing_rows_are_not_rewritten(self, written, minimal_HidraWorkspace, minimal_PeakCollection):
        """Append grows the arrays; it never touches what is already in them."""
        # Arrange
        with h5py.File(written, "r") as f:
            before = f["entry/FIT/peak_parameters/center"][()].copy()
        ws, peaks = workspace(minimal_HidraWorkspace, minimal_PeakCollection, direction="22", sub_runs=(4, 5, 6))

        # Act
        with NXstress(written, "a") as nx:
            nx.write([ws], [peaks])

        # Assert
        with h5py.File(written, "r") as f:
            after = f["entry/FIT/peak_parameters/center"][()]
        assert np.array_equal(after[: len(before)], before)

    def test_two_workspaces_appended_in_one_call(self, written, minimal_HidraWorkspace, minimal_PeakCollection):
        """Append takes the same `list[HidraWorkspace]` shape `write` does."""
        # Arrange
        second, second_peaks = workspace(
            minimal_HidraWorkspace, minimal_PeakCollection, direction="22", sub_runs=(4, 5, 6)
        )
        third, third_peaks = workspace(
            minimal_HidraWorkspace, minimal_PeakCollection, direction="33", sub_runs=(7, 8, 9)
        )

        # Act
        with NXstress(written, "a") as nx:
            nx.write([second, third], [second_peaks, third_peaks])
        with NXstress(written, "r") as nx:
            workspaces, _ = nx.read()

        # Assert
        assert [w.direction for w in workspaces] == ["11", "22", "33"]
        assert [w.get_sub_runs().raw_copy().tolist() for w in workspaces] == [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]

    def test_repeated_appends_onto_a_multi_workspace_entry(
        self, written, minimal_HidraWorkspace, minimal_PeakCollection
    ):
        """Nothing about append is first-append-only; the entry keeps splitting."""
        # Arrange / Act -- three successive appends, each its own session
        for direction, points in (("22", (4, 5, 6)), ("33", (7, 8, 9)), ("44", (10, 11))):
            ws, peaks = workspace(minimal_HidraWorkspace, minimal_PeakCollection, direction=direction, sub_runs=points)
            with NXstress(written, "a") as nx:
                nx.write([ws], [peaks])

        # Assert
        with NXstress(written, "r") as nx:
            workspaces, peakss = nx.read()
        assert [w.direction for w in workspaces] == ["11", "22", "33", "44"]
        assert [len(w.get_sub_runs().raw_copy()) for w in workspaces] == [3, 3, 3, 2]
        assert all(len(collections) == 1 for collections in peakss)

    def test_raw_counts_survive_an_append_round_trip(
        self, discriminated, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path
    ):
        """`detector_counts` is the one dataset the architecture refuses to round-trip."""
        # Arrange
        path = tmp_path / "counts_roundtrip.nxs"
        first, first_peaks = workspace(
            minimal_HidraWorkspace,
            minimal_PeakCollection,
            direction="11",
            sub_runs=(1, 2, 3),
            with_raw_counts=True,
        )
        second, second_peaks = workspace(
            minimal_HidraWorkspace,
            minimal_PeakCollection,
            direction="22",
            sub_runs=(4, 5, 6),
            with_raw_counts=True,
        )
        with NXstress(path, "w") as nx:
            nx.write([first], [first_peaks])

        # Act
        with NXstress(path, "a") as nx:
            nx.write([second], [second_peaks])
        with NXstress(path, "r") as nx:
            workspaces, _ = nx.read()

        # Assert -- each workspace gets back its own counts, indexed by its own scan points
        for recovered, original, points in (
            (workspaces[0], first, (1, 2, 3)),
            (workspaces[1], second, (4, 5, 6)),
        ):
            for point in points:
                assert np.array_equal(recovered.get_detector_counts(point), original.get_detector_counts(point)), point

    def test_a_named_mask_grows_its_own_diffractogram_group(
        self, discriminated, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path
    ):
        """One DIFFRACTOGRAM group per mask, and every one of them grows."""

        # Arrange
        def masked(direction: str, points: tuple):
            sub_runs = np.array(points)
            ws = minimal_HidraWorkspace(
                with_instrument=True, with_masks=True, mask_names=("mask_a",), sub_runs=sub_runs
            )
            ws.set_sample_log("direction", sub_runs, np.array([direction] * len(sub_runs)))
            two_theta = np.tile(np.linspace(60.0, 120.0, 20), (len(sub_runs), 1))
            ones = np.ones((len(sub_runs), 20))
            ws.set_reduced_diffraction_data_set(
                two_theta, {None: ones, "mask_a": ones * 2}, {None: ones, "mask_a": ones}
            )
            collections = [
                minimal_PeakCollection(N_subrun=len(sub_runs), sub_runs=sub_runs, mask=mask)
                for mask in (DEFAULT_TAG, "mask_a")
            ]
            return ws, collections

        path = tmp_path / "masked.nxs"
        first, first_peaks = masked("11", (1, 2, 3))
        with NXstress(path, "w") as nx:
            nx.write([first], [first_peaks])
        second, second_peaks = masked("22", (4, 5, 6))

        # Act
        with NXstress(path, "a") as nx:
            nx.write([second], [second_peaks])

        # Assert -- both groups grew, and the index carries both masks per workspace
        with h5py.File(path, "r") as f:
            groups = sorted(name for name in f["entry/FIT"] if name.startswith("DIFFRACTOGRAM"))
            shapes = {name: f[f"entry/FIT/{name}/diffractogram"].shape for name in groups}
            masks = [v.decode() for v in f["entry/peaks/mask"][()]]
        assert groups == ["DIFFRACTOGRAM", "DIFFRACTOGRAM_mask_a"]
        assert set(shapes.values()) == {(6, 20)}, shapes
        assert masks == [DEFAULT_TAG] * 3 + ["mask_a"] * 3 + [DEFAULT_TAG] * 3 + ["mask_a"] * 3

    def test_a_workspace_lacking_a_mask_is_nan_filled_on_append(
        self, discriminated, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path
    ):
        """The mask check compares the UNION, so one input may lack a mask the other has.

        `_concatenated_diffraction` NaN-fills that input's rows so the group
        stays aligned with the scan-point axis. Reachable on an append exactly
        as on a fresh write, and asserted by counting NaN rows rather than by
        the absence of an exception.
        """
        # Arrange
        path = tmp_path / "nanfill.nxs"

        def ws_for(direction: str, points: tuple, masks: tuple):
            sub_runs = np.array(points)
            ws = minimal_HidraWorkspace(
                with_instrument=True, with_masks=True, mask_names=("mask_a",), sub_runs=sub_runs
            )
            ws.set_sample_log("direction", sub_runs, np.array([direction] * len(sub_runs)))
            two_theta = np.tile(np.linspace(60.0, 120.0, 20), (len(sub_runs), 1))
            ones = np.ones((len(sub_runs), 20))
            ws.set_reduced_diffraction_data_set(two_theta, {m: ones for m in masks}, {m: ones for m in masks})
            collections = [
                minimal_PeakCollection(N_subrun=len(sub_runs), sub_runs=sub_runs, mask=mask) for mask in (DEFAULT_TAG,)
            ]
            return ws, collections

        first, first_peaks = ws_for("11", (1, 2, 3), (None, "mask_a"))
        with NXstress(path, "w") as nx:
            nx.write([first], [first_peaks])
        # Two appended workspaces: the second reduced the default mask only.
        second, second_peaks = ws_for("22", (4, 5, 6), (None, "mask_a"))
        third, third_peaks = ws_for("33", (7, 8, 9), (None,))

        # Act
        with NXstress(path, "a") as nx:
            nx.write([second, third], [second_peaks, third_peaks])

        # Assert -- only the third workspace's rows are NaN in the named-mask group
        with h5py.File(path, "r") as f:
            masked = f["entry/FIT/DIFFRACTOGRAM_mask_a/diffractogram"][()]
            default = f["entry/FIT/DIFFRACTOGRAM/diffractogram"][()]
        assert masked.shape == default.shape == (9, 20)
        assert [int(np.isnan(row).all()) for row in masked] == [0] * 6 + [1] * 3
        assert not np.isnan(default).any()


class TestTextureEntryAppends:
    """An entry whose reductions are eta ROIs only can be appended to.

    A `_diff_data_set` key is a *reduction identifier*, not a detector-mask name:
    `reduction_manager` builds it as `"{mask_id}_eta_{eta_cent}"`, dropping either
    part that is absent. `nxstress_mask_names` injects `DEFAULT_TAG` because every
    reduction does use the default detector mask -- but the diffractogram keyspace
    must not be defaulted into, because a bare `DEFAULT_TAG` key means *the default
    mask with no eta ROI*, which an eta-resolved reduction never produced.

    Reading the incoming side through the mask function therefore invented a key
    the writer never emits, and every texture append was refused for a mismatch
    that did not exist -- with `self._invalid` set, so the instance died with it.
    Every other test here reduces under `{None: ...}`, where the two functions
    agree, which is why the whole module passed over it. See 04's Follow-up 8.
    """

    ETA_KEYS = ("eta_-5.0", "eta_0.0", "eta_5.0")

    @classmethod
    def _texture(cls, minimal_HidraWorkspace, minimal_PeakCollection, *, direction, sub_runs):
        """A workspace whose only reductions are eta ROIs -- no bare-default key."""
        points = np.array(sub_runs)
        ws = minimal_HidraWorkspace(
            with_instrument=True, with_masks=True, sub_runs=points, with_reduced_diffraction=False
        )
        ws.set_sample_log("direction", points, np.array([direction] * len(points)))
        two_theta = np.tile(np.linspace(60.0, 120.0, 20), (len(points), 1))
        ones = np.ones((len(points), 20))
        ws.set_reduced_diffraction_data_set(
            two_theta, {key: ones for key in cls.ETA_KEYS}, {key: ones.copy() for key in cls.ETA_KEYS}
        )
        peaks = [minimal_PeakCollection(N_subrun=len(points), sub_runs=points, mask=cls.ETA_KEYS[0])]
        return ws, peaks

    @pytest.fixture
    def textured(self, discriminated, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path):
        path = tmp_path / "texture.nxs"
        ws, peaks = self._texture(minimal_HidraWorkspace, minimal_PeakCollection, direction="11", sub_runs=(1, 2, 3))
        with NXstress(path, "w") as nx:
            nx.write([ws], [peaks])
        return path

    def test_no_bare_default_diffractogram_is_written(self, textured):
        """The precondition that makes this entry shaped differently from the others."""
        # Act
        with h5py.File(textured, "r") as f:
            groups = sorted(name for name in f["entry/FIT"] if name.startswith("DIFFRACTOGRAM"))

        # Assert
        assert groups == [f"DIFFRACTOGRAM_{key}" for key in sorted(self.ETA_KEYS)]
        assert "DIFFRACTOGRAM" not in groups

    def test_appends_and_reads_back_as_two_workspaces(self, textured, minimal_HidraWorkspace, minimal_PeakCollection):
        # Arrange
        ws, peaks = self._texture(minimal_HidraWorkspace, minimal_PeakCollection, direction="22", sub_runs=(4, 5, 6))

        # Act
        with NXstress(textured, "a") as nx:
            nx.write([ws], [peaks])
        with NXstress(textured, "r") as nx:
            workspaces, peakss = nx.read()

        # Assert
        assert [w.get_sub_runs().raw_copy().tolist() for w in workspaces] == [[1, 2, 3], [4, 5, 6]]
        assert [w.direction for w in workspaces] == ["11", "22"]
        assert all(set(w.reduction_masks) == set(self.ETA_KEYS) for w in workspaces)

    def test_a_genuinely_different_key_set_is_still_refused(
        self, textured, minimal_HidraWorkspace, minimal_PeakCollection
    ):
        """The check still does its job: only the spurious half of it was wrong."""
        # Arrange
        ws, peaks = self._texture(minimal_HidraWorkspace, minimal_PeakCollection, direction="22", sub_runs=(4, 5, 6))
        points = ws.get_sub_runs().raw_copy()
        two_theta = np.tile(np.linspace(60.0, 120.0, 20), (len(points), 1))
        ones = np.ones((len(points), 20))
        # One eta ROI the entry has no group for; the collection's own key stays valid,
        # so this reaches the key-set check rather than the earlier per-collection one.
        extra = (*self.ETA_KEYS, "eta_9.0")
        ws.set_reduced_diffraction_data_set(
            two_theta, {key: ones for key in extra}, {key: ones.copy() for key in extra}
        )
        before = digest(textured)

        # Act / Assert
        with NXstress(textured, "a") as nx:
            with pytest.raises(RuntimeError, match=r"reduced-diffraction keys do not match"):
                nx.write([ws], [peaks])
        assert digest(textured) == before


class TestEntryTargeting:
    def test_omitted_entry_number_grows_the_highest_entry(
        self, written, minimal_HidraWorkspace, minimal_PeakCollection
    ):
        # Arrange -- a second entry, so "highest" is not "only"
        ws2, peaks2 = workspace(minimal_HidraWorkspace, minimal_PeakCollection, direction="22", sub_runs=(7, 8, 9))
        with NXstress(written, "a") as nx:
            nx.write([ws2], [peaks2], entry_number=2)
        ws3, peaks3 = workspace(minimal_HidraWorkspace, minimal_PeakCollection, direction="33", sub_runs=(10, 11, 12))

        # Act
        with NXstress(written, "a") as nx:
            nx.write([ws3], [peaks3])

        # Assert -- entry_2 grew, entry did not
        assert scan_points(written) == [1, 2, 3]
        assert scan_points(written, entry="entry_2") == [7, 8, 9, 10, 11, 12]

    def test_entry_number_targets_an_earlier_entry(self, written, minimal_HidraWorkspace, minimal_PeakCollection):
        # Arrange
        ws2, peaks2 = workspace(minimal_HidraWorkspace, minimal_PeakCollection, direction="22", sub_runs=(7, 8, 9))
        with NXstress(written, "a") as nx:
            nx.write([ws2], [peaks2], entry_number=2)
        ws3, peaks3 = workspace(minimal_HidraWorkspace, minimal_PeakCollection, direction="33", sub_runs=(4, 5, 6))

        # Act -- explicitly target the FIRST entry, not the last
        with NXstress(written, "a", entry_number=1) as nx:
            nx.write([ws3], [peaks3])

        # Assert -- only the targeted entry grew
        assert scan_points(written) == [1, 2, 3, 4, 5, 6]
        assert scan_points(written, entry="entry_2") == [7, 8, 9]

    def test_entry_number_naming_no_existing_entry_writes_a_fresh_one(
        self, written, minimal_HidraWorkspace, minimal_PeakCollection
    ):
        # Arrange
        ws, peaks = workspace(minimal_HidraWorkspace, minimal_PeakCollection, direction="22", sub_runs=(4, 5, 6))

        # Act
        with NXstress(written, "a", entry_number=2) as nx:
            nx.write([ws], [peaks])

        # Assert
        with h5py.File(written, "r") as f:
            assert sorted(f.keys()) == ["entry", "entry_2"]
        assert scan_points(written) == [1, 2, 3]
        assert scan_points(written, entry="entry_2") == [4, 5, 6]

    def test_entry_number_skipping_the_next_free_number_raises(
        self, written, minimal_HidraWorkspace, minimal_PeakCollection
    ):
        """A gap is far likelier a typo than an intent, and the silent outcome is a stray entry."""
        # Arrange
        ws, peaks = workspace(minimal_HidraWorkspace, minimal_PeakCollection, direction="22", sub_runs=(4, 5, 6))
        before = digest(written)

        # Act / Assert
        with pytest.raises(ValueError, match=r"skips past the next free entry"):
            with NXstress(written, "a", entry_number=12) as nx:
                nx.write([ws], [peaks])
        # Refused after the file was opened for append, so the no-op guarantee applies
        # here as it does to every other refusal in this module. The two raising tests
        # below do not assert it because they raise in `__init__`, before any open.
        assert digest(written) == before

    def test_entry_number_is_rejected_for_a_read(self, written):
        with pytest.raises(ValueError, match=r"meaningless for a read"):
            NXstress(written, "r", entry_number=1)

    def test_entry_number_below_one_raises(self, written):
        with pytest.raises(ValueError, match=r"1-based"):
            NXstress(written, "a", entry_number=0)

    @pytest.mark.parametrize("entry_number", [0, -1])
    def test_read_entry_number_below_one_raises(self, written, entry_number):
        """`group_naming_scheme` names entry 1 `entry`, so these used to read entry 1."""
        with NXstress(written, "r") as nx:
            with pytest.raises(ValueError, match=r"1-based"):
                nx.read(entry_number)


class TestClassification:
    """Case A proceeds, Case B is unsupported, and a duplicate is a violated invariant."""

    def test_case_b_extending_an_existing_key_raises_not_implemented(
        self, written, minimal_HidraWorkspace, minimal_PeakCollection
    ):
        """More scan points under a key already on disk needs a mid-array insertion."""
        # Arrange -- same discriminator AND same peak tag as the entry already holds
        ws, peaks = workspace(minimal_HidraWorkspace, minimal_PeakCollection, direction="11", sub_runs=(4, 5, 6))
        before = digest(written)

        # Act / Assert
        with NXstress(written, "a") as nx:
            with pytest.raises(NotImplementedError, match=r"not supported"):
                nx.write([ws], [peaks])
        assert digest(written) == before

    def test_case_b_leaves_the_instance_usable(self, written, minimal_HidraWorkspace, minimal_PeakCollection):
        """Nothing was found wrong with the entry -- only unsupported."""
        # Arrange
        rejected, rejected_peaks = workspace(
            minimal_HidraWorkspace, minimal_PeakCollection, direction="11", sub_runs=(4, 5, 6)
        )
        accepted, accepted_peaks = workspace(
            minimal_HidraWorkspace, minimal_PeakCollection, direction="22", sub_runs=(7, 8, 9)
        )

        # Act
        with NXstress(written, "a") as nx:
            with pytest.raises(NotImplementedError):
                nx.write([rejected], [rejected_peaks])
            nx.write([accepted], [accepted_peaks])

        # Assert
        assert scan_points(written) == [1, 2, 3, 7, 8, 9]

    def test_duplicate_row_raises_runtime_error(self, written, minimal_HidraWorkspace, minimal_PeakCollection):
        """Same key AND an overlapping scan point is a violated invariant, not an unsupported one."""
        # Arrange
        ws, peaks = workspace(minimal_HidraWorkspace, minimal_PeakCollection, direction="11", sub_runs=(2, 3, 4))
        before = digest(written)

        # Act / Assert
        with NXstress(written, "a") as nx:
            with pytest.raises(RuntimeError, match=r"already present in this entry"):
                nx.write([ws], [peaks])
        assert digest(written) == before

    def test_duplicate_invalidates_the_instance(self, written, minimal_HidraWorkspace, minimal_PeakCollection):
        """After a violated invariant the caller must re-open rather than continue."""
        # Arrange
        duplicate, duplicate_peaks = workspace(
            minimal_HidraWorkspace, minimal_PeakCollection, direction="11", sub_runs=(2, 3, 4)
        )
        valid, valid_peaks = workspace(
            minimal_HidraWorkspace, minimal_PeakCollection, direction="22", sub_runs=(7, 8, 9)
        )

        # Act / Assert
        with NXstress(written, "a") as nx:
            with pytest.raises(RuntimeError, match=r"already present"):
                nx.write([duplicate], [duplicate_peaks])
            with pytest.raises(RuntimeError, match=r"no longer usable"):
                nx.write([valid], [valid_peaks])


class TestPreconditions:
    """Each rejects before any resize, so the entry is left byte-for-byte unchanged."""

    def test_workspace_with_no_peak_collections_raises(self, written, minimal_HidraWorkspace, minimal_PeakCollection):
        """Without a `PeakCollection` its discriminator value is unrecoverable on read."""
        # Arrange
        ws, _ = workspace(minimal_HidraWorkspace, minimal_PeakCollection, direction="22", sub_runs=(4, 5, 6))
        before = digest(written)

        # Act / Assert
        with NXstress(written, "a") as nx:
            with pytest.raises(RuntimeError, match=r"contribute no `PeakCollection`"):
                nx.write([ws], [[]])
        assert digest(written) == before

    def test_entry_without_a_discriminator_scheme_raises(
        self, config_override, tmp_path, minimal_HidraWorkspace, minimal_PeakCollection
    ):
        """There is no column to attach the new workspace's value to."""
        # Arrange -- written with no discriminator configured at all
        path = tmp_path / "bare.nxs"
        plain = minimal_HidraWorkspace(with_instrument=True, with_masks=True, sub_runs=np.array([1, 2, 3]))
        with NXstress(path, "w") as nx:
            nx.write([plain], [[minimal_PeakCollection(N_subrun=3, sub_runs=np.array([1, 2, 3]))]])
        before = digest(path)

        configure(config_override, "nxstress:\n  discriminator_fields: ['direction']\n")
        ws, peaks = workspace(minimal_HidraWorkspace, minimal_PeakCollection, direction="22", sub_runs=(4, 5, 6))

        # Act / Assert
        with NXstress(path, "a") as nx:
            with pytest.raises(RuntimeError, match=r"carries no discriminator columns"):
                nx.write([ws], [peaks])
        assert digest(path) == before

    def test_scan_point_already_in_the_entry_raises(self, written, minimal_HidraWorkspace, minimal_PeakCollection):
        """The reader attributes rows by scan-point VALUE, so a repeat cannot be attributed."""
        # Arrange -- a genuinely new key (Case A), but reusing scan point 3
        ws, peaks = workspace(
            minimal_HidraWorkspace,
            minimal_PeakCollection,
            direction="22",
            sub_runs=(3, 4, 5),
            peak_tag="Si111",
        )
        before = digest(written)

        # Act / Assert
        with NXstress(written, "a") as nx:
            with pytest.raises(RuntimeError, match=r"are already in this entry"):
                nx.write([ws], [peaks])
        assert digest(written) == before

    def test_mismatched_mask_set_raises(self, written, minimal_HidraWorkspace, minimal_PeakCollection):
        """A mask the entry has no DIFFRACTOGRAM for would need a group with no history."""
        # Arrange
        points = np.array([4, 5, 6])
        ws = minimal_HidraWorkspace(with_instrument=True, with_masks=True, mask_names=("mask_a",), sub_runs=points)
        ws.set_sample_log("direction", points, np.array(["22"] * 3))
        two_theta = np.tile(np.linspace(60.0, 120.0, 20), (3, 1))
        ones = np.ones((3, 20))
        ws.set_reduced_diffraction_data_set(two_theta, {None: ones, "mask_a": ones}, {None: ones, "mask_a": ones})
        peaks = [minimal_PeakCollection(N_subrun=3, sub_runs=points)]
        before = digest(written)

        # Act / Assert
        with NXstress(written, "a") as nx:
            with pytest.raises(RuntimeError, match=r"masks do not match"):
                nx.write([ws], [peaks])
        assert digest(written) == before

    def test_mismatched_retained_log_set_raises(self, written, minimal_HidraWorkspace, minimal_PeakCollection):
        """Appending a log the existing rows have no value for would desynchronize the group."""
        # Arrange
        ws, peaks = workspace(minimal_HidraWorkspace, minimal_PeakCollection, direction="22", sub_runs=(4, 5, 6))
        ws.set_sample_log("an_extra_log", np.array([4, 5, 6]), np.zeros(3))
        before = digest(written)

        # Act / Assert
        with NXstress(written, "a") as nx:
            with pytest.raises(RuntimeError, match=r"retained sample logs do not match"):
                nx.write([ws], [peaks])
        assert digest(written) == before

    def test_mismatched_detector_geometry_raises(self, written, minimal_HidraWorkspace, minimal_PeakCollection):
        """One NXentry describes one instrument configuration."""
        # Arrange
        from pyrs.core.instrument_geometry import DENEXDetectorGeometry

        ws, peaks = workspace(minimal_HidraWorkspace, minimal_PeakCollection, direction="22", sub_runs=(4, 5, 6))
        ws.set_instrument_geometry(
            DENEXDetectorGeometry(
                num_rows=4, num_columns=4, pixel_size_x=0.002, pixel_size_y=0.002, arm_length=3.0, calibrated=True
            )
        )
        before = digest(written)

        # Act / Assert
        with NXstress(written, "a") as nx:
            with pytest.raises(RuntimeError, match=r"disagree with the target entry on instrument geometry"):
                nx.write([ws], [peaks])
        assert digest(written) == before

    def test_mismatched_peak_profile_raises(self, written, minimal_HidraWorkspace, minimal_PeakCollection):
        """All peak collections in one entry share a single fit model."""
        # Arrange
        points = np.array([4, 5, 6])
        ws = minimal_HidraWorkspace(with_instrument=True, with_masks=True, sub_runs=points)
        ws.set_sample_log("direction", points, np.array(["22"] * 3))
        peaks = [minimal_PeakCollection(N_subrun=3, sub_runs=points, peak_tag="Si111", peak_profile="PseudoVoigt")]
        before = digest(written)

        # Act / Assert
        with NXstress(written, "a") as nx:
            with pytest.raises(RuntimeError, match=r"peak profile .* does not match"):
                nx.write([ws], [peaks])
        assert digest(written) == before

    def test_raw_counts_present_on_only_one_side_raises(self, written, minimal_HidraWorkspace, minimal_PeakCollection):
        """An entry written without raw counts cannot gain them for only some scan points."""
        # Arrange -- `written` has no raw counts; this input does
        ws, peaks = workspace(
            minimal_HidraWorkspace,
            minimal_PeakCollection,
            direction="22",
            sub_runs=(4, 5, 6),
            with_raw_counts=True,
        )
        before = digest(written)

        # Act / Assert
        with NXstress(written, "a") as nx:
            with pytest.raises(RuntimeError, match=r"does not have raw detector counts"):
                nx.write([ws], [peaks])
        assert digest(written) == before

    def test_mismatched_two_theta_width_raises(self, written, minimal_HidraWorkspace, minimal_PeakCollection):
        """Two reduction passes onto different 2-theta grids -- routine, and unappendable.

        Regression for the one refusal that was NOT pre-flighted: it fired from
        inside `tail_append`, by which point entry times, instrument, sample and
        both fit-parameter groups had already grown. The entry was left with 27
        datasets at the new length and three at the old one, and **read back
        without error**. The `digest` assertion is the whole point of this test;
        `pytest.raises` alone passed throughout.
        """
        # Arrange
        points = np.array([4, 5, 6])
        ws = minimal_HidraWorkspace(with_instrument=True, with_masks=True, sub_runs=points, n_two_theta=25)
        ws.set_sample_log("direction", points, np.array(["22"] * 3))
        peaks = [minimal_PeakCollection(N_subrun=3, sub_runs=points)]
        before = digest(written)

        # Act / Assert
        with NXstress(written, "a") as nx:
            with pytest.raises(RuntimeError, match=r"two-theta bin"):
                nx.write([ws], [peaks])
        assert digest(written) == before

    def test_mismatched_detector_pixel_count_raises(
        self, discriminated, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path
    ):
        """`detector_counts` is scan point by pixel; only the first axis grows."""
        # Arrange
        path = tmp_path / "pixels.nxs"
        first, first_peaks = workspace(
            minimal_HidraWorkspace,
            minimal_PeakCollection,
            direction="11",
            sub_runs=(1, 2, 3),
            with_raw_counts=True,
        )
        with NXstress(path, "w") as nx:
            nx.write([first], [first_peaks])
        second, second_peaks = workspace(
            minimal_HidraWorkspace,
            minimal_PeakCollection,
            direction="22",
            sub_runs=(4, 5, 6),
            with_raw_counts=True,
        )
        # A detector of a different size, built directly: the fixture's pixel
        # count is not a parameter.
        for point in (4, 5, 6):
            second.set_raw_counts(point, np.arange(32, dtype=np.int64))
        before = digest(path)

        # Act / Assert
        with NXstress(path, "a") as nx:
            with pytest.raises(RuntimeError, match=r"detector pixel"):
                nx.write([second], [second_peaks])
        assert digest(path) == before

    def test_mismatched_detector_shift_raises(self, written, minimal_HidraWorkspace, minimal_PeakCollection):
        """The second arm of the geometry check, and the one calibration state rides on."""
        # Arrange
        from pyrs.core.instrument_geometry import DENEXDetectorShift

        ws, peaks = workspace(minimal_HidraWorkspace, minimal_PeakCollection, direction="22", sub_runs=(4, 5, 6))
        ws.set_detector_shift(
            DENEXDetectorShift(
                shift_x=0.9, shift_y=0.8, shift_z=0.7, rotation_x=9.0, rotation_y=8.0, rotation_z=7.0, tth_0=6.0
            )
        )
        before = digest(written)

        # Act / Assert
        with NXstress(written, "a") as nx:
            with pytest.raises(RuntimeError, match=r"disagree with the target entry on detector shift"):
                nx.write([ws], [peaks])
        assert digest(written) == before

    def test_mismatched_background_function_raises(self, written, minimal_HidraWorkspace, minimal_PeakCollection):
        """The second arm of the fit-model check; `test_mismatched_peak_profile` covers the first."""
        # Arrange
        points = np.array([4, 5, 6])
        ws = minimal_HidraWorkspace(with_instrument=True, with_masks=True, sub_runs=points)
        ws.set_sample_log("direction", points, np.array(["22"] * 3))
        peaks = [minimal_PeakCollection(N_subrun=3, sub_runs=points, peak_tag="Si111", background_type="Quadratic")]
        before = digest(written)

        # Act / Assert
        with NXstress(written, "a") as nx:
            with pytest.raises(RuntimeError, match=r"background function .* does not match"):
                nx.write([ws], [peaks])
        assert digest(written) == before

    def test_mismatched_detector_mask_set_raises(self, written, minimal_HidraWorkspace, minimal_PeakCollection):
        """Masks are entry-wide and fixed-size; a differing set would be silently dropped."""
        # Arrange
        points = np.array([4, 5, 6])
        ws = minimal_HidraWorkspace(with_instrument=True, with_masks=True, mask_names=("mask_a",), sub_runs=points)
        ws.set_sample_log("direction", points, np.array(["22"] * 3))
        peaks = [minimal_PeakCollection(N_subrun=3, sub_runs=points)]
        before = digest(written)

        # Act / Assert
        with NXstress(written, "a") as nx:
            with pytest.raises(RuntimeError, match=r"detector masks do not match"):
                nx.write([ws], [peaks])
        assert digest(written) == before

    def test_a_legacy_fixed_size_entry_is_refused_as_a_whole(
        self, written, minimal_HidraWorkspace, minimal_PeakCollection
    ):
        """A file predating `growable` is refused up front, naming the datasets.

        Built by replacing one field with a fixed-size one rather than by
        reverting the writer, so the refusal is checked against the condition
        itself rather than against the code that avoids it.
        """
        # Arrange
        with nxopen(written, "rw") as root:
            sample = root["entry"]["SAMPLE_DESCRIPTION"]
            values = np.asarray(sample["vx"].nxdata)
            del sample["vx"]
            sample["vx"] = NXfield(values)
        before = digest(written)
        ws, peaks = workspace(minimal_HidraWorkspace, minimal_PeakCollection, direction="22", sub_runs=(4, 5, 6))

        # Act / Assert
        with NXstress(written, "a") as nx:
            with pytest.raises(RuntimeError, match=r"written at a fixed size"):
                nx.write([ws], [peaks])
        assert digest(written) == before

    def test_a_failure_during_mutation_invalidates_the_instance(
        self, written, minimal_HidraWorkspace, minimal_PeakCollection, monkeypatch
    ):
        """The case where the file IS damaged must not leave the caller able to continue.

        Every known cause of a mid-append failure is now pre-flighted, so this
        reaches the mutation phase's failure path the only way left: by making
        `_sample.tail_append` raise once the entry times and the instrument
        wavelength have already grown. The guarantee under test is the
        `except BaseException` invalidation in `_append`, not the trigger.
        """
        # Arrange
        from pyrs.utilities.NXstress import _sample

        ws, peaks = workspace(minimal_HidraWorkspace, minimal_PeakCollection, direction="22", sub_runs=(4, 5, 6))

        def failing(field, values):
            raise RuntimeError("simulated mid-append failure")

        monkeypatch.setattr(_sample, "tail_append", failing)

        # Act / Assert
        with NXstress(written, "a") as nx:
            with pytest.raises(RuntimeError, match=r"simulated mid-append failure"):
                nx.write([ws], [peaks])
            # The entry is damaged now; a further append on this instance is refused.
            monkeypatch.undo()
            with pytest.raises(RuntimeError, match=r"no longer usable"):
                nx.write([ws], [peaks])


class TestPreconditionsAddedAfterReview:
    """Refusals added after the 04c review, each previously a failure part-way through
    the mutation phase or a silent loss. Like every precondition, each leaves the
    entry byte-for-byte unchanged."""

    def test_raw_counts_on_some_incoming_workspaces_raises(
        self, discriminated, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path
    ):
        """Used to fail inside `_InputData.init_group`, after the entry times had grown;
        the damaged entry then read back without error."""
        # Arrange
        path = tmp_path / "counts.nxs"
        first, first_peaks = workspace(
            minimal_HidraWorkspace, minimal_PeakCollection, direction="11", sub_runs=(1, 2, 3), with_raw_counts=True
        )
        with NXstress(path, "w") as nx:
            nx.write([first], [first_peaks])
        loaded, loaded_peaks = workspace(
            minimal_HidraWorkspace, minimal_PeakCollection, direction="22", sub_runs=(4, 5, 6), with_raw_counts=True
        )
        empty, empty_peaks = workspace(minimal_HidraWorkspace, minimal_PeakCollection, direction="33", sub_runs=(7, 8))
        before = digest(path)

        # Act / Assert
        with NXstress(path, "a") as nx:
            with pytest.raises(RuntimeError, match=r"some incoming workspaces and not others"):
                nx.write([loaded, empty], [loaded_peaks, empty_peaks])
        assert digest(path) == before

    @pytest.mark.parametrize("log, field", [("SampleName", "name"), ("chemical formula", "chemical_formula")])
    def test_different_sample_description_raises(
        self, written, minimal_HidraWorkspace, minimal_PeakCollection, log, field
    ):
        """The entry's sample name and formula are never rewritten by an append, so the
        appended rows would have been recorded under the first write's sample."""
        # Arrange
        ws, peaks = workspace(minimal_HidraWorkspace, minimal_PeakCollection, direction="22", sub_runs=(4, 5, 6))
        ws.set_sample_log(log, np.array([4, 5, 6]), np.array(["aluminium"] * 3))
        before = digest(written)

        # Act / Assert
        with NXstress(written, "a") as nx:
            with pytest.raises(RuntimeError, match=rf"sample {field} 'aluminium' differs"):
                nx.write([ws], [peaks])
        assert digest(written) == before

    def test_mask_with_different_contents_raises(self, written, minimal_HidraWorkspace, minimal_PeakCollection):
        """Masks are compared by contents, not only by name: the entry's masks are
        written once, so an appended workspace's different array was silently
        replaced by the entry's on read."""
        # Arrange
        ws, peaks = workspace(minimal_HidraWorkspace, minimal_PeakCollection, direction="22", sub_runs=(4, 5, 6))
        ws.set_detector_mask(np.zeros(16, dtype=np.int64), True)
        before = digest(written)

        # Act / Assert
        with NXstress(written, "a") as nx:
            with pytest.raises(RuntimeError, match=r"masks \['_DEFAULT_'\] have different contents"):
                nx.write([ws], [peaks])
        assert digest(written) == before

    def test_sample_log_of_another_type_raises(self, written, minimal_HidraWorkspace, minimal_PeakCollection):
        """Used to raise `TypeError` inside h5py during the mutation phase."""
        # Arrange
        ws, peaks = workspace(minimal_HidraWorkspace, minimal_PeakCollection, direction="22", sub_runs=(4, 5, 6))
        ws.set_sample_log("mrot", np.array([4, 5, 6]), np.array(["a", "b", "c"]))
        before = digest(written)

        # Act / Assert
        with NXstress(written, "a") as nx:
            with pytest.raises(RuntimeError, match=r"cannot be stored in the target entry's columns"):
                nx.write([ws], [peaks])
        assert digest(written) == before

    def test_discriminator_of_a_narrower_type_raises(
        self, config_override, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path
    ):
        """A float appended to an integer discriminator column would be truncated silently."""
        # Arrange -- a sample-log discriminator, so its type is the log's own
        configure(config_override, "nxstress:\n  discriminator_fields: ['group']\n")
        path = tmp_path / "group.nxs"
        first = minimal_HidraWorkspace(with_instrument=True, with_masks=True, sub_runs=np.array([1, 2, 3]))
        first.set_sample_log("group", np.array([1, 2, 3]), np.array([1, 1, 1]))
        with NXstress(path, "w") as nx:
            nx.write([first], [[minimal_PeakCollection(N_subrun=3, sub_runs=np.array([1, 2, 3]))]])
        ws = minimal_HidraWorkspace(with_instrument=True, with_masks=True, sub_runs=np.array([4, 5, 6]))
        ws.set_sample_log("group", np.array([4, 5, 6]), np.array([2.5, 2.5, 2.5]))
        before = digest(path)

        # Act / Assert
        with NXstress(path, "a") as nx:
            with pytest.raises(RuntimeError, match=r"discriminator field 'group' is stored as int"):
                nx.write([ws], [[minimal_PeakCollection(N_subrun=3, sub_runs=np.array([4, 5, 6]))]])
        assert digest(path) == before

    def test_a_group_refusal_invalidates_the_instance(self, written, minimal_HidraWorkspace, minimal_PeakCollection):
        """A per-group disagreement is a violated invariant like any other, so a
        later write on the same instance is refused, as `write` documents."""
        # Arrange
        from pyrs.core.instrument_geometry import DENEXDetectorGeometry

        ws, peaks = workspace(minimal_HidraWorkspace, minimal_PeakCollection, direction="22", sub_runs=(4, 5, 6))
        ws.set_instrument_geometry(
            DENEXDetectorGeometry(
                num_rows=4, num_columns=4, pixel_size_x=0.002, pixel_size_y=0.002, arm_length=3.0, calibrated=True
            )
        )
        good, good_peaks = workspace(minimal_HidraWorkspace, minimal_PeakCollection, direction="33", sub_runs=(7, 8))

        # Act / Assert
        with NXstress(written, "a") as nx:
            with pytest.raises(RuntimeError, match=r"instrument geometry"):
                nx.write([ws], [peaks])
            with pytest.raises(RuntimeError, match=r"no longer usable"):
                nx.write([good], [good_peaks])


class TestReadBackScanPointOrder:
    """An append may not leave a workspace's scan points out of order on read.

    Rows sharing a discriminator value are read back as one workspace, in
    on-disk order, and `SubRuns` refuses anything not strictly increasing. See
    DD-2 in `plans/PR_review/deferred-defects.md`. Unsupported rather than
    violated, like Case B: the entry is unchanged and the instance stays usable.
    """

    @pytest.fixture
    def higher(self, discriminated, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path):
        """An entry holding scan points [4, 5, 6] for direction 11."""
        path = tmp_path / "higher.nxs"
        ws, peaks = workspace(minimal_HidraWorkspace, minimal_PeakCollection, direction="11", sub_runs=(4, 5, 6))
        with NXstress(path, "w") as nx:
            nx.write([ws], [peaks])
        return path

    def test_lower_points_under_a_shared_discriminator_raise(
        self, higher, minimal_HidraWorkspace, minimal_PeakCollection
    ):
        """Used to append without error, leaving an entry that `read` refused."""
        # Arrange -- a new peak tag, so Case A, but the same direction
        ws, peaks = workspace(
            minimal_HidraWorkspace, minimal_PeakCollection, direction="11", sub_runs=(1, 2, 3), peak_tag="Ni200"
        )
        before = digest(higher)

        # Act / Assert
        with NXstress(higher, "a") as nx:
            with pytest.raises(NotImplementedError, match=r"\[4, 5, 6, 1, 2, 3\]"):
                nx.write([ws], [peaks])
        assert digest(higher) == before

    def test_the_refusal_leaves_the_instance_usable(self, higher, minimal_HidraWorkspace, minimal_PeakCollection):
        """Lower points under a new discriminator value are read back on their own."""
        # Arrange
        refused, refused_peaks = workspace(
            minimal_HidraWorkspace, minimal_PeakCollection, direction="11", sub_runs=(1, 2, 3), peak_tag="Ni200"
        )
        accepted, accepted_peaks = workspace(
            minimal_HidraWorkspace, minimal_PeakCollection, direction="22", sub_runs=(1, 2, 3)
        )

        # Act
        with NXstress(higher, "a") as nx:
            with pytest.raises(NotImplementedError):
                nx.write([refused], [refused_peaks])
            nx.write([accepted], [accepted_peaks])
        with NXstress(higher, "r") as nx:
            read_wss, _ = nx.read()

        # Assert
        assert [ws.get_sub_runs().raw_copy().tolist() for ws in read_wss] == [[4, 5, 6], [1, 2, 3]]


class TestWriterEmitsResizableDatasets:
    """The property the whole design rests on, and the one nothing was checking.

    A dataset written without `maxshape` is contiguous: no mechanism can extend
    it, so a field that loses `growable` silently becomes un-appendable and only
    a later append would notice. These iterate the entry rather than restating a
    list of names, so a field added afterwards is covered automatically -- the
    idiom of `test_definitions.py`.

    The production counterpart is `NXstress._validateAppendableShapes`, which
    runs the same sweep in the pre-flight pass. These reimplement the walk
    locally rather than calling it: a detector that shares its implementation
    with the code under test cannot fail independently of it.
    """

    # `n_scan` must not collide with any entry-wide array length, or an
    # entry-wide array would be swept in and wrongly demanded to be resizable.
    # The writer emits fixed-size arrays of length 1 (`masks/names`), 2
    # (`data_size`) and `n_pixels` == 16 (`masks/detector/*`). 7 collides with
    # none of them; 3 is also safe today but only by two of those being small.
    N_SCAN = 7

    @staticmethod
    def per_scan_point_datasets(path: Path, n_scan: int) -> dict:
        """Every dataset in the entry whose first axis is the scan-point axis.

        Scalars (`shape == ()`) are entry-wide by construction and never match,
        which is what excludes `definition`, `title`, `center_type`, the
        instrument names and the rest without a maintained denylist.
        """
        found = {}

        def walk(group, prefix):
            for name, item in group.items():
                sub = f"{prefix}/{name}"
                if isinstance(item, h5py.Group):
                    walk(item, sub)
                elif item.shape and item.shape[0] == n_scan:
                    found[sub] = item.maxshape

        with h5py.File(path, "r") as f:
            walk(f["entry"], "entry")
        return found

    @pytest.fixture
    def swept(self, discriminated, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path):
        """An entry with raw counts, so `input_data` is in the swept set too."""
        path = tmp_path / "swept.nxs"
        ws, peaks = workspace(
            minimal_HidraWorkspace,
            minimal_PeakCollection,
            direction="11",
            sub_runs=tuple(range(1, self.N_SCAN + 1)),
            with_raw_counts=True,
        )
        with NXstress(path, "w") as nx:
            nx.write([ws], [peaks])
        return path

    def test_every_per_scan_point_dataset_is_extendable(self, swept):
        # Act
        datasets = self.per_scan_point_datasets(swept, self.N_SCAN)

        # Assert -- the sweep found something, and every one of them can grow
        assert datasets, "the sweep matched no datasets; its scan-point heuristic has drifted"
        fixed = {name: maxshape for name, maxshape in datasets.items() if maxshape[0] is not None}
        assert not fixed, (
            f"{len(fixed)} per-scan-point dataset(s) were written at a fixed size and can never be "
            f"appended to -- pass `**growable(rank)` at their `NXfield(...)`: {sorted(fixed)}"
        )

    def test_the_sweep_reaches_input_data(self, swept):
        """`detector_counts` is the bulk of a real entry and must be inside the guarantee.

        It was outside it: the shared fixture carries no raw counts, so both
        `input_data` datasets are zero-length and drop out of a length-matched
        sweep.
        """
        # Act
        datasets = self.per_scan_point_datasets(swept, self.N_SCAN)

        # Assert
        assert "entry/input_data/detector_counts" in datasets
        assert "entry/input_data/scan_point" in datasets

    def test_the_sweep_notices_a_field_that_loses_growable(self, swept):
        """The guarantee above is only worth having if the sweep can actually fail."""
        # Arrange -- add a per-scan-point field the way a careless edit would
        with nxopen(swept, "rw") as root:
            root["entry"]["SAMPLE_DESCRIPTION"]["careless"] = NXfield(np.zeros(self.N_SCAN))

        # Act
        datasets = self.per_scan_point_datasets(swept, self.N_SCAN)

        # Assert
        assert datasets["entry/SAMPLE_DESCRIPTION/careless"][0] is not None


class TestAppendableDatasetsAreDeclaredByTheirOwner:
    """Which datasets an append grows is declared by the module that writes them.

    It used to be inferred by sweeping the whole entry for `shape[0] == n_scan`,
    which was wrong in both directions.

    It **over**-matched, and over-matching is not the harmless direction the old
    docstring claimed: the result feeds a resizability *requirement*, so an
    entry-wide array that happens to be `n_scan` long rejects a legal append.
    `masks/names` is the live case -- entry-wide, never grown, and required by
    `_Instrument.validateAppend` to stay constant. Note that
    `TestWriterEmitsResizableDatasets` picked `N_SCAN = 7` with a comment
    reasoning about exactly this collision: the hazard was understood and
    designed around in a test instead of being fixed in the code.

    It also **under**-matched: the peak-index family is one row per peak record,
    so unless that count happened to equal `n_scan`, none of `peaks/*` or the
    parameter groups was pre-flighted at all -- which is the mid-mutation refusal
    04c's Follow-up 3 F3.1 exists to prevent. See 04c's Follow-up 5.
    """

    def test_as_many_detector_masks_as_scan_points_still_appends(
        self, discriminated, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path
    ):
        """Three masks, three scan points -- the collision that refused a legal append."""
        # Arrange
        masks = ("mask_a", "mask_b")  # plus `_DEFAULT_` -> `masks/names` has length 3
        path = tmp_path / "collide.nxs"
        first, first_peaks = workspace(
            minimal_HidraWorkspace, minimal_PeakCollection, direction="11", sub_runs=(1, 2, 3), mask_names=masks
        )
        with NXstress(path, "w") as nx:
            nx.write([first], [first_peaks])
        with h5py.File(path, "r") as f:
            assert f["entry/instrument/masks/names"].shape == (3,)
            assert len(f["entry/SAMPLE_DESCRIPTION/scan_point"]) == 3

        second, second_peaks = workspace(
            minimal_HidraWorkspace, minimal_PeakCollection, direction="22", sub_runs=(4, 5, 6), mask_names=masks
        )

        # Act
        with NXstress(path, "a") as nx:
            nx.write([second], [second_peaks])
        with NXstress(path, "r") as nx:
            workspaces, _ = nx.read()

        # Assert
        assert [w.get_sub_runs().raw_copy().tolist() for w in workspaces] == [[1, 2, 3], [4, 5, 6]]

    def test_the_peak_index_family_is_pre_flighted_when_it_outnumbers_scan_points(
        self, discriminated, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path
    ):
        """Two phases over three scan points: six peak rows, three scan points.

        The peak index is one row per (compound key, scan point), so it equals
        `n_scan` only while each workspace fits exactly one peak. Fit a second
        phase -- the ordinary multi-peak case -- and the two counts diverge, at
        which point a length-matched sweep stopped covering `peaks/*` and the
        parameter groups entirely.
        """
        # Arrange
        points = (1, 2, 3)
        path = tmp_path / "peakindex.nxs"
        ws, peaks = workspace(minimal_HidraWorkspace, minimal_PeakCollection, direction="11", sub_runs=points)
        peaks.append(minimal_PeakCollection(N_subrun=len(points), peak_tag="Si111", sub_runs=np.array(points)))
        with NXstress(path, "w") as nx:
            nx.write([ws], [peaks])

        # Act
        with NXstress(path, "r") as nx:
            entry = nx._root["entry"]
            n_scan = len(entry["SAMPLE_DESCRIPTION"]["scan_point"].nxdata)
            declared = set(NXstress._appendableDatasets(entry))

        # Assert
        with h5py.File(path, "r") as f:
            assert f["entry/peaks/h"].shape == (2 * n_scan,)
        assert "entry/peaks/h" in declared
        assert "entry/FIT/peak_parameters/center" in declared
        assert "entry/FIT/background_parameters/A0" in declared

    def test_every_growable_dataset_has_exactly_one_owner(
        self, discriminated, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path
    ):
        """The guarantee the length sweep gave for free, now stated directly.

        A field written `growable` is one an append is expected to extend, so a
        new one that no module declares would silently drop out of the pre-flight
        -- the failure the sweep was there to prevent. This keeps that covered
        without the sweep's false positives.
        """
        # Arrange
        path = tmp_path / "owned.nxs"
        ws, peaks = workspace(
            minimal_HidraWorkspace,
            minimal_PeakCollection,
            direction="11",
            sub_runs=(1, 2, 3),
            with_raw_counts=True,
        )
        with NXstress(path, "w") as nx:
            nx.write([ws], [peaks])

        growable_on_disk = set()
        with h5py.File(path, "r") as f:

            def walk(group, prefix):
                for name, item in group.items():
                    sub = f"{prefix}/{name}"
                    if isinstance(item, h5py.Group):
                        walk(item, sub)
                    elif item.maxshape and item.maxshape[0] is None:
                        growable_on_disk.add(sub)

            walk(f["entry"], "entry")

        # Act
        with NXstress(path, "r") as nx:
            declared = set(NXstress._appendableDatasets(nx._root["entry"]))

        # Assert
        assert growable_on_disk, "no resizable datasets were written at all"
        assert growable_on_disk - declared == set(), (
            "dataset(s) written `growable` that no module declares, so an append would never "
            f"pre-flight them: {sorted(growable_on_disk - declared)}"
        )
        assert declared - growable_on_disk == set(), (
            f"declared dataset(s) that are not resizable on disk: {sorted(declared - growable_on_disk)}"
        )


class TestTailAppendHelper:
    """`tail_append` is where the growth rule and the legacy-file refusal both live."""

    def test_grows_by_exactly_n_and_leaves_existing_rows_unchanged(self, tmp_path):
        # Arrange
        path = tmp_path / "helper.nxs"
        with nxopen(path, "w") as root:
            root["entry"] = NXentry()
            root["entry"]["field"] = NXfield(np.arange(3, dtype=np.float64), maxshape=(None,), chunks=(100,))

        # Act
        with nxopen(path, "rw") as root:
            tail_append(root["entry"]["field"], np.array([9.0, 9.0]))

        # Assert
        with h5py.File(path, "r") as f:
            assert f["entry/field"][()].tolist() == [0.0, 1.0, 2.0, 9.0, 9.0]

    def test_refuses_a_dataset_written_at_a_fixed_size(self, tmp_path):
        """A file written by a PyRS predating `growable` cannot be appended to."""
        # Arrange
        path = tmp_path / "legacy.nxs"
        with nxopen(path, "w") as root:
            root["entry"] = NXentry()
            root["entry"]["field"] = NXfield(np.arange(3, dtype=np.float64))
        before = digest(path)

        # Act / Assert
        with nxopen(path, "rw") as root:
            assert not appendable(root["entry"]["field"])
            with pytest.raises(RuntimeError, match=r"written at a fixed size and cannot be extended"):
                tail_append(root["entry"]["field"], np.array([9.0]))
        assert digest(path) == before

    def test_refuses_rows_whose_trailing_axes_disagree(self, tmp_path):
        """A diffractogram appended at a different two-theta width would misalign silently."""
        # Arrange
        path = tmp_path / "width.nxs"
        with nxopen(path, "w") as root:
            root["entry"] = NXentry()
            root["entry"]["field"] = NXfield(
                np.zeros((2, 20), dtype=np.float64), maxshape=(None, None), chunks=(1, 100)
            )

        # Act / Assert
        with nxopen(path, "rw") as root:
            with pytest.raises(RuntimeError, match=r"trailing shape"):
                tail_append(root["entry"]["field"], np.zeros((1, 25)))
