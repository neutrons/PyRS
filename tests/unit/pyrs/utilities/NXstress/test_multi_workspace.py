"""Multi-workspace NXstress I/O: N workspaces in, N workspaces out.

One `NXentry` may hold more than one `HidraWorkspace`. Their rows are
concatenated in workspace order throughout, and the boundary between them is
recovered on read from discriminator columns on the peak index. These tests
cover the round trip, the policies that gate it, and the two shapes that are
easy to get wrong -- interleaved scan-point *values*, and configuration drift
between write and read.
"""

from collections.abc import Callable

import h5py
import numpy as np
import yaml
import pytest

from pyrs.core.workspaces import HidraWorkspace
from pyrs.peaks.peak_collection import PeakCollection
from pyrs.utilities.NXstress._definitions import GROUP_NAME
from pyrs.utilities.NXstress.NXstress import NXstress


def configure(config_override, yaml_text: str) -> None:
    """Apply an `nxstress` config override that production modules can actually see.

    Goes through `config_override` rather than `default_config`: the latter swaps
    the `Config` singleton, leaving every module that bound it the documented way
    reading the previous instance. See that fixture's docstring.
    """
    config_override(yaml.safe_load(yaml_text))


def with_direction(ws: HidraWorkspace, direction: str, sub_runs: np.ndarray) -> HidraWorkspace:
    ws.set_sample_log("direction", sub_runs, np.array([direction] * len(sub_runs)))
    return ws


def two_workspaces(
    minimal_HidraWorkspace: Callable[..., HidraWorkspace],
    minimal_PeakCollection: Callable[..., PeakCollection],
    *,
    sub_runs_a=(1, 2, 3),
    sub_runs_b=(4, 5, 6),
    directions=("11", "22"),
    peak_tags=("Fe110", "Fe110"),
):
    """Two workspaces with disjoint scan points and distinct discriminator values."""
    workspaces, peakss = [], []
    for direction, sub_runs, peak_tag in zip(directions, (np.array(sub_runs_a), np.array(sub_runs_b)), peak_tags):
        ws = minimal_HidraWorkspace(with_instrument=True, with_masks=True, sub_runs=sub_runs)
        with_direction(ws, direction, sub_runs)
        workspaces.append(ws)
        peakss.append([minimal_PeakCollection(N_subrun=len(sub_runs), peak_tag=peak_tag, sub_runs=sub_runs)])
    return workspaces, peakss


class TestDiscriminatorRoundtrip:
    def test_two_workspaces_recovered_with_their_own_scan_points(
        self, config_override, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path
    ):
        configure(config_override, "nxstress:\n  discriminator_fields: ['direction']\n")
        workspaces, peakss = two_workspaces(minimal_HidraWorkspace, minimal_PeakCollection)

        path = tmp_path / "two.nxs"
        with NXstress(path, "w") as nx:
            nx.write(workspaces, peakss)
        with NXstress(path, "r") as nx:
            read_wss, read_peakss = nx.read()

        assert len(read_wss) == 2
        by_direction = {ws.direction: ws for ws in read_wss}
        assert sorted(by_direction) == ["11", "22"]
        np.testing.assert_array_equal(by_direction["11"].get_sub_runs().raw_copy(), [1, 2, 3])
        np.testing.assert_array_equal(by_direction["22"].get_sub_runs().raw_copy(), [4, 5, 6])
        assert [len(p) for p in read_peakss] == [1, 1]

    def test_discriminator_column_is_written_to_the_peaks_group(
        self, config_override, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path
    ):
        configure(config_override, "nxstress:\n  discriminator_fields: ['direction']\n")
        workspaces, peakss = two_workspaces(minimal_HidraWorkspace, minimal_PeakCollection)

        path = tmp_path / "column.nxs"
        with NXstress(path, "w") as nx:
            nx.write(workspaces, peakss)

        with NXstress(path, "r") as nx:
            peaks = nx._root["entry"][GROUP_NAME.PEAKS]
            assert "direction" in peaks
            assert peaks["direction"].attrs["local_name"] == "direction"
            values = [v.decode() if isinstance(v, bytes) else str(v) for v in peaks["direction"].nxdata]

        # Discriminators are the most slowly varying coordinate, so each
        # workspace's rows form one contiguous super-block.
        assert values == ["11"] * 3 + ["22"] * 3

    def test_single_workspace_also_carries_the_column(
        self, config_override, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path
    ):
        """`N == 1` is not a special case: the column is written whenever a field is configured."""
        configure(config_override, "nxstress:\n  discriminator_fields: ['direction']\n")
        sub_runs = np.array([1, 2, 3])
        ws = with_direction(
            minimal_HidraWorkspace(with_instrument=True, with_masks=True, sub_runs=sub_runs), "33", sub_runs
        )
        peaks = [minimal_PeakCollection(N_subrun=3, peak_tag="Fe110", sub_runs=sub_runs)]

        path = tmp_path / "one.nxs"
        with NXstress(path, "w") as nx:
            nx.write([ws], [peaks])
        with NXstress(path, "r") as nx:
            read_wss, _ = nx.read()

        assert len(read_wss) == 1
        assert read_wss[0].direction == "33"


class TestBackCompat:
    def test_length_one_list_round_trips_without_any_discriminator(
        self, config_override, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path
    ):
        """Regression guard for specs 02/03, which write a single workspace."""
        ws = minimal_HidraWorkspace(with_instrument=True, with_masks=True)
        n_subrun = len(ws.get_sub_runs())
        peaks = [minimal_PeakCollection(N_subrun=n_subrun, peak_tag="Fe110")]

        path = tmp_path / "back-compat.nxs"
        with NXstress(path, "w") as nx:
            nx.write([ws], [peaks])
        with NXstress(path, "r") as nx:
            read_wss, read_peakss = nx.read()

        assert len(read_wss) == 1 and len(read_peakss) == 1
        np.testing.assert_array_equal(read_wss[0].get_sub_runs().raw_copy(), ws.get_sub_runs().raw_copy())
        assert read_peakss[0][0].peak_tag == "Fe110"

    def test_file_without_columns_reads_as_one_workspace_under_configured_fields(
        self, config_override, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path
    ):
        """A pre-04b file stays readable once a deployment configures a field."""
        ws = minimal_HidraWorkspace(with_instrument=True, with_masks=True)
        peaks = [minimal_PeakCollection(N_subrun=len(ws.get_sub_runs()), peak_tag="Fe110")]

        path = tmp_path / "legacy.nxs"
        with NXstress(path, "w") as nx:  # written with no field configured
            nx.write([ws], [peaks])

        configure(config_override, "nxstress:\n  discriminator_fields: ['direction']\n")
        with NXstress(path, "r") as nx:
            read_wss, _ = nx.read()

        assert len(read_wss) == 1

    def test_file_with_a_different_field_set_raises(
        self, config_override, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path
    ):
        """Configuration is the authority, and a disagreement is loud, not silent."""
        configure(config_override, "nxstress:\n  discriminator_fields: ['direction']\n")
        workspaces, peakss = two_workspaces(minimal_HidraWorkspace, minimal_PeakCollection)
        path = tmp_path / "drifted.nxs"
        with NXstress(path, "w") as nx:
            nx.write(workspaces, peakss)

        configure(config_override, "nxstress:\n  discriminator_fields: ['run_number']\n")
        with NXstress(path, "r") as nx:
            with pytest.raises(RuntimeError, match=r"written with discriminator fields \['direction'\]"):
                nx.read()

    def test_an_entry_with_no_peak_collections_carries_no_discriminator_column(
        self, config_override, minimal_HidraWorkspace, tmp_path
    ):
        """`write([ws], [[]])` must not emit a column it has no values for.

        This is `CombineRunsModel`'s export shape. The column used to be written
        anyway, and typed from no values at all -- `np.asarray([]).dtype` is
        `float64` -- so a string discriminator went to disk as a numeric column.
        Emitting nothing is both correct and what makes the entry read back as a
        single workspace.
        """
        # Arrange
        configure(config_override, "nxstress:\n  discriminator_fields: ['direction']\n")
        ws = minimal_HidraWorkspace(with_instrument=True, with_masks=True)
        path = tmp_path / "no_peaks.nxs"

        # Act
        with NXstress(path, "w") as nx:
            nx.write([ws], [[]])

        # Assert
        with h5py.File(path, "r") as f:
            assert "direction" not in f["entry/peaks"]
        with NXstress(path, "r") as nx:
            workspaces, peakss = nx.read()
        assert len(workspaces) == 1
        assert peakss == [[]]

    def test_a_mask_one_input_never_reduced_does_not_survive_the_round_trip(
        self, config_override, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path
    ):
        """A NaN-filled mask is alignment padding, not a mask the workspace has.

        One DIFFRACTOGRAM group spans the whole entry, so when one input reduced a
        mask and another did not, the second's rows are NaN-filled to keep the
        scan-point axis aligned. Handing those rows back would add an entry to
        `reduction_masks` -- a property `texture_fitting_crtl` and
        `mantid_peakfit_calibration` both *count*.
        """

        # Arrange -- both define `mask_a`; only the first reduces it.
        def ws_for(direction: str, points: tuple, reduced: tuple):
            sub_runs = np.array(points)
            ws = minimal_HidraWorkspace(
                with_instrument=True, with_masks=True, mask_names=("mask_a",), sub_runs=sub_runs
            )
            ws.set_sample_log("direction", sub_runs, np.array([direction] * len(sub_runs)))
            two_theta = np.tile(np.linspace(60.0, 120.0, 20), (len(sub_runs), 1))
            ones = np.ones((len(sub_runs), 20))
            ws.set_reduced_diffraction_data_set(two_theta, {m: ones for m in reduced}, {m: ones for m in reduced})
            return ws, [minimal_PeakCollection(N_subrun=len(sub_runs), sub_runs=sub_runs)]

        configure(config_override, "nxstress:\n  discriminator_fields: ['direction']\n")
        first, first_peaks = ws_for("11", (1, 2, 3), (None, "mask_a"))
        second, second_peaks = ws_for("22", (4, 5, 6), (None,))
        assert first.reduction_masks == [None, "mask_a"]
        assert second.reduction_masks == [None]
        path = tmp_path / "uneven_masks.nxs"

        # Act
        with NXstress(path, "w") as nx:
            nx.write([first, second], [first_peaks, second_peaks])
        with NXstress(path, "r") as nx:
            workspaces, _ = nx.read()

        # Assert -- each workspace gets back exactly the masks it reduced
        assert workspaces[0].reduction_masks == [None, "mask_a"]
        assert workspaces[1].reduction_masks == [None]
        # ...and the group really is on disk, carrying the first input's real data
        with h5py.File(path, "r") as f:
            masked = f["entry/FIT/DIFFRACTOGRAM_mask_a/diffractogram"][()]
        assert masked.shape == (6, 20)
        assert not np.isnan(masked[:3]).any()
        assert np.isnan(masked[3:]).all()


class TestEmptyConfigPolicy:
    def test_multiple_workspaces_without_a_discriminator_raises(
        self, config_override, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path
    ):
        workspaces, peakss = two_workspaces(minimal_HidraWorkspace, minimal_PeakCollection)

        path = tmp_path / "no-field.nxs"
        with NXstress(path, "w") as nx:
            with pytest.raises(ValueError, match="names no field"):
                nx.write(workspaces, peakss)

    def test_the_refusal_points_to_merging_in_pyrs(
        self, config_override, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path
    ):
        """NXstress does not merge workspaces; `nxstress.merge_workspaces` was removed (Decisions Log row 45)."""
        workspaces, peakss = two_workspaces(minimal_HidraWorkspace, minimal_PeakCollection)

        with NXstress(tmp_path / "no-field.nxs", "w") as nx:
            with pytest.raises(ValueError, match="merge the workspaces into one in PyRS"):
                nx.write(workspaces, peakss)


class TestAWorkspaceMergedInPyRS:
    """One workspace whose `direction` log varies, as CombineRuns can produce.

    It is one workspace, so it is written as one: with no discriminator field it
    round-trips with its log intact. Under a `direction` discriminator it has no
    single value, and is refused with a `ValueError` saying so -- it used to fail
    with `get_sample_log_value`'s bare `AssertionError`.
    """

    @staticmethod
    def merged(minimal_HidraWorkspace) -> HidraWorkspace:
        sub_runs = np.arange(1, 7)
        ws = minimal_HidraWorkspace(with_instrument=True, with_masks=True, sub_runs=sub_runs)
        ws.set_sample_log("direction", sub_runs, np.array(["11"] * 3 + ["22"] * 3))
        return ws

    def test_it_round_trips_with_no_discriminator(
        self, config_override, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path
    ):
        # Arrange
        ws = self.merged(minimal_HidraWorkspace)
        path = tmp_path / "merged.nxs"

        # Act
        with NXstress(path, "w") as nx:
            nx.write([ws], [[minimal_PeakCollection(N_subrun=6)]])
        with NXstress(path, "r") as nx:
            (read_ws,), _ = nx.read()

        # Assert
        assert read_ws.get_sub_runs().raw_copy().tolist() == [1, 2, 3, 4, 5, 6]
        assert read_ws.get_sample_log_values("direction").tolist() == ["11"] * 3 + ["22"] * 3

    def test_a_direction_discriminator_is_refused(
        self, config_override, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path
    ):
        # Arrange
        configure(config_override, "nxstress:\n  discriminator_fields: ['direction']\n")
        ws = self.merged(minimal_HidraWorkspace)

        # Act / Assert
        with NXstress(tmp_path / "merged.nxs", "w") as nx:
            with pytest.raises(ValueError, match=r"'direction' has no single value"):
                nx.write([ws], [[minimal_PeakCollection(N_subrun=6)]])
            assert nx.entry_numbers() == ()


class TestWriteTimeInvariants:
    def test_mismatched_list_lengths_raise(
        self, config_override, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path
    ):
        configure(config_override, "nxstress:\n  discriminator_fields: ['direction']\n")
        workspaces, peakss = two_workspaces(minimal_HidraWorkspace, minimal_PeakCollection)

        path = tmp_path / "mismatched.nxs"
        with NXstress(path, "w") as nx:
            with pytest.raises(ValueError, match="one list of `PeakCollection` per workspace"):
                nx.write(workspaces, peakss[:1])

    def test_workspace_with_no_peak_collections_raises(
        self, config_override, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path
    ):
        """Its discriminator value would exist nowhere on disk, so it could not be recovered."""
        configure(config_override, "nxstress:\n  discriminator_fields: ['direction']\n")
        workspaces, peakss = two_workspaces(minimal_HidraWorkspace, minimal_PeakCollection)
        peakss[1] = []

        path = tmp_path / "no-peaks.nxs"
        with NXstress(path, "w") as nx:
            with pytest.raises(ValueError, match="contribute no `PeakCollection`"):
                nx.write(workspaces, peakss)

    def test_single_workspace_with_no_peak_collections_is_allowed(
        self, config_override, minimal_HidraWorkspace, tmp_path
    ):
        """Spec 03's case: `N == 1`, so nothing has to be told apart."""
        configure(config_override, "nxstress:\n  discriminator_fields: ['direction']\n")
        ws = minimal_HidraWorkspace(with_instrument=True, with_masks=True)

        path = tmp_path / "no-peaks-one-ws.nxs"
        with NXstress(path, "w") as nx:
            nx.write([ws], [[]])
        with NXstress(path, "r") as nx:
            read_wss, read_peakss = nx.read()

        assert len(read_wss) == 1
        assert read_peakss == [[]]

    def test_overlapping_scan_points_raise(
        self, config_override, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path
    ):
        """The reader attributes rows by scan-point value, so an overlap is unsplittable."""
        configure(config_override, "nxstress:\n  discriminator_fields: ['direction']\n")
        workspaces, peakss = two_workspaces(
            minimal_HidraWorkspace, minimal_PeakCollection, sub_runs_a=(1, 2, 3), sub_runs_b=(3, 4, 5)
        )

        path = tmp_path / "overlap.nxs"
        with NXstress(path, "w") as nx:
            with pytest.raises(ValueError, match="cover disjoint scan points"):
                nx.write(workspaces, peakss)

    def test_masks_of_the_same_name_with_different_contents_raise(
        self, config_override, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path
    ):
        """Masks are written once, from the first input: a second input's different
        array under the same name was silently replaced by the first's on read."""
        # Arrange
        configure(config_override, "nxstress:\n  discriminator_fields: ['direction']\n")
        workspaces, peakss = two_workspaces(minimal_HidraWorkspace, minimal_PeakCollection)
        workspaces[1].set_detector_mask(np.zeros(16, dtype=np.int64), True)

        # Act / Assert
        with NXstress(tmp_path / "masks.nxs", "w") as nx:
            with pytest.raises(RuntimeError, match=r"same name with different contents"):
                nx.write(workspaces, peakss)


class TestConfigDrift:
    def test_reordering_the_fields_between_write_and_read_is_harmless(
        self, config_override, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path
    ):
        """Values are attributed by name, never by position in the configured list."""
        configure(config_override, "nxstress:\n  discriminator_fields: ['a_field', 'b_field']\n")

        workspaces, peakss = [], []
        for n, (a, b, sub_runs) in enumerate(
            (("alpha", "beta", np.array([1, 2, 3])), ("gamma", "delta", np.array([4, 5, 6])))
        ):
            ws = minimal_HidraWorkspace(with_instrument=True, with_masks=True, sub_runs=sub_runs)
            ws.set_sample_log("a_field", sub_runs, np.array([a] * 3))
            ws.set_sample_log("b_field", sub_runs, np.array([b] * 3))
            workspaces.append(ws)
            peakss.append([minimal_PeakCollection(N_subrun=3, peak_tag="Fe110", sub_runs=sub_runs)])

        path = tmp_path / "reordered.nxs"
        with NXstress(path, "w") as nx:
            nx.write(workspaces, peakss)

        configure(config_override, "nxstress:\n  discriminator_fields: ['b_field', 'a_field']\n")
        with NXstress(path, "r") as nx:
            read_wss, _ = nx.read()

        recovered = {(ws.get_sample_log_value("a_field"), ws.get_sample_log_value("b_field")) for ws in read_wss}
        assert recovered == {("alpha", "beta"), ("gamma", "delta")}


class TestScanPointFamilySplit:
    def test_interleaved_scan_point_values_split_by_value_not_position(
        self, config_override, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path
    ):
        """Workspace A holds [1,3,5] and B holds [2,4,6].

        Their rows stay positionally contiguous -- the write side never
        interleaves positions -- but their *values* interleave, so the
        concatenated axis [1,3,5,2,4,6] is not monotonic. Recovering each
        workspace therefore has to go by value-set membership, and has to do so
        before any `SubRuns` is built: `SubRuns` rejects a non-monotonic array
        outright. See `plans/NXstress-prod/probes/a5_subruns_nonmonotonic.py`.
        """
        configure(config_override, "nxstress:\n  discriminator_fields: ['direction']\n")
        workspaces, peakss = two_workspaces(
            minimal_HidraWorkspace, minimal_PeakCollection, sub_runs_a=(1, 3, 5), sub_runs_b=(2, 4, 6)
        )

        path = tmp_path / "interleaved.nxs"
        with NXstress(path, "w") as nx:
            nx.write(workspaces, peakss)
        with NXstress(path, "r") as nx:
            read_wss, _ = nx.read()

        by_direction = {ws.direction: ws for ws in read_wss}
        np.testing.assert_array_equal(by_direction["11"].get_sub_runs().raw_copy(), [1, 3, 5])
        np.testing.assert_array_equal(by_direction["22"].get_sub_runs().raw_copy(), [2, 4, 6])

        # The reduced data must follow the same split, not just the scan points.
        for direction, ws_original in zip(("11", "22"), workspaces):
            np.testing.assert_allclose(by_direction[direction]._diff_data_set[None], ws_original._diff_data_set[None])


class TestReadBackScanPointOrder:
    """A workspace read back must hold strictly increasing scan points.

    That requirement belongs to `SubRuns`, not to the file format, and DD-2 in
    `plans/PR_review/deferred-defects.md` recommends dropping it. Until then a
    write that could not be read back is refused, rather than re-sorted: the
    on-disk rows are in the inputs' order, and sorting them would break the
    round trip.
    """

    @pytest.mark.parametrize(
        "sub_runs_a, sub_runs_b", [((4, 5, 6), (1, 2, 3)), ((1, 3, 5), (2, 4, 6))], ids=["decreasing", "interleaved"]
    )
    def test_a_shared_discriminator_value_out_of_order_raises(
        self, config_override, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path, sub_runs_a, sub_runs_b
    ):
        """Used to write an entry that `read` then refused: one discriminator value reads back as one workspace."""
        # Arrange
        configure(config_override, "nxstress:\n  discriminator_fields: ['direction']\n")
        workspaces, peakss = two_workspaces(
            minimal_HidraWorkspace,
            minimal_PeakCollection,
            sub_runs_a=sub_runs_a,
            sub_runs_b=sub_runs_b,
            directions=("11", "11"),
            peak_tags=("Fe110", "Ni200"),
        )
        expected = str(list(sub_runs_a + sub_runs_b)).replace("[", r"\[").replace("]", r"\]")

        # Act / Assert
        with NXstress(tmp_path / "shared.nxs", "w") as nx:
            with pytest.raises(NotImplementedError, match=expected):
                nx.write(workspaces, peakss)
            assert nx.entry_numbers() == ()

    def test_distinct_discriminators_in_decreasing_order_round_trip(
        self, config_override, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path
    ):
        """Each workspace is read back from its own rows, which are increasing."""
        # Arrange
        configure(config_override, "nxstress:\n  discriminator_fields: ['direction']\n")
        workspaces, peakss = two_workspaces(
            minimal_HidraWorkspace, minimal_PeakCollection, sub_runs_a=(4, 5, 6), sub_runs_b=(1, 2, 3)
        )
        path = tmp_path / "decreasing.nxs"

        # Act
        with NXstress(path, "w") as nx:
            nx.write(workspaces, peakss)
        with NXstress(path, "r") as nx:
            read_wss, _ = nx.read()

        # Assert
        assert [ws.get_sub_runs().raw_copy().tolist() for ws in read_wss] == [[4, 5, 6], [1, 2, 3]]

    @pytest.mark.xfail(
        reason="DD-2: `SubRuns` requires sorted scan points, so a decreasing workspace cannot be held",
        raises=NotImplementedError,
        strict=True,
    )
    def test_shared_value_in_decreasing_order_round_trips(
        self, config_override, minimal_HidraWorkspace, minimal_PeakCollection, tmp_path
    ):
        """The target once the ordering requirement is dropped: the inputs' order is kept."""
        # Arrange
        configure(config_override, "nxstress:\n  discriminator_fields: ['direction']\n")
        workspaces, peakss = two_workspaces(
            minimal_HidraWorkspace,
            minimal_PeakCollection,
            sub_runs_a=(4, 5, 6),
            sub_runs_b=(1, 2, 3),
            directions=("11", "11"),
            peak_tags=("Fe110", "Ni200"),
        )
        path = tmp_path / "shared.nxs"

        # Act
        with NXstress(path, "w") as nx:
            nx.write(workspaces, peakss)
        with NXstress(path, "r") as nx:
            (read_ws,), _ = nx.read()

        # Assert
        assert read_ws.get_sub_runs().raw_copy().tolist() == [4, 5, 6, 1, 2, 3]
