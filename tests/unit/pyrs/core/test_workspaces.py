import numpy as np
import pytest

from pyrs.core.workspaces import HidraWorkspace
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode  # type: ignore
from tests.util.workspace_helpers import make_minimal_HidraWorkspace


class TestHidraWorkspace:
    def test_set_sample_log(self):
        workspace = HidraWorkspace()
        subruns, vx = np.array([1, 2, 3], dtype=int), np.array([0.0, 0.1, 0.2])
        workspace.set_sample_log("vx", subruns, vx)
        assert workspace.get_sample_log_units("vx") == ""
        workspace.set_sample_log("vx", subruns, vx, "mm")
        assert workspace.get_sample_log_units("vx") == "mm"

    @pytest.mark.integration
    def test_append_projectfiels(self):
        # import data file: detector ID and file name
        test_data = [
            "tests/data/HB2B_1327.h5",
            "tests/data/HB2B_1328.h5",
            "tests/data/HB2B_1331.h5",
            "tests/data/HB2B_1332.h5",
        ]

        test_ws = HidraWorkspace("test_powder_pattern")
        test_project = HidraProjectFile(test_data[0])
        test_ws.load_hidra_project(test_project, load_raw_counts=False, load_reduced_diffraction=True)
        test_project.close()

        for test_file in test_data[1:]:
            test_project = HidraProjectFile(test_file)
            test_ws.append_hidra_project(test_project)
            test_project.close()

        assert test_ws.get_sub_runs().size == 362

    def test_integer_intensities_keep_two_theta(self):
        """2theta must not take the intensities' dtype: integer counts used to truncate it"""
        # Arrange
        workspace = HidraWorkspace()
        workspace.set_sub_runs([1, 2])
        two_theta = np.array([75.79, 75.82, 75.85])
        counts = np.array([3, 10, 4])

        # Act
        workspace.set_reduced_diffraction_data(1, None, two_theta, counts)

        # Assert
        np.testing.assert_array_equal(workspace.get_reduced_diffraction_data_2theta(1), two_theta)
        _, intensity, _ = workspace.get_reduced_diffraction_data(1)
        np.testing.assert_array_equal(intensity, counts)

    def test_new_mask_stores_float_intensities(self):
        """A mask added after the first one must store float intensities whatever its first input"""
        # Arrange
        workspace = HidraWorkspace()
        workspace.set_sub_runs([1, 2])
        two_theta = np.array([75.79, 75.82, 75.85])
        workspace.set_reduced_diffraction_data(1, None, two_theta, np.array([1.5, 2.5, 3.5]))
        variances = np.ones(3)

        # Act
        workspace.set_reduced_diffraction_data(1, "wedge", two_theta, np.array([3, 10, 4]), variances)
        workspace.set_reduced_diffraction_data(2, "wedge", two_theta, np.array([0.5, 1.25, 2.75]), variances)

        # Assert
        _, intensity, _ = workspace.get_reduced_diffraction_data(2, mask_id="wedge")
        np.testing.assert_array_equal(intensity, [0.5, 1.25, 2.75])


def test_mask_source_defaults_to_not_recorded():
    """#1068: a workspace not made by NeXus conversion does not know where its mask came from"""
    # Act / Assert
    assert HidraWorkspace().get_detector_mask_source() == {"source": "not recorded"}


def test_mask_source_round_trips_as_a_copy():
    # Arrange
    workspace = HidraWorkspace()
    source = {"source": "file", "file": "/masks/HB2B_Mask.xml", "sha256": "ab" * 32}

    # Act
    workspace.set_detector_mask_source(source)
    source["file"] = "changed"

    # Assert
    assert workspace.get_detector_mask_source()["file"] == "/masks/HB2B_Mask.xml"


@pytest.mark.parametrize("source", [None, "HB2B_Mask.xml", {"file": "HB2B_Mask.xml"}])
def test_mask_source_without_source_key_raises(source):
    # Act / Assert
    with pytest.raises(ValueError, match="'source' key"):
        HidraWorkspace().set_detector_mask_source(source)


@pytest.mark.parametrize("stored", ["[1, 2]", '{"mask": "HB2B_Mask.xml"}', "not json"], ids=["list", "string", "text"])
def test_unreadable_mask_source_still_loads(tmp_path, stored):
    """A bad record leaves the mask source "not recorded"; it never stops a project file loading (review)"""
    # Arrange
    import h5py

    from pyrs.projectfile import HidraProjectFileMode  # type: ignore

    path = tmp_path / "bad.h5"
    HidraProjectFile(str(path), HidraProjectFileMode.OVERWRITE).save(verbose=False)
    with h5py.File(path, "a") as h5:
        h5.require_group("instrument").create_group("reduction").create_dataset(
            "reduction_json", data=stored, dtype=h5py.string_dtype(encoding="utf-8")
        )
    workspace = HidraWorkspace()
    project = HidraProjectFile(str(path), HidraProjectFileMode.READONLY)

    # Act
    workspace._load_masks(project)
    project.close()

    # Assert
    assert workspace.get_detector_mask_source() == {"source": "not recorded"}


class TestSubRunNumbersAreNotPositions:
    """Sub-run numbers are labels, not array positions.

    Each xfail is a site recorded under DD-2 in
    `plans/PR_review/deferred-defects.md` that treats a sub-run number as a
    position, or pairs values with sub-runs by position alone. They break on
    sub-runs that do not run 1..N, even sorted ones.
    """

    @pytest.mark.xfail(reason="DD-2: rows are selected as `sub_run - 1`", raises=IndexError, strict=True)
    def test_save_reduced_data_of_sub_runs_not_starting_at_one(self, tmp_path):
        """With no raw counts loaded, the save always takes the subset branch."""
        # Arrange
        ws = make_minimal_HidraWorkspace(sub_runs=np.array([5, 6, 7]))
        path = tmp_path / "project.h5"
        project = HidraProjectFile(str(path), HidraProjectFileMode.OVERWRITE)
        ws.save_experimental_data(project, sub_runs=ws.get_sub_runs(), ignore_raw_counts=True)

        # Act
        ws.save_reduced_diffraction_data(project)
        project.save(False)

        # Assert
        readback = HidraWorkspace("readback")
        with HidraProjectFile(str(path), HidraProjectFileMode.READONLY) as project:
            readback.load_hidra_project(project, load_raw_counts=False, load_reduced_diffraction=True)
        np.testing.assert_allclose(readback._diff_data_set[None], ws._diff_data_set[None])

    @pytest.mark.xfail(
        reason="DD-2: `set_sample_log` discards the `matching_subruns` result", raises=AssertionError, strict=True
    )
    def test_set_sample_log_pairs_values_by_sub_run(self):
        """Used to store [30, 20, 10] for sub-runs [1, 2, 3], as given positionally."""
        # Arrange
        ws = make_minimal_HidraWorkspace(sub_runs=np.array([1, 2, 3]))

        # Act
        ws.set_sample_log("x", np.array([3, 2, 1]), np.array([30.0, 20.0, 10.0]))

        # Assert
        assert ws.get_sample_log_values("x").tolist() == [10.0, 20.0, 30.0]

    @pytest.mark.xfail(
        reason="DD-2: `set_sample_log` discards the `matching_subruns` result",
        raises=pytest.fail.Exception,
        strict=True,
    )
    def test_set_sample_log_for_other_sub_runs_raises(self):
        """Used to store the values against sub-runs they were not given for."""
        ws = make_minimal_HidraWorkspace(sub_runs=np.array([1, 2, 3]))
        with pytest.raises(RuntimeError):
            ws.set_sample_log("x", np.array([7, 8, 9]), np.array([30.0, 20.0, 10.0]))


class TestDirection:
    def test_a_constant_log_is_the_direction(self):
        ws = make_minimal_HidraWorkspace(sub_runs=np.array([1, 2, 3]))
        ws.set_sample_log("direction", np.array([1, 2, 3]), np.array(["22"] * 3))

        assert ws.direction == "22"

    def test_a_varying_log_raises(self):
        """A workspace merged from several directions has no single one.

        Used to be `get_sample_log_value`'s bare `AssertionError`, "Single log ...
        is a np array with multiple items".
        """
        ws = make_minimal_HidraWorkspace(sub_runs=np.array([1, 2, 3]))
        ws.set_sample_log("direction", np.array([1, 2, 3]), np.array(["11", "11", "22"]))

        with pytest.raises(ValueError, match=r"no single direction.*\['11', '22'\]"):
            ws.direction
