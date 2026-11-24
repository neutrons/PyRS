import h5py
import numpy as np
import pytest

from pyrs.core.instrument_geometry import DENEXDetectorGeometry, DENEXDetectorShift, HidraSetup
from pyrs.core.workspaces import HidraWorkspace
from pyrs.dataobjects.constants import HidraConstants
from pyrs.dataobjects.sample_logs import SampleLogs
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode  # type: ignore
from pyrs.utilities.calibration_file_io import write_calibration_to_json
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
        reason="DD-2: values are paired with sub-runs positionally, so another order is refused",
        raises=RuntimeError,
        strict=True,
    )
    def test_set_sample_log_pairs_values_by_sub_run(self):
        """Used to store [30, 20, 10] for sub-runs [1, 2, 3], as given positionally; now refused."""
        # Arrange
        ws = make_minimal_HidraWorkspace(sub_runs=np.array([1, 2, 3]))

        # Act
        ws.set_sample_log("x", np.array([3, 2, 1]), np.array([30.0, 20.0, 10.0]))

        # Assert
        assert ws.get_sample_log_values("x").tolist() == [10.0, 20.0, 30.0]

    def test_set_sample_log_for_other_sub_runs_raises(self):
        """Used to store the values against sub-runs they were not given for."""
        ws = make_minimal_HidraWorkspace(sub_runs=np.array([1, 2, 3]))
        with pytest.raises(RuntimeError, match=r"given for sub-runs \[7, 8, 9\]"):
            ws.set_sample_log("x", np.array([7, 8, 9]), np.array([30.0, 20.0, 10.0]))


class TestSubRunAxisMatchesTheData:
    """The sub-run axis and every per-sub-run array must agree, whichever is set first.

    Each array is paired with the axis positionally, so a disagreement attaches
    values to the wrong sub-runs. Data set after the axis were already checked;
    these cover the axis set, or replaced, after the data.
    """

    @staticmethod
    def _reduced(ws: HidraWorkspace, n_rows: int) -> None:
        shape = (n_rows, 4)
        ws.set_reduced_diffraction_data_set(np.ones(shape), {None: np.ones(shape)}, {None: np.ones(shape)})

    def test_axis_after_data_of_another_length_raises(self):
        ws = HidraWorkspace()
        self._reduced(ws, 5)

        with pytest.raises(RuntimeError, match=r"5 row\(s\) of two-theta matrix"):
            ws.set_sub_runs([1, 2, 3])

    def test_axis_after_data_of_the_same_length_is_accepted(self):
        ws = HidraWorkspace()
        self._reduced(ws, 3)

        ws.set_sub_runs([1, 2, 3])

        assert ws.get_sub_runs().raw_copy().tolist() == [1, 2, 3]

    def test_raw_counts_for_another_sub_run_raise(self):
        ws = HidraWorkspace()
        ws.set_raw_counts(9, np.zeros(4))

        with pytest.raises(RuntimeError, match=r"raw counts for sub-run\(s\) \[9\]"):
            ws.set_sub_runs([1, 2, 3])

    def test_replacement_logs_with_another_axis_raise(self):
        ws = make_minimal_HidraWorkspace(sub_runs=[1, 2, 3])
        replacement = SampleLogs()
        replacement.subruns = [1, 2]

        with pytest.raises(RuntimeError, match=r"replacement sample logs has sub-runs \[1, 2\]"):
            ws.set_sample_logs_from_object(replacement)


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


def _project_with_record(tmp_path, instrument=None):
    """A project file written from the minimal workspace, with the reduction's calibration record if given"""
    ws = make_minimal_HidraWorkspace(sub_runs=np.array([1, 2, 3]))
    path = tmp_path / "project.h5"
    project = HidraProjectFile(str(path), HidraProjectFileMode.OVERWRITE)
    ws.save_experimental_data(project, sub_runs=ws.get_sub_runs(), ignore_raw_counts=True)
    if instrument is not None:
        project.write_reduction_provenance(instrument=instrument)
    project.save(False)
    readback = HidraWorkspace("readback")
    with HidraProjectFile(str(path), HidraProjectFileMode.READONLY) as project:
        readback.load_hidra_project(project, load_raw_counts=False, load_reduced_diffraction=False)
    return readback


def test_recorded_calibration_is_loaded_apart(tmp_path):
    """The calibration reduction applied is kept with its errors (#1099), and is not the shift reduction applies"""
    # Arrange - a reduction's record of a calibration file
    shift = DENEXDetectorShift(0.001, -0.002, 0.003, 0.1, -0.2, 0.3, 0.25)
    errors = DENEXDetectorShift(1e-4, 2e-4, 3e-4, 0.01, 0.02, 0.03, -1.0)
    calibration = tmp_path / "HB2B_CAL.json"
    write_calibration_to_json(shift, errors, 1.5306, 1e-5, 1, str(calibration))
    instrument = HidraSetup(DENEXDetectorGeometry(4, 4, 0.001, 0.001, 2.0, False))
    instrument.load_calibration_file(str(calibration))

    # Act
    readback = _project_with_record(tmp_path, instrument)

    # Assert
    recorded_shift, recorded_errors = readback.get_recorded_calibration()
    assert recorded_shift.convert_to_dict() == pytest.approx(shift.convert_to_dict())
    assert recorded_errors.convert_to_dict() == pytest.approx(errors.convert_to_dict())
    assert readback.get_detector_shift() is None  # reduction of this project still applies none by default


def test_no_record_no_recorded_calibration(tmp_path):
    """A file whose reduction recorded nothing has no applied calibration to report"""
    # Arrange / Act - a project file written without a reduction record
    readback = _project_with_record(tmp_path)

    # Assert
    assert readback.get_recorded_calibration() is None


def test_record_of_no_calibration_is_none(tmp_path):
    """A reduction that recorded applying no calibration (source "none") is no applied calibration"""
    # Arrange - an instrument with no calibration loaded
    instrument = HidraSetup(DENEXDetectorGeometry(4, 4, 0.001, 0.001, 2.0, False))

    # Act
    readback = _project_with_record(tmp_path, instrument)

    # Assert
    assert readback.get_recorded_calibration() is None


class _Warnings:
    """Stands in for the workspace's Mantid logger, keeping what it warns"""

    def __init__(self):
        self.messages = []

    def warning(self, message):
        self.messages.append(message)

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def test_unreadable_record_warns_and_the_data_loads(tmp_path):
    """The record is optional: one that cannot be read is warned about and the data still load"""
    # Arrange - a calibration record whose JSON is damaged
    path = _project_file(tmp_path, "project.h5", _calibrated_instrument(tmp_path, 0.3))
    with h5py.File(path, "a") as f:
        group = f[f"{HidraConstants.INSTRUMENT}/{HidraConstants.CALIBRATION}"]
        del group["calibration_json"]
        group["calibration_json"] = "{not json"
    ws = HidraWorkspace("readback")
    ws._log = _Warnings()

    # Act
    with HidraProjectFile(str(path), HidraProjectFileMode.READONLY) as project:
        ws.load_hidra_project(project, load_raw_counts=False, load_reduced_diffraction=False)

    # Assert
    assert ws.get_recorded_calibration() is None
    np.testing.assert_array_equal(ws.get_sub_runs(), [1, 2, 3])
    assert any("applied calibration could not be read" in message for message in ws._log.messages)


def _calibrated_instrument(tmp_path, rot_z):
    calibration = tmp_path / f"HB2B_CAL_{rot_z}.json"
    shift = DENEXDetectorShift(0.001, -0.002, 0.003, 0.1, -0.2, rot_z, 0.25)
    write_calibration_to_json(shift, DENEXDetectorShift(*[1e-4] * 7), 1.5306, 1e-5, 1, str(calibration))
    instrument = HidraSetup(DENEXDetectorGeometry(4, 4, 0.001, 0.001, 2.0, False))
    instrument.load_calibration_file(str(calibration))
    return instrument


def _project_file(tmp_path, name, instrument=None):
    ws = make_minimal_HidraWorkspace(sub_runs=np.array([1, 2, 3]))
    path = tmp_path / name
    project = HidraProjectFile(str(path), HidraProjectFileMode.OVERWRITE)
    ws.save_experimental_data(project, sub_runs=ws.get_sub_runs(), ignore_raw_counts=True)
    if instrument is not None:
        project.write_reduction_provenance(instrument=instrument)
    project.save(False)
    return path


@pytest.mark.parametrize("second", ["same", "other", "none"])
def test_appended_file_keeps_only_a_shared_calibration(tmp_path, second):
    """Files reduced with different calibrations (or one without) record none for the whole (#1099): NXstress must
    not label every scan point with the first file's"""
    # Arrange
    first = _project_file(tmp_path, "a.h5", _calibrated_instrument(tmp_path, 0.3))
    other = {"same": _calibrated_instrument(tmp_path, 0.3), "other": _calibrated_instrument(tmp_path, 0.4)}
    appended = _project_file(tmp_path, "b.h5", other.get(second))
    ws = HidraWorkspace("combined")
    with HidraProjectFile(str(first), HidraProjectFileMode.READONLY) as project:
        ws.load_hidra_project(project, load_raw_counts=False, load_reduced_diffraction=True)

    # Act
    with HidraProjectFile(str(appended), HidraProjectFileMode.READONLY) as project:
        ws.append_hidra_project(project)

    # Assert
    if second == "same":
        assert ws.get_recorded_calibration()[0].rotation_z == pytest.approx(0.3)
    else:
        assert ws.get_recorded_calibration() is None
