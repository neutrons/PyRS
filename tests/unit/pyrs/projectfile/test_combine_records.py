"""Combining reduced project files checks their calibrations (#1063, user decisions)

Different calibrations are refused; the same calibration with different vanadium runs needs the user's
approval; historic files without a calibration record may be combined but are flagged.
"""

import pytest

from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode  # type: ignore
from pyrs.projectfile.file_object import (
    CalibrationMismatchError,
    ReductionMismatchError,
    combine_reduction_records,
)

GEOMETRY = {"detector_size": [1024, 1024], "pixel_dimension": [0.3 / 1024, 0.3 / 1024], "arm_length": 0.985}
CALIBRATION = {
    "Shift_x": 0.001,
    "Shift_y": -0.002,
    "Shift_z": 0.003,
    "Rot_x": 0.1,
    "Rot_y": -0.2,
    "Rot_z": 0.3,
    "TTH_0": 0.25,
    "Lambda": 1.5306,
    "error_Shift_x": 1e-4,
    "error_Shift_y": 2e-4,
    "error_Shift_z": 3e-4,
    "error_Rot_x": 0.01,
    "error_Rot_y": 0.02,
    "error_Rot_z": 0.03,
    "error_TTH_0": -1.0,
    "error_Lambda": 1e-5,
    "Status": 1.0,
    "geometry": GEOMETRY,
}
SETTINGS = {
    "num_bins": 720,
    "min_2theta": None,
    "max_2theta": None,
    "delta_2theta": None,
    "eta_step": 3.0,
    "eta_min": -8.2,
    "eta_max": 8.2,
    "mask_id": None,
    "extra_mask": None,
}
MASK = {"source": "file", "file": "/HFIR/HB2B/shared/CALIBRATION/HB2B_Mask_Latest.xml", "sha256": "f00d" * 16}
REDUCTION = {"mask": MASK, "settings": SETTINGS}
RECORD = {
    "source": "file",
    "calibration_file": "/HFIR/HB2B/shared/CALIBRATION/Si422/HB2B_CAL_2026-11-03T11:40:37.json",
    "calibration_file_sha256": "abc",
    "calibration": CALIBRATION,
    "vanadium_run": "HB2B_931.h5",
    "reduction": REDUCTION,
}


def source(name, record=None):
    return {"file": f"/data/{name}.h5", "sha256": name * 2, "record": dict(record) if record is not None else {}}


def test_same_calibration_keeps_the_shared_record():
    # Act
    combined = combine_reduction_records([source("HB2B_1327", RECORD), source("HB2B_1328", RECORD)])

    # Assert
    assert {key: combined[key] for key in RECORD} == RECORD
    assert [s["file"] for s in combined["sources"]] == ["/data/HB2B_1327.h5", "/data/HB2B_1328.h5"]
    assert "warning" not in combined


@pytest.mark.parametrize(
    "change",
    [
        {"calibration": dict(CALIBRATION, Lambda=1.5291)},
        {"calibration": dict(CALIBRATION, error_Lambda=2e-5)},
        {"calibration": dict(CALIBRATION, Status=2.0)},
        {"calibration": dict(CALIBRATION, geometry=dict(GEOMETRY, arm_length=0.95))},
        {"calibration_file_sha256": "x"},
    ],
    ids=["values", "errors", "status", "geometry", "calibration-file"],
)
def test_different_calibrations_are_refused(change):
    # Arrange
    other = dict(RECORD, **change)

    # Act / Assert - the message groups the files by calibration
    with pytest.raises(CalibrationMismatchError, match="different calibrations") as refused:
        combine_reduction_records(
            [source("HB2B_1327", RECORD), source("HB2B_1328", RECORD), source("HB2B_1331", other)]
        )
    message = str(refused.value)
    assert "HB2B_1327.h5, HB2B_1328.h5:" in message and "HB2B_1331.h5:" in message


def test_different_vanadium_needs_approval():
    # Arrange
    sources = [source("HB2B_1327", RECORD), source("HB2B_1328", dict(RECORD, vanadium_run="HB2B_932.h5"))]

    # Act / Assert
    with pytest.raises(ReductionMismatchError, match="different vanadium runs") as asked:
        combine_reduction_records(sources)
    assert "HB2B_931.h5: HB2B_1327.h5" in str(asked.value) and "HB2B_932.h5: HB2B_1328.h5" in str(asked.value)


def test_approved_vanadium_difference_is_recorded():
    # Arrange
    sources = [source("HB2B_1327", RECORD), source("HB2B_1328", dict(RECORD, vanadium_run="HB2B_932.h5"))]

    # Act
    combined = combine_reduction_records(sources, accept_reduction_differences=True)

    # Assert - no single vanadium run claimed; each source keeps its own; the approval is noted
    assert "vanadium_run" not in combined and combined["calibration"] == CALIBRATION
    assert [s["record"]["vanadium_run"] for s in combined["sources"]] == ["HB2B_931.h5", "HB2B_932.h5"]
    assert "approved by the user" in combined["warning"]


def test_historic_files_are_combined_and_flagged():
    # Act
    combined = combine_reduction_records([source("HB2B_1327"), source("HB2B_1328")])

    # Assert
    assert combined["source"] == "not recorded" and len(combined["sources"]) == 2
    assert "2 of 2 files have no calibration record" in combined["warning"]


def test_historic_file_beside_recorded_ones_is_flagged():
    # Act
    combined = combine_reduction_records([source("HB2B_1327", RECORD), source("HB2B_1331")])

    # Assert - the recorded calibration is kept; the historic file is named as unchecked
    assert combined["calibration"] == CALIBRATION
    assert "1 of 2 files" in combined["warning"] and "HB2B_1331.h5" in combined["warning"]


def test_combined_record_round_trips_with_its_sources(tmp_path):
    # Arrange
    combined = combine_reduction_records([source("HB2B_1327", RECORD), source("HB2B_1331")])
    path = tmp_path / "combined.h5"

    # Act
    project = HidraProjectFile(str(path), HidraProjectFileMode.OVERWRITE)
    project.write_reduction_provenance(applied_calibration=combined)
    project.save(verbose=False)
    record = HidraProjectFile(str(path), HidraProjectFileMode.READONLY).read_reduction_provenance()

    # Assert - the shared record, the sources' own fingerprint verbatim, and the flag
    assert record["calibration_file"] == RECORD["calibration_file"]
    assert record["calibration_file_sha256"] == "abc"
    assert record["sources"] == combined["sources"] and record["warning"] == combined["warning"]


@pytest.mark.parametrize(
    "reduction, label",
    [
        (dict(REDUCTION, mask=dict(MASK, sha256="beef" * 16)), "masks"),  # the same file name, edited
        (dict(REDUCTION, mask={"source": "none"}), "masks"),
        (dict(REDUCTION, settings=dict(SETTINGS, extra_mask={"name": "upper"})), "masks"),
        (dict(REDUCTION, settings=dict(SETTINGS, num_bins=512)), "binning"),
        (dict(REDUCTION, settings=dict(SETTINGS, eta_step=None)), "binning"),
    ],
    ids=["mask-edited", "no-mask", "extra-mask", "num-bins", "no-eta-slices"],
)
def test_different_mask_or_binning_needs_approval(reduction, label):
    """#1068 (user decision): asked like a vanadium difference, not refused like a calibration one"""
    # Arrange
    sources = [source("HB2B_1327", RECORD), source("HB2B_1328", dict(RECORD, reduction=reduction))]

    # Act / Assert
    with pytest.raises(ReductionMismatchError, match=f"different {label}:"):
        combine_reduction_records(sources)


def test_approved_mask_difference_keeps_each_sources_record():
    # Arrange
    other = dict(REDUCTION, mask={"source": "none"})
    sources = [source("HB2B_1327", RECORD), source("HB2B_1328", dict(RECORD, reduction=other))]

    # Act
    combined = combine_reduction_records(sources, accept_reduction_differences=True)

    # Assert - no single mask claimed; the vanadium run, the same for both, is kept
    assert "reduction" not in combined and combined["vanadium_run"] == "HB2B_931.h5"
    assert [s["record"]["reduction"] for s in combined["sources"]] == [REDUCTION, other]
    assert "different masks, approved by the user" in combined["warning"]


def test_differences_are_listed_together():
    """One question for the user, naming every difference"""
    # Arrange
    other = dict(RECORD, vanadium_run="HB2B_932.h5", reduction=dict(REDUCTION, mask={"source": "none"}))

    # Act / Assert
    with pytest.raises(ReductionMismatchError, match="different vanadium runs, masks:"):
        combine_reduction_records([source("HB2B_1327", RECORD), source("HB2B_1328", other)])


def test_file_without_mask_record_is_flagged():
    """A file reduced after #1053 but before #1068: its calibration is checked, its mask cannot be"""
    # Arrange
    older = {key: value for key, value in RECORD.items() if key != "reduction"}

    # Act
    combined = combine_reduction_records([source("HB2B_1327", RECORD), source("HB2B_1328", older)])

    # Assert - the recorded mask and binning are kept; the older file is named
    assert combined["reduction"] == REDUCTION
    assert "1 of 2 files do not record which mask" in combined["warning"]
    assert "HB2B_1328.h5" in combined["warning"]


def test_unknown_masks_are_flagged_not_shared():
    """Re-reduced older files record their mask as "not recorded": not a mask the files share (review)"""
    # Arrange
    unknown = dict(RECORD, reduction=dict(REDUCTION, mask={"source": "not recorded"}))

    # Act
    combined = combine_reduction_records([source("HB2B_1327", unknown), source("HB2B_1328", unknown)])

    # Assert - the binning, recorded, is still shared
    assert combined["reduction"] == {"settings": SETTINGS}
    assert "2 of 2 files do not record which mask" in combined["warning"]


def test_unknown_mask_beside_a_known_one_is_not_asked():
    """Nothing is known to differ, so nothing to approve: the file is flagged instead"""
    # Arrange
    unknown = dict(RECORD, reduction=dict(REDUCTION, mask={"source": "not recorded"}))

    # Act
    combined = combine_reduction_records([source("HB2B_1327", RECORD), source("HB2B_1328", unknown)])

    # Assert
    assert combined["reduction"] == REDUCTION
    assert "HB2B_1328.h5" in combined["warning"]


def test_failed_reduction_record_is_named_as_failed():
    # Arrange
    failed = dict(RECORD, reduction={"error": "mask and binning could not be recorded: OSError: disk full"})

    # Act
    combined = combine_reduction_records([source("HB2B_1327", RECORD), source("HB2B_1328", failed)])

    # Assert
    assert "1 of 2 files' mask and binning record could not be written" in combined["warning"]
    assert "do not record which mask" not in combined["warning"]


def test_null_reduction_entry_is_not_recorded():
    """e.g. in an older combined file's sources"""
    # Act
    combined = combine_reduction_records(
        [source("HB2B_1327", RECORD), source("HB2B_1328", dict(RECORD, reduction=None))]
    )

    # Assert
    assert "HB2B_1328.h5" in combined["warning"]


def test_approved_binning_difference_keeps_the_shared_mask():
    # Arrange
    other = dict(REDUCTION, settings=dict(SETTINGS, num_bins=512))
    sources = [source("HB2B_1327", RECORD), source("HB2B_1328", dict(RECORD, reduction=other))]

    # Act
    combined = combine_reduction_records(sources, accept_reduction_differences=True)

    # Assert
    assert combined["reduction"] == {"mask": MASK}


def test_integer_and_float_settings_are_the_same_binning():
    # Arrange
    other = dict(REDUCTION, settings=dict(SETTINGS, eta_step=3))

    # Act
    combined = combine_reduction_records(
        [source("HB2B_1327", RECORD), source("HB2B_1328", dict(RECORD, reduction=other))]
    )

    # Assert
    assert "warning" not in combined
