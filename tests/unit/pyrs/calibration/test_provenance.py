"""Unit tests for pyrs.calibration.provenance: the calibration's provenance block (#1053)"""

import datetime
import hashlib
import json

import numpy as np
import pytest

import pyrs
from pyrs.calibration import provenance


def test_header_records_time_version_user_and_method():
    # Act
    block = provenance.header("ring")

    # Assert
    assert block["schema"] == provenance.SCHEMA_VERSION
    created = datetime.datetime.fromisoformat(block["created"])
    assert created.tzinfo is not None
    assert abs((datetime.datetime.now().astimezone() - created).total_seconds()) < 60
    assert block["pyrs_version"] == pyrs.__version__
    assert block["user"] and block["host"]
    assert block["method"] == "ring"


def test_header_unknown_method_raises():
    # Act / Assert
    with pytest.raises(ValueError, match="method must be one of"):
        provenance.header("guess")


def test_file_record_gives_path_and_sha256(tmp_path):
    # Arrange
    path = tmp_path / "run.nxs.h5"
    path.write_bytes(b"neutrons")

    # Act
    record = provenance.file_record(path)

    # Assert
    assert record == {"file": str(path.resolve()), "sha256": hashlib.sha256(b"neutrons").hexdigest()}


def test_file_record_missing_file_has_no_sha256(tmp_path):
    # Act
    record = provenance.file_record(tmp_path / "missing.xml")

    # Assert
    assert record["sha256"] is None


def test_file_record_none_is_none():
    # Act / Assert
    assert provenance.file_record(None) is None


@pytest.mark.parametrize(
    "powder, structure, hkl_squared, expected",
    [("Ni", "FCC", 3, "Ni 111"), ("Ni", "FCC", 8, "Ni 220"), ("Fe", "BCC", 2, "Fe 110"), ("Fe", "BCC", 6, "Fe 211")],
)
def test_reflection_label(powder, structure, hkl_squared, expected):
    # Act / Assert
    assert provenance.reflection_label(powder, structure, hkl_squared) == expected


def test_step_record_maps_values_and_errors_to_parameters():
    # Act
    step = provenance.step_record(
        2,
        "CalibrateShift",
        ("Shift_x", "Shift_y"),
        np.array([0.001, -0.002]),
        np.array([1e-4, -2.0]),
        status=0,
        n=50,
        rmse=0.025,
        nfev=20,
        message="max_nfev reached",
        undetermined_reasons={"Shift_y": "stopped at a bound"},
    )

    # Assert
    assert step["step"] == 2 and step["routine"] == "CalibrateShift"
    assert step["values"] == {"Shift_x": 0.001, "Shift_y": -0.002}
    assert step["errors"] == {"Shift_x": 1e-4, "Shift_y": -2.0}
    assert (step["status"], step["n"], step["rmse_deg"], step["nfev"]) == (0, 50, 0.025, 20)
    assert step["undetermined_reasons"] == {"Shift_y": "stopped at a bound"}
    json.dumps(step)


def test_step_record_mismatched_lengths_raises():
    # Act / Assert
    with pytest.raises(ValueError, match="2 free parameters"):
        provenance.step_record(1, "x", ("a", "b"), [1.0], [0.1, 0.2], status=1, n=None, rmse=None, nfev=None)


def test_fit_quality_matches_definitions():
    # Arrange
    residual = np.array([0.01, -0.02, 0.03, 0.0, -0.01, 0.02])
    labels = [(1, "Ni", "Ni 111", None)] * 3 + [(2, "Fe", "Fe 110", None)] * 3

    # Act
    quality = provenance.fit_quality(residual, n_free=2, labels=labels)

    # Assert
    chi2 = np.sum(residual**2) / 4
    assert (quality["n"], quality["n_free_parameters"], quality["dof"]) == (6, 2, 4)
    assert quality["rmse_deg"] == pytest.approx(np.sqrt(np.mean(residual**2)))
    assert quality["chi2_per_dof_deg2"] == pytest.approx(chi2)
    assert quality["scatter_deg"] == pytest.approx(np.sqrt(chi2))
    groups = {(entry["sub_run"], entry["reflection"]): entry for entry in quality["by_reflection_and_run"]}
    assert groups[(1, "Ni 111")]["n"] == 3
    assert groups[(1, "Ni 111")]["mean_deg"] == pytest.approx(np.mean(residual[:3]))
    assert groups[(2, "Fe 110")]["rms_deg"] == pytest.approx(np.sqrt(np.mean(residual[3:] ** 2)))
    assert "eta_mask" not in groups[(1, "Ni 111")]


def test_fit_quality_keeps_eta_mask_groups_apart():
    # Arrange
    labels = [(1, "Ni", "Ni 111", "eta_-3.0"), (1, "Ni", "Ni 111", "eta_3.0")]

    # Act
    quality = provenance.fit_quality(np.array([0.01, -0.01]), n_free=1, labels=labels)

    # Assert
    assert sorted(entry["eta_mask"] for entry in quality["by_reflection_and_run"]) == ["eta_-3.0", "eta_3.0"]


def test_fit_quality_no_dof_has_no_chi2():
    # Act
    quality = provenance.fit_quality(np.array([0.01, 0.02]), n_free=2)

    # Assert
    assert quality["dof"] == 0 and quality["chi2_per_dof_deg2"] is None and quality["scatter_deg"] is None


@pytest.mark.parametrize(
    "labels", [[(1, "Ni", "Ni 111", None)] * 2, [(1, "Ni", "Ni 111", None), None, (1, "Ni", "Ni 111", None)]]
)
def test_fit_quality_mismatched_labels_are_noted_not_guessed(labels):
    # Act
    quality = provenance.fit_quality(np.zeros(3), n_free=1, labels=labels)

    # Assert - the totals stay, the breakdown is empty and says why
    assert quality["n"] == 3 and quality["by_reflection_and_run"] == []
    assert "do not match" in quality["by_reflection_and_run_unavailable"]


def test_fit_quality_sorts_sub_runs_numerically():
    # Arrange
    labels = [(10, "Fe", "Fe 110", None), (2, "Ni", "Ni 111", None), (1, "Ni", "Ni 111", None)]

    # Act
    quality = provenance.fit_quality(np.array([0.1, 0.2, 0.3]), n_free=1, labels=labels)

    # Assert
    assert [entry["sub_run"] for entry in quality["by_reflection_and_run"]] == [1, 2, 10]


def test_fit_quality_nan_is_json_null():
    # Act
    quality = provenance.fit_quality(np.array([np.nan, 0.01]), n_free=1)

    # Assert
    assert quality["rmse_deg"] is None
    assert "NaN" not in json.dumps(quality)


def test_undetermined_lists_minus_two_with_reasons():
    # Act
    listed = provenance.undetermined(
        ["Shift_x", "Rot_y", "TTH_0"], [-2.0, 0.6, -2.0], {"Shift_x": "degenerate with Rot_y and TTH_0"}
    )

    # Assert
    assert listed == [
        {"parameter": "Shift_x", "reason": "degenerate with Rot_y and TTH_0"},
        {"parameter": "TTH_0", "reason": "not recorded"},
    ]


def test_reflection_label_never_raises():
    # Act / Assert - an unknown structure or a forbidden reflection gives a plain label
    assert provenance.reflection_label("Cu", None, 3) == "Cu h2+k2+l2=3"
    assert provenance.reflection_label("Ni", "FCC", 2) == "Ni h2+k2+l2=2"


def test_file_record_says_why_there_is_no_sha256(tmp_path):
    # Act
    record = provenance.file_record(tmp_path / "missing.xml")

    # Assert
    assert record["sha256"] is None and record["sha256_error"].startswith("FileNotFoundError")


def test_step_record_nan_becomes_null():
    # Act
    step = provenance.step_record(1, "x", ["a"], [np.nan], [np.inf], status=1, n=None, rmse=np.nan, nfev=None)

    # Assert
    assert step["values"] == {"a": None} and step["errors"] == {"a": None} and step["rmse_deg"] is None
    json.dumps(step, allow_nan=False)


def test_jsonable_converts_numpy_types():
    # Arrange
    import enum

    class Mono(enum.Enum):
        Si422 = 1.54

    value = {
        "bins": np.int64(512),
        "eta": np.float32(3.0),
        "keep": np.array([True, False]),
        "t": (1, 2),
        "mono": Mono.Si422,
        "nan": np.nan,
    }

    # Act
    converted = provenance.jsonable(value)

    # Assert
    assert converted == {"bins": 512, "eta": 3.0, "keep": [True, False], "t": [1, 2], "mono": "Si422", "nan": None}
    assert type(converted["bins"]) is int
    json.dumps(converted, allow_nan=False)


def test_safe_build_returns_the_block():
    # Act / Assert
    assert provenance.safe_build(lambda: {"schema": 1, "bins": np.int64(5)}) == {"schema": 1, "bins": 5}


@pytest.mark.parametrize(
    "build, reason",
    [
        (lambda: 1 / 0, "ZeroDivisionError"),
        (lambda: {"thing": object()}, "TypeError"),  # not JSON: caught here, not when writing the file
    ],
)
def test_safe_build_never_raises(build, reason):
    # Act
    with pytest.warns(UserWarning, match="could not be built"):
        block = provenance.safe_build(build)

    # Assert
    assert block["schema"] == provenance.SCHEMA_VERSION and reason in block["error"]


def test_assemble_renumbers_steps_and_collects_reasons():
    # Arrange
    steps = [
        provenance.step_record(
            7,
            "CalibrateShift",
            ["Shift_x"],
            [0.0],
            [-2.0],
            1,
            10,
            0.1,
            3,
            undetermined_reasons={"Shift_x": "degenerate"},
        ),
        provenance.step_record(8, "CalibrateRotation", ["Rot_x"], [0.0], [0.1], 1, 10, 0.1, 3),
    ]

    # Act
    block = provenance.assemble(
        "ring",
        {"runs": []},
        "nominal",
        [0.0, 1.54],
        steps,
        None,
        "model",
        ["Shift_x", "Lambda"],
        [-2.0, 1e-4],
        extra={"held_parameters": []},
    )

    # Assert
    assert [step["step"] for step in block["steps"]] == [1, 2]
    assert block["start"] == {"source": "nominal", "values": {"Shift_x": 0.0, "Lambda": 1.54}}
    assert block["undetermined"] == [{"parameter": "Shift_x", "reason": "degenerate"}]
    assert block["held_parameters"] == [] and block["method"] == "ring"
