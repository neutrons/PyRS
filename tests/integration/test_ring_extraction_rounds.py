"""Repeated ring extraction on real calibration runs (#1038)

A single extraction from a poor start can find only part of the rings and give a geometry tenths of a degree
wrong with a better RMS (HB2B_3489); extracting again at the refined geometry until converged recovers it.
"""

import json

import numpy as np
import pytest

from pyrs.calibration.pixel_pos_calibration import (
    CONVERGENCE_FRACTION,
    calibrate_rings,
    pixel_two_theta_change,
    statistical_precision,
)
from pyrs.core.nexus_conversion import NeXusConvertingApp

pytestmark = pytest.mark.integration

MASK = "tests/data/HB2B_Mask_12-18-19.xml"
RUN_3489 = "tests/data/calibration_tests/HB2B_3489.nxs.h5"
RUN_3510 = "tests/data/calibration_tests/HB2B_3510.nxs.h5"


@pytest.fixture(scope="module")
def run_3489():
    workspace = NeXusConvertingApp(RUN_3489, MASK).convert()
    return workspace, float(workspace.get_wavelength(False, False))


@pytest.fixture(scope="module")
def reference(run_3489):
    """The calibration from the nominal start"""
    workspace, mono = run_3489
    return calibrate_rings(workspace, mono)


def _poor_start(mono):
    """The start that failed silently in the #1038 study: shifts 5 mm, rotations 1 deg, wavelength +0.3%"""
    return np.array([0.005, 0.0, 0.005, -1.0, 1.0, -1.0, 0.0, mono * 1.003])


def test_one_extraction_from_a_poor_start_fails(run_3489, reference):
    """The failure the rounds remove: documented so a change to the extraction that fixes it is noticed"""
    # Arrange
    workspace, mono = run_3489

    # Act
    single = calibrate_rings(workspace, mono, start=_poor_start(mono), max_rounds=1)

    # Assert - tenths of a degree off, from about half the ring points
    change = pixel_two_theta_change(reference, reference.calibration_array, single.calibration_array)
    assert change > 0.3 and len(single.ring_points) < 0.7 * len(reference.ring_points)


def test_rounds_recover_from_a_poor_start(run_3489, reference):
    # Arrange
    workspace, mono = run_3489

    # Act
    recovered = calibrate_rings(workspace, mono, start=_poor_start(mono))

    # Assert - within the run's statistical precision (about 0.01 deg) of the nominal start's result
    change = pixel_two_theta_change(reference, reference.calibration_array, recovered.calibration_array)
    assert change < 3 * statistical_precision(reference)
    assert recovered.extraction_rounds[0]["ring_points"] < 0.7 * recovered.extraction_rounds[-1]["ring_points"]


def test_driver_records_every_round(tmp_path):
    """pyrs-pixel-calibration on HB2B_3510: converged by the rule, every round in the file's provenance"""
    import importlib.util
    from pathlib import Path

    # Arrange
    spec = importlib.util.spec_from_file_location(
        "pyrs_pixel_calibration", Path(__file__).resolve().parents[2] / "scripts" / "pyrs_pixel_calibration.py"
    )
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    output = tmp_path / "HB2B_3510.json"

    # Act
    calibration = driver.run_calibration(RUN_3510, str(output), mask_file=MASK)

    # Assert
    rounds = json.loads(output.read_text())["provenance"]["extraction_rounds"]
    assert 2 <= len(rounds) < 10 and rounds == calibration.extraction_rounds
    assert rounds[-1]["step_deg"] < CONVERGENCE_FRACTION * rounds[-1]["precision_deg"]
    assert calibration.warning_log == []
