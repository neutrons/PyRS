"""
By-hand study, part 1 of 2 (not collected by pytest): export the ring calibration's forward model
(DetectorModel) as full-detector 2theta and eta maps, for comparison with pyFAI in
``pyfai_forward_model_check.py``. pyFAI is not a PyRS dependency, so the check runs in a throwaway
environment and only reads this file.

Geometries: nominal (no calibration) and two refinements of HB2B_3510 (docs/ground_truths.md):

* ``free_0.293``: the driver's free fit (mask, 5 strips, 2020 lattice), pitch 0.29297 mm;
* ``joint_0.314``: the joint 3510 + 3489 fit with the 0.314 mm pitch (3510's 2theta_0).

Arm angles are those of HB2B_3510 plus 0 deg (straight through), in the PyRS convention
(``two_theta = -get_detector_2theta``).

Run from the repository root:

    pixi run python tests/scripts/pyfai_forward_model_export.py OUTPUT.npz
"""

import sys

import numpy as np

from pyrs.calibration.pixel_pos_calibration import DetectorModel
from pyrs.core.instrument_geometry import DENEXDetectorGeometry, DENEXDetectorShift

ARM_LENGTH = 0.985
DETECTOR_ARMS = (0.0, 76.0, 89.0, 103.0)  # get_detector_2theta values
GEOMETRIES = {
    # name: (pitch m, calibration or None)
    "nominal": (0.3 / 1024, None),
    "free_0.293": (
        0.3 / 1024,
        DENEXDetectorShift(-0.024217696, -0.002044255, -0.074748588, -0.021086935, 2.920418640, 0.180709492, 1.454954624),
    ),
    "joint_0.314": (0.314e-3, DENEXDetectorShift(-0.003534, 0.000435, -0.009885, -0.266, 5.817, 0.120, 0.141)),
}


def main(output: str) -> None:
    grid = np.meshgrid(np.arange(1024), np.arange(1024), indexing="ij")  # (index_0, index_1)
    arrays = {"detector_arms": np.array(DETECTOR_ARMS), "arm_length": np.array(ARM_LENGTH)}
    for name, (pitch, calibration) in GEOMETRIES.items():
        model = DetectorModel(DENEXDetectorGeometry(1024, 1024, pitch, pitch, ARM_LENGTH, False))
        arrays[f"{name}/pitch"] = np.array(pitch)
        arrays[f"{name}/calibration"] = np.array(
            [0.0] * 7
            if calibration is None
            else [
                calibration.center_shift_x,
                calibration.center_shift_y,
                calibration.center_shift_z,
                calibration.rotation_x,
                calibration.rotation_y,
                calibration.rotation_z,
                calibration.two_theta_0,
            ]
        )
        for arm in DETECTOR_ARMS:
            arrays[f"{name}/{arm:g}/tth"] = model.two_theta(grid[0], grid[1], -arm, ARM_LENGTH, calibration)
            arrays[f"{name}/{arm:g}/eta"] = model.eta(grid[0], grid[1], -arm, ARM_LENGTH, calibration)
    np.savez(output, **arrays)
    print(f"wrote {len(arrays)} arrays to {output}")


if __name__ == "__main__":
    main(sys.argv[1])
