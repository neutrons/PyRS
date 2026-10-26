"""
By-hand study (not collected by pytest): export the ring points of calibration runs for the pyFAI
refinement in ``pyfai_goniometer_refinement.py``, and the detector images for pyFAI's own control-point
extraction in ``pyfai_control_points.py`` (pyFAI is not a PyRS dependency, so those scripts run in a
throwaway environment and only read this file).

Ring points are extracted exactly as the driver does: the mask as the workspace default mask, the
monochromator wavelength as the start, 5 strips, the 2020 lattice constants (docs/ground_truths.md).

Run from the repository root:

    pixi run python tests/scripts/pyfai_ring_points_export.py OUTPUT.npz RUN.nxs.h5 [RUN.nxs.h5 ...] [--mask MASK.xml]
"""

import argparse
from pathlib import Path

import numpy as np

from pyrs.calibration.pixel_pos_calibration import extract_ring_points
from pyrs.calibration.powder_standards import HB2B_STANDARDS, standard_for_sy
from pyrs.core.nexus_conversion import NeXusConvertingApp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output")
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--mask", default=None)
    args = parser.parse_args()

    arrays = {}
    names = []
    for run in args.runs:
        name = Path(run).name.split(".")[0]
        workspace = NeXusConvertingApp(run, args.mask).convert()
        wavelength = float(workspace.get_wavelength(False, False))
        setup = workspace.get_instrument_setup()
        points = extract_ring_points(workspace, wavelength, strips=5)
        names.append(name)
        arrays[f"{name}/mono_wavelength"] = np.array(wavelength)
        arrays[f"{name}/arm_length"] = np.array(setup.arm_length)
        arrays[f"{name}/detector_size"] = np.array(setup.detector_size)
        for field in ("sub_run", "index_0", "index_1", "det_2theta", "l2", "d_spacing", "hkl_squared", "sigma"):
            arrays[f"{name}/{field}"] = getattr(points, field)
        arrays[f"{name}/powder"] = points.powder.astype("U8")
        # detector images [index_0, index_1] per sub-run, the mask (1 = keep) and each sub-run's powder
        num_rows, num_columns = setup.detector_size
        sub_runs = [int(s) for s in workspace.get_sub_runs()]
        mask = workspace.get_detector_mask(True)
        keep = np.ones(num_columns * num_rows) if mask is None else np.asarray(mask, dtype=float)
        arrays[f"{name}/mask_keep"] = keep.reshape(num_columns, num_rows)
        arrays[f"{name}/image_sub_runs"] = np.array(sub_runs)
        arrays[f"{name}/images"] = np.array(
            [workspace.get_detector_counts(s).astype(float).reshape(num_columns, num_rows) for s in sub_runs]
        )
        arrays[f"{name}/image_det_2theta"] = np.array([float(workspace.get_detector_2theta(s)) for s in sub_runs])
        standards = [standard_for_sy(float(workspace.get_sample_log_value("sy", s)), HB2B_STANDARDS) for s in sub_runs]
        arrays[f"{name}/image_powder"] = np.array(["" if st is None else st.name for st in standards], dtype="U8")
        arrays[f"{name}/image_lattice"] = np.array([np.nan if st is None else st.lattice_a for st in standards])
        arrays[f"{name}/image_structure"] = np.array(["" if st is None else st.structure for st in standards], dtype="U8")
        print(f"{name}: {len(points)} ring points, mono {wavelength} A")
    arrays["runs"] = np.array(names)
    np.savez(args.output, **arrays)


if __name__ == "__main__":
    main()
