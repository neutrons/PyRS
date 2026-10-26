"""
By-hand study (not collected by pytest): export the ring points of calibration runs for the pyFAI
refinement in ``pyfai_goniometer_refinement.py`` (pyFAI is not a PyRS dependency, so that script runs in a
throwaway environment and only reads this file).

Ring points are extracted exactly as the driver does: the mask as the workspace default mask, the
monochromator wavelength as the start, 5 strips, the 2020 lattice constants (docs/ground_truths.md).

Run from the repository root:

    pixi run python tests/scripts/pyfai_ring_points_export.py OUTPUT.npz RUN.nxs.h5 [RUN.nxs.h5 ...] [--mask MASK.xml]
"""

import argparse
from pathlib import Path

import numpy as np

from pyrs.calibration.pixel_pos_calibration import extract_ring_points
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
        print(f"{name}: {len(points)} ring points, mono {wavelength} A")
    arrays["runs"] = np.array(names)
    np.savez(args.output, **arrays)


if __name__ == "__main__":
    main()
