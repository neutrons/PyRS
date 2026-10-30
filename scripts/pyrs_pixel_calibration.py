"""
Ring-based HB2B detector calibration from one calibration run.

Workflow (each step is a function of :mod:`pyrs.calibration.pixel_pos_calibration`):

1. Convert the NeXus run with ``NeXusConvertingApp``; the mask file becomes the workspace's default
   mask.
2. Extract ring points from the detector images (``extract_ring_points``, 5 strips by default).
3. Refine the geometry and wavelength from the nominal start (``PixelCalibration.iterate``:
   Shift -> Rotation -> wavelength + 2theta_0 passes, then a joint refinement).
4. Estimate uncertainties with 2theta_0 held (``estimate_errors``): Shift_x, Rot_y and 2theta_0 are
   degenerate at one detector distance, so 2theta_0's error is written as -1.
5. Write the standard calibration JSON (``write_calibration``).

Usage (through pixi, from the repository)::

    pixi run pyrs-pixel-calibration HB2B_3510.nxs.h5 --output HB2B_CAL.json
    pixi run pyrs-pixel-calibration HB2B_3510.nxs.h5 --output cal.json --mask tests/data/HB2B_Mask_12-18-19.xml

The starting geometry is the nominal one; it must not be badly wrong (docs/ground_truths.md).
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

DEFAULT_MASK = "/HFIR/HB2B/shared/CALIBRATION/HB2B_Mask_Latest.xml"


def run_calibration(
    nexus_file: str,
    output: str,
    mask_file: Optional[str] = DEFAULT_MASK,
    wavelength: Optional[float] = None,
    strips: int = 5,
    passes: int = 3,
    polish: bool = True,
    ring_points_file: Optional[str] = None,
    sub_runs: Optional[Sequence[int]] = None,
):
    """Run the ring-based calibration on one NeXus run and write the calibration JSON.

    Args:
        nexus_file: Calibration run (NeXus).
        output: Calibration JSON to write.
        mask_file: Mantid XML detector mask, or None for no mask.
        wavelength: Starting wavelength (A); default: the run's monochromator setting.
        strips: Strips per sub-run for ring extraction.
        passes: Passes of the staged refinement.
        polish: Finish the passes with a joint refinement of all parameters.
        ring_points_file: If given, save the extracted ring points (``.npz``) here.
        sub_runs: Sub-runs to use (default: all).

    Returns:
        The refined ``PixelCalibration``.

    Raises:
        FileNotFoundError: If the NeXus or mask file does not exist.
    """
    from pyrs.calibration.pixel_pos_calibration import PixelCalibration, extract_ring_points
    from pyrs.core.nexus_conversion import NeXusConvertingApp

    if not Path(nexus_file).is_file():
        raise FileNotFoundError(f"NeXus file {nexus_file} does not exist")
    if mask_file is not None and not Path(mask_file).is_file():
        raise FileNotFoundError(f"mask file {mask_file} does not exist; give --mask PATH or --no-mask")

    workspace = NeXusConvertingApp(nexus_file, mask_file).convert()
    if wavelength is None:
        wavelength = float(workspace.get_wavelength(False, False))

    ring_points = extract_ring_points(workspace, wavelength, strips=strips, sub_runs=sub_runs)
    if ring_points_file is not None:
        ring_points.save(ring_points_file)

    calibration = PixelCalibration(ring_points, workspace.get_instrument_setup(), wavelength=wavelength)
    calibration.iterate(passes=passes, polish=polish)
    calibration.estimate_errors()
    calibration.write_calibration(output)
    return calibration


def _summary(calibration) -> str:
    from pyrs.calibration.pixel_pos_calibration import NOT_REFINED, PARAMETER_NAMES, PARAMETER_UNITS, UNDETERMINED

    points = calibration.ring_points
    rings = len(set(zip(points.sub_run.tolist(), points.hkl_squared.tolist())))
    lines = [f"{len(points)} ring points, {rings} rings; final RMS {calibration.stages[-1].rmse:.4f} deg"]
    for name, value, error, (unit, scale) in zip(
        PARAMETER_NAMES, calibration.calibration_array, calibration.calibration_error_array, PARAMETER_UNITS
    ):
        if error == NOT_REFINED:
            error_text = "held (2theta_0 is degenerate with Shift_x/Rot_y)" if name == "TTH_0" else "held"
        elif error == UNDETERMINED:
            error_text = "undetermined (degenerate, or stopped at a bound)"
        else:
            error_text = f"+/- {error * scale:.4g}"
        lines.append(f"  {name:8s} {value * scale:12.5f} {unit:3s} {error_text}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Command-line parser for :func:`main`; ``--help`` lists every option with its default."""
    parser = argparse.ArgumentParser(
        description="Ring-based HB2B detector calibration from one calibration run (NeXus).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("nexus_file", help="calibration run (NeXus)")
    parser.add_argument("--output", "-o", required=True, help="calibration JSON to write")
    masks = parser.add_mutually_exclusive_group()
    masks.add_argument("--mask", default=DEFAULT_MASK, help="Mantid XML detector mask")
    masks.add_argument("--no-mask", action="store_true", help="use every detector pixel")
    parser.add_argument("--wavelength", type=float, help="starting wavelength (A); None: the run's mono setting")
    parser.add_argument("--strips", type=int, default=5, help="strips per sub-run for ring extraction")
    parser.add_argument("--passes", type=int, default=3, help="passes of the staged refinement")
    parser.add_argument("--no-polish", action="store_true", help="skip the final joint refinement")
    parser.add_argument("--ring-points", help="save the extracted ring points to this .npz file")
    parser.add_argument("--sub-runs", type=int, nargs="+", help="sub-runs to use; None: all")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Run the calibration from the command line and print a parameter summary.

    Args:
        argv: Command-line arguments without the program name; default: ``sys.argv[1:]``.

    Returns:
        Exit status: 0 on success, 1 if an input file is missing, an input is invalid, or the
        conversion, fit or file writing fails.
    """
    args = build_parser().parse_args(argv)
    try:
        calibration = run_calibration(
            args.nexus_file,
            args.output,
            mask_file=None if args.no_mask else args.mask,
            wavelength=args.wavelength,
            strips=args.strips,
            passes=args.passes,
            polish=not args.no_polish,
            ring_points_file=args.ring_points,
            sub_runs=args.sub_runs,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(_summary(calibration))
    print(f"calibration written to {args.output}")
    return 0


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    sys.exit(main())
