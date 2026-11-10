"""
Ring-based HB2B detector calibration from one calibration run.

Workflow (each step is a function of :mod:`pyrs.calibration.pixel_pos_calibration`):

1. Convert the NeXus run with ``NeXusConvertingApp``; the mask file becomes the workspace's default
   mask.
2. Extract ring points from the detector images (``extract_ring_points``, 5 strips by default) and
   refine the geometry and wavelength from the nominal start (``PixelCalibration.iterate``:
   Shift -> Rotation -> wavelength + 2theta_0 passes, then a joint refinement).
3. Extract the rings again where the refined geometry predicts them and refine, until a round moves the
   pixel 2theta and the predicted ring positions by less than half the data's statistical precision (at
   least 2, at most ``--max-rounds``
   rounds; ``calibrate_rings``, #1038). A single extraction from a poor start can find part of the rings
   and give a wrong geometry with a good RMS (docs/ground_truths.md).
4. Estimate uncertainties with 2theta_0 held (``estimate_errors``): Shift_x, Rot_y and 2theta_0 are
   degenerate at one detector distance, so 2theta_0's error is written as -1.
5. Write the standard calibration JSON (``write_calibration``), with every round in its provenance.

Usage (through pixi, from the repository)::

    pixi run pyrs-pixel-calibration HB2B_3510.nxs.h5 --output HB2B_CAL.json
    pixi run pyrs-pixel-calibration HB2B_3510.nxs.h5 --output cal.json --mask tests/data/HB2B_Mask_12-18-19.xml

The starting geometry is the nominal one; it must not be badly wrong (docs/ground_truths.md).
"""

import argparse
import warnings
import sys
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from pyrs.calibration.pixel_pos_calibration import PixelCalibration

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
    max_rounds: Optional[int] = None,
    nexus_output: Optional[str] = None,
) -> "PixelCalibration":
    """Run the ring-based calibration on one NeXus run and write the calibration JSON (and, if asked, NeXus).

    Args:
        nexus_file: Calibration run (NeXus).
        output: Calibration JSON to write.
        mask_file: Mantid XML detector mask, or None for no mask.
        wavelength: Starting wavelength (A); default: the run's monochromator setting.
        strips: Strips per sub-run for ring extraction.
        passes: Passes of the staged refinement.
        polish: Finish the passes with a joint refinement of all parameters.
        ring_points_file: If given, save the last round's ring points (``.npz``) here. Deprecated (#1054):
            the NeXus file (``nexus_output``) holds them, with the strip fits and residuals.
        sub_runs: Sub-runs to use (default: all).
        max_rounds: Most extraction + refinement rounds (default: ``DEFAULT_MAX_ROUNDS``); 1 extracts once.
        nexus_output: If given, also write the calibration as a NeXus file (``.nxs``) with its diagnostics:
            ring points, strip fits, residuals and covariance (#1054).

    Returns:
        The refined ``PixelCalibration``.

    Raises:
        FileNotFoundError: If the NeXus or mask file does not exist.
        ValueError: If ``nexus_output`` is not a ``.nxs`` file (checked before the calibration runs).
    """
    from pyrs.calibration.pixel_pos_calibration import DEFAULT_MAX_ROUNDS, calibrate_rings
    from pyrs.utilities.calibration_file_io import calibration_format
    from pyrs.calibration.provenance import file_record
    from pyrs.core import MonoSetting  # type: ignore
    from pyrs.core.nexus_conversion import NeXusConvertingApp

    if not Path(nexus_file).is_file():
        raise FileNotFoundError(f"NeXus file {nexus_file} does not exist")
    if mask_file is not None and not Path(mask_file).is_file():
        raise FileNotFoundError(f"mask file {mask_file} does not exist; give --mask PATH or --no-mask")
    if nexus_output is not None and calibration_format(nexus_output) != "nexus":
        raise ValueError(f"--nexus must name a .nxs file, got {nexus_output!r}")
    if ring_points_file is not None:
        warnings.warn(
            "--ring-points (.npz) is deprecated and will be removed: write the NeXus calibration file with --nexus, "
            "which holds the ring points under /entry/calibration/ring_points",
            FutureWarning,
            stacklevel=2,
        )

    workspace = NeXusConvertingApp(nexus_file, mask_file).convert()
    if wavelength is None:
        wavelength = float(workspace.get_wavelength(False, False))

    if max_rounds is None:
        max_rounds = DEFAULT_MAX_ROUNDS
    strip_fits: list = []  # the last round's, for the NeXus diagnostics
    calibration = calibrate_rings(
        workspace,
        wavelength,
        strips=strips,
        sub_runs=sub_runs,
        passes=passes,
        polish=polish,
        max_rounds=max_rounds,
        diagnostics=strip_fits,
    )
    if ring_points_file is not None:
        calibration.ring_points.save(ring_points_file)
    calibration.estimate_errors()
    monochromator_error = None
    try:
        monochromator: Optional[str] = MonoSetting.getFromRotation(workspace.get_sample_log_value("mrot", 1)).name
    except (AssertionError, KeyError, ValueError) as mono_err:  # no or ambiguous mrot log: recorded as unknown
        monochromator, monochromator_error = None, f"{type(mono_err).__name__}: {mono_err}"
    inputs = {
        "runs": [file_record(nexus_file)],
        "mask": file_record(mask_file),
        "monochromator": monochromator,
        "monochromator_error": monochromator_error,
        "wavelength_start_A": wavelength,
        "extraction": {
            "strips": strips,
            "sub_runs": None if sub_runs is None else [int(s) for s in sub_runs],
            "max_rounds": max_rounds,
        },
        "iteration": {"passes": passes, "polish": polish, "errors": "estimate_errors, TTH_0 held"},
        "driver": "scripts/pyrs_pixel_calibration.py",
    }
    calibration.write_calibration(output, inputs=inputs)
    if nexus_output is not None:
        calibration.write_calibration(nexus_output, inputs=inputs, strip_fits=strip_fits)
    return calibration


def _summary(calibration: "PixelCalibration") -> str:
    """Parameter table of a refined calibration, with each error code spelled out."""
    from pyrs.calibration.pixel_pos_calibration import NOT_REFINED, PARAMETER_NAMES, PARAMETER_UNITS, UNDETERMINED

    points = calibration.ring_points
    lines = [f"{len(points)} ring points, {points.num_rings} rings; final RMS {calibration.stages[-1].rmse:.4f} deg"]
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
    if calibration.calibration_status == 0:
        lines.append("WARNING: the last stage stopped at max_nfev before converging (Status 0)")
    rounds = calibration.extraction_rounds or []
    if rounds:
        lines.append(
            f"ring extraction rounds: {len(rounds)} (step: change from the previous round, the larger of the pixel "
            "2theta and the ring positions)"
        )
        for record in rounds:
            step = "-" if record["step_deg"] is None else f"{record['step_deg'] * 1e3:.2f}"
            lines.append(
                f"  round {record['round']:2d}: {record['ring_points']:4d} points, RMS {record['rmse_deg']:.4f} deg,"
                f" step {step:>6s} mdeg, precision {record['precision_deg'] * 1e3:.2f} mdeg"
            )
    if calibration.extraction_converged is False:
        lines.append(f"WARNING: the ring extraction did not converge within {len(rounds)} rounds (see the steps)")
    return "\n".join(lines)


def _positive_int(text: str) -> int:
    """An argparse type: a whole number of at least 1."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {text!r}")
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


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
    parser.add_argument(
        "--ring-points", help="save the extracted ring points to this .npz file (deprecated: use --nexus)"
    )
    parser.add_argument(
        "--nexus",
        help="also write the calibration as this NeXus (.nxs) file, with ring points, strip fits and residuals",
    )
    parser.add_argument("--sub-runs", type=int, nargs="+", help="sub-runs to use; None: all")
    parser.add_argument(
        "--max-rounds",
        type=_positive_int,
        default=10,  # pixel_pos_calibration.DEFAULT_MAX_ROUNDS; pyrs is imported lazily so --help stays fast
        help="most ring extraction + refinement rounds; the rings are extracted again at the refined geometry "
        "until a round's step (pixel 2theta or ring positions) is less than half the statistical precision "
        "(1: extract once)",
    )
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
            max_rounds=args.max_rounds,
            nexus_output=args.nexus,
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(_summary(calibration))
    print(f"calibration written to {args.output}")
    if args.nexus:
        print(f"NeXus calibration (with diagnostics) written to {args.nexus}")
    return 0


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    sys.exit(main())
