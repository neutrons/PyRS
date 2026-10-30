"""
By-hand audit (not collected by pytest): which calibration files carry a non-zero 2theta offset, and so
change reductions once the reader uses it (#1021, docs/ground_truths.md).

Before the fix, ``read_calibration_json_file`` read the 2theta offset from ``two_theta_0`` while files store
it as ``TTH_0``, so reduction used 2theta_0 = 0 for every file. With the fix, a file with ``TTH_0 = d``
shifts every reduced pattern by ``d`` in 2theta (exactly on the detector's centre row, 0.98-1.00 d
elsewhere), i.e. an absolute d-spacing / strain change of ``(d / 2) cot(theta)``.

Run on a machine that sees the archive (default ``/HFIR/HB2B/shared/CALIBRATION``):

    pixi run python tests/scripts/audit_calibration_tth0.py [DIRECTORY ...] [--csv OUT.csv]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

DEFAULT = "/HFIR/HB2B/shared/CALIBRATION"
TWO_THETAS = (60.0, 90.0, 120.0)  # deg, for the strain change


def offset(calibration: dict):
    """(2theta_0, key it came from) of a calibration dictionary; (None, '') if it has none."""
    for key in ("TTH_0", "two_theta_0"):
        if key in calibration:
            return float(calibration[key]), key
    return None, ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("directories", nargs="*", default=[DEFAULT])
    parser.add_argument("--csv", help="also write the table to this CSV file")
    args = parser.parse_args()

    rows = []
    for directory in args.directories:
        root = Path(directory)
        if not root.is_dir():
            print(f"error: {root} is not a directory", file=sys.stderr)
            return 1
        for path in sorted(root.rglob("*.json")):
            try:
                calibration = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if not isinstance(calibration, dict) or "Shift_x" not in calibration:
                continue  # not a detector calibration
            tth0, key = offset(calibration)
            strains = [
                abs(np.radians(tth0 or 0.0) / 2 / np.tan(np.radians(t) / 2)) * 1e6 for t in TWO_THETAS
            ]
            rows.append(
                dict(
                    file=str(path),
                    key=key or "(none)",
                    tth0=tth0 if tth0 is not None else 0.0,
                    status=calibration.get("Status", ""),
                    wavelength=calibration.get("Lambda", ""),
                    **{f"strain_ue_at_{t:g}": round(s, 1) for t, s in zip(TWO_THETAS, strains)},
                )
            )

    affected = [r for r in rows if r["tth0"] != 0.0]
    print(f"{len(rows)} calibration files; {len(affected)} with a non-zero 2theta offset (reductions change)")
    header = f"{'TTH_0 (deg)':>12} {'key':>12} {'status':>7} {'lambda':>8}  " + " ".join(
        f"{'ue@' + format(t, 'g'):>8}" for t in TWO_THETAS
    )
    print(header + "  file")
    for r in sorted(rows, key=lambda r: -abs(r["tth0"])):
        strain_text = " ".join(f"{r[f'strain_ue_at_{t:g}']:8.0f}" for t in TWO_THETAS)
        print(f"{r['tth0']:12.5f} {r['key']:>12} {str(r['status']):>7} {str(r['wavelength']):>8}  {strain_text}  {r['file']}")
    if args.csv and rows:
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"table written to {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
