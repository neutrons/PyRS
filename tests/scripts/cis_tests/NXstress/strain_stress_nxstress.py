"""The strain/stress viewer's NXstress save and load, and its saved state, on real data.

Drives the viewer's model, the same object the window drives, through what the
GUI steps do by hand. Shows that:

1. three real runs -- e11 from 1328, e22 from 1331, e33 from 1332 -- with stress
   calculated, are saved as one `.nxs` file (**Save as NXstress...**);
2. loaded into a fresh model (**Load NXstress...**), every direction comes back
   from its own run, and the stress summary CSV is reproduced. Peak heights
   agree to about 2e-7 relative, not exactly: NXstress stores a pseudo-Voigt fit
   as height, FWHM and form factor in float32 and rebuilds the rest on read.
   Strain and stress are unaffected;
3. the saved state (**File > Save**) of the loaded model names the `.nxs` file
   for each workspace, and restoring it (**File > Load**) in a third model
   reproduces the summary exactly.

Leaves `strain_stress.nxs` and `strain_stress_state.json` in the output directory
for the PR's GUI steps.

Run from the repository root:

    pixi run python tests/scripts/cis_tests/NXstress/strain_stress_nxstress.py [output-dir]
"""

import json
import math
from pathlib import Path

from _fixtures import TEST_DATA, check, output_dir, run, section

from pyrs.interface.strainstressviewer.model import Model

FILES = {"11": "HB2B_1328", "22": "HB2B_1331", "33": "HB2B_1332"}
PEAK = "peak0"
E, NU = 200.0, 0.3


def model_reporting_failures(failures: list) -> Model:
    model = Model()
    model.failureMsg.connect(lambda *args: failures.append(args[:2]))
    return model


def calculate(model: Model) -> None:
    model.selectedPeak = PEAK
    model.calculate_stress("diagonal", E, NU, None)


def runs(model: Model) -> dict[str, list[int]]:
    return {d: [int(peaks[PEAK].runnumber) for peaks in getattr(model, f"e{d}_peaks")] for d in FILES}


def largest_difference(original: Path, other: Path) -> tuple[float, int, set[str]]:
    """The largest relative difference between two CSV summaries, the cells that differ, and their columns.

    Compared cell by cell in position, as written; text cells must match exactly.
    """
    lines_a, lines_b = original.read_text().splitlines(), other.read_text().splitlines()
    if len(lines_a) != len(lines_b):
        raise SystemExit(f"FAILED: {len(lines_a)} lines against {len(lines_b)}")
    header = next(line for line in lines_a if not line.startswith("#")).split(", ")
    worst, differing, columns = 0.0, 0, set()
    for a, b in zip(lines_a, lines_b):
        cells_a, cells_b = a.split(","), b.split(",")
        if len(cells_a) != len(cells_b):
            raise SystemExit(f"FAILED: {a!r} against {b!r}")
        for n, (x, y) in enumerate(zip(cells_a, cells_b)):
            if x.strip() == y.strip():
                continue
            fx, fy = float(x), float(y)
            # A NaN relative difference would lose to any `worst` in `max`, so a value
            # that became NaN or infinite would pass unnoticed.
            if not (math.isfinite(fx) and math.isfinite(fy)):
                raise SystemExit(f"FAILED: {x.strip()!r} against {y.strip()!r} in {a!r}")
            if fx == fy:  # 0.0 against -0.0
                continue
            relative = abs(fx - fy) / max(abs(fx), abs(fy))
            worst, differing = max(worst, relative), differing + 1
            columns.add(header[n] if n < len(header) else f"column {n}")
    return worst, differing, columns


def main() -> None:
    out = output_dir()
    failures: list = []

    section("three real runs, stress calculated, saved as NXstress")
    original = model_reporting_failures(failures)
    for direction, stem in FILES.items():
        setattr(original, f"e{direction}", [str(TEST_DATA / f"{stem}_with_instrument.h5")])
    calculate(original)
    nxs = out / "strain_stress.nxs"
    original.save_as_nxstress(str(nxs))
    original_csv = out / "strain_stress_original.csv"
    original.write_stress_to_csv(str(original_csv), False)
    print(f"  runs per direction: {runs(original)}")
    check("the file is written", nxs.exists())

    section("loaded into a fresh model")
    loaded = model_reporting_failures(failures)
    check("Load NXstress succeeds", loaded.load_nxstress(str(nxs)))
    calculate(loaded)
    print(f"  runs per direction: {runs(loaded)}")
    check("every direction comes back from its own run", runs(loaded) == runs(original))
    loaded_csv = out / "strain_stress_loaded.csv"
    loaded.write_stress_to_csv(str(loaded_csv), False)
    worst, differing, columns = largest_difference(original_csv, loaded_csv)
    print(f"  summary CSV: {differing} cells differ, at most {worst:.1e} relative, in {sorted(columns)}")
    check("the summary is reproduced to 1e-6 relative", worst < 1e-6)
    check("only peak heights move", all(column.startswith("Peak_Height") for column in columns))

    section("the saved state names the .nxs file, and restores from it")
    state = out / "strain_stress_state.json"
    loaded.to_json(str(state))
    data = json.loads(state.read_text())
    named = {d: data[f"filenames_{d}"] for d in FILES}
    print(f"  files named per direction: { {d: [Path(f).name for f in names] for d, names in named.items()} }")
    check("each workspace names the .nxs file", all(names == [str(nxs)] for names in named.values()))
    restored = model_reporting_failures(failures)
    restored.from_json(str(state))
    restored_csv = out / "strain_stress_restored.csv"
    restored.write_stress_to_csv(str(restored_csv), False)
    check("the restored summary is identical", restored_csv.read_text() == loaded_csv.read_text())

    check("nothing was reported as a failure", failures == [])
    print(f"  Left for the GUI steps: {nxs}, {state}")


if __name__ == "__main__":
    run(main)
