"""Strain/stress viewer defects fixed in phase 3, each shown fixed.

All of these predate phase 3. Shows that:

1. every row of the summary stress CSV has as many cells as its header. A row
   past the end of a direction's data used to carry up to four extra blank
   cells, so every later value sat under the wrong column name;
2. a saved state naming a missing file is reported, and restores nothing: a
   state restores whole or not at all. It used to raise `AssertionError` out of
   the Load action;
3. a saved state whose peak one direction lacks is not a crash. It used to
   raise `KeyError`;
4. a file with no fitted peaks loads as e22 (it used to raise `IndexError`), and
   in a list of files it is never paired with another file's peaks;
5. a d0 equal to the last one entered is applied again after a reload rebuilds
   the stress. It used to be skipped, so the stress was built without it.

Run from the repository root:

    pixi run python tests/scripts/cis_tests/NXstress/viewer_defects.py [output-dir]
"""

import json

from _fixtures import TEST_DATA, check, output_dir, run, section

import numpy as np

from pyrs.core.stress_facade import StressFacade
from pyrs.core.summary_generator_stress import SummaryGeneratorStress
from pyrs.dataobjects.fields import StrainField, StressField
from pyrs.interface.strainstressviewer.model import Model


def model_with_failures() -> tuple[Model, list]:
    model = Model()
    failures: list = []
    model.failureMsg.connect(lambda *args: failures.append(args[0]))
    return model, failures


def write_state(path, **overrides) -> str:
    """A saved state: in-plane stress from HB2B_1320 in 11 and 22, with `overrides` applied."""
    state = {
        "stress_case": "in-plane-stress",
        "filenames_11": [str(TEST_DATA / "HB2B_1320.h5")],
        "filenames_22": [str(TEST_DATA / "HB2B_1320.h5")],
        "filenames_33": [],
        "peak_tag": "peak0",
        "youngs_modulus": 200,
        "poisson_ratio": 0.3,
        "d0": None,
    }
    state.update(overrides)
    path.write_text(json.dumps(state))
    return str(path)


def csv_rows(out) -> None:
    section("summary CSV rows match the header")
    strain11 = StrainField(str(TEST_DATA / "HB2B_1331.h5"), peak_tag="peak0")
    strain22 = StrainField(str(TEST_DATA / "HB2B_1332.h5"), peak_tag="peak0")
    # `None` is how an in-plane case is built; `strain33`'s annotation omits it (DD-9 in docs/deferred_defects.md).
    stress = StressField(strain11, strain22, None, 200, 0.3, stress_type="in-plane-strain")  # type: ignore[arg-type]
    path = out / "summary_1331_1332.csv"
    SummaryGeneratorStress(str(path), StressFacade(stress)).write_summary_csv()
    lines = [line for line in path.read_text().splitlines() if not line.startswith("#")]
    header, counts = len(lines[0].split(",")), {len(line.split(",")) for line in lines[1:]}
    print(f"  {len(lines) - 1} rows; header {header} cells; row cell counts {sorted(counts)}")
    check("every row has the header's cell count", counts == {header})


def missing_file(out) -> None:
    section("a saved state naming a missing file")
    model, failures = model_with_failures()
    model.from_json(write_state(out / "state_missing.json", filenames_22=[str(out / "missing.h5")]))
    print(f"  reported: {failures}")
    print(f"  e11 holds {len(model.e11)} workspace(s), e22 {len(model.e22)}; stress: {model.stress}")
    check("reported once, with no exception", len(failures) == 1)
    check("nothing of the state is kept", model.e11 == model.e22 == model.e33 == [] and model.stress is None)


def peak_one_direction_lacks(out) -> None:
    section("a saved state whose peak one direction lacks")
    model, failures = model_with_failures()
    state = write_state(out / "state_peak1.json", filenames_11=[str(TEST_DATA / "HB2B_1328.h5")], peak_tag="peak1")
    model.from_json(state)
    print(f"  HB2B_1328 has peak1, HB2B_1320 does not; window says: {model.validate_selection('22')!r}")
    check("no exception, and stress waits", model.stress is None and failures == [])


def file_without_peaks(out) -> None:
    section("a file with no fitted peaks")
    model, failures = model_with_failures()
    model.e11 = str(TEST_DATA / "HB2B_1320.h5")
    model.e22 = str(TEST_DATA / "HB2B_1423.h5")
    print(f"  HB2B_1423 as e22: {len(model.e22)} workspace(s); window says: {model.validate_selection('22')!r}")
    check("it loads as e22, with no IndexError", len(model.e22) == 1)

    files = [str(TEST_DATA / "HB2B_1423.h5"), str(TEST_DATA / "HB2B_1320.h5")]
    model, failures = model_with_failures()
    model.from_json(write_state(out / "state_unfitted.json", filenames_11=files))
    print(f"  e11 = [HB2B_1423, HB2B_1320]: peaks per file {[sorted(peaks) for peaks in model.e11_peaks]}")
    check("each file keeps its own peaks", model.e11_peaks[0] == {} and "peak0" in model.e11_peaks[1])
    check("the window names the gap", model.validate_selection("11") == "Peak peak0 is not in e11")


def d0_after_reload() -> None:
    section("an unchanged d0 after a reload")
    model, _ = model_with_failures()
    model.e11 = str(TEST_DATA / "HB2B_1320.h5")
    model.e22 = str(TEST_DATA / "HB2B_1320.h5")
    model.selectedPeak = "peak0"
    d0 = (1.05, 0.0001)
    model.calculate_stress("in-plane-stress", 200, 0.3, d0)
    model.e22 = str(TEST_DATA / "HB2B_1320.h5")  # a reload drops the stress
    model.calculate_stress("in-plane-stress", 200, 0.3, d0)
    values = model.stress_facade.d_reference.values
    print(f"  d0 entered {d0[0]}; the rebuilt stress uses {sorted(set(np.round(values, 6).tolist()))}")
    check("the rebuilt stress uses the d0 entered", np.allclose(values, d0[0]))


def main() -> None:
    out = output_dir()
    csv_rows(out)
    missing_file(out)
    peak_one_direction_lacks(out)
    file_without_peaks(out)
    d0_after_reload()


if __name__ == "__main__":
    run(main)
