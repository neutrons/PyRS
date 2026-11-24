# Ring-calibration study (#1097)

Which measurement conditions and calibration settings change the ring-based calibration, and what a joint
Si331AF + Si422 refinement does. Eleven powder runs of IPTS-36131 (run 5339, 5340 and 5343, the vanadium floods and the
empty run, were not calibrated). A study only: it changes no production code. Refs #1097, #1090. The findings are
summarised in [ground_truths.md](../ground_truths.md) ("Calibration study: what the measurement conditions and the
settings change").

| File | Contents |
|---|---|
| [1_part1_single_run_calibrations.md](1_part1_single_run_calibrations.md) | Part 1: default calibrations and seven variants per run |
| [2_part2_conditions.md](2_part2_conditions.md) | Part 2: slit, gauge volume, Ni/Fe and position, compared by pixel 2θ |
| [3_part3_joint_fit.md](3_part3_joint_fit.md) | Part 3: the joint Si331AF + Si422 fits and the 2θ₀ constraint |
| [4_recommendations.md](4_recommendations.md) | Recommendations |
| `part1_all_calibrations.csv` | Part 1, every calibration: status, RMS, precision, rounds, ring points, time, the 8 values and errors |
| `part2_conditions.csv` | Part 2 comparison table |
| `part3_5352_5351.json`, `part3_5342_5346.json` | Part 3, per pair: rounds, parameters and errors, Jacobian singular values |

The calibrations were made with `pyrs.calibration.pixel_pos_calibration.calibrate_rings` on the current `next`, the shared
mask `HB2B_Mask_Latest.xml` and 5 strips unless a variant says otherwise. The "2θ₀ free" variant patched
`ring_rounds.PixelCalibration` inside the study script. The study scripts are not in the repository.
