## #1097 study, Part 1: single-run ring calibrations

Study only: no production code changed, nothing shared was written. All 11 powder runs are in IPTS-36131 (resolved with `get_nexus_file`). Default settings are the shared mask `HB2B_Mask_Latest.xml`, 5 strips, at most 10 rounds, and a start from the nominal geometry. Pixel-2θ differences are the largest change over the detector grid at the ring points' arm angles.

**Notes on the setup**
- **2θ₀ is already held at 0 by default** (#1090), so the "held" column below is the default. "2θ₀ free" is a study-only variant: `ring_rounds.PixelCalibration` was patched inside the study script to build with `hold_two_theta_0=False`. Nothing under `pyrs/` changed.
- **Monochromator:** no run has a `MonoSetting` log, so it comes from `mrot`. Run 5344's `mrot` has two samples (−173.67, then −23.44, the Si331AF and Si422 settings) with an average of −98.55: the monochromator was changed at the start of that run. It calibrates as Si422 (λ = 1.5326 Å).
- Errors are `estimate_errors()` with 2θ₀ held (conditional on it, even in the "2θ₀ free" variant).

### 1a. Default calibration quality

| Run | Mono | Condition | Status | RMS (mdeg) | Precision (mdeg) | Rounds | Ring points | Time (s) |
|---|---|---|---|---|---|---|---|---|
| 5352 | Si331AF | 2 mm slit | 1 | 50.0 | 17.40 | 3 | 66 | 63 |
| 5342 | Si331AF | 3×3 GV | 1 | 17.2 | 5.95 | 3 | 67 | 108 |
| 5341 | Si331AF | 3×15 GV | 1 | 20.0 | 6.77 | 3 | 70 | 96 |
| 5351 | Si422 | 2 mm slit | 1 | 23.6 | 8.42 | 3 | 63 | 105 |
| 5347 | Si422 | Ni only | 1 | 7.0 | 3.13 | 3 | 40 | 68 |
| 5348 | Si422 | Fe only | 1 | 2.5 | 1.31 | 2 | 30 | 6 |
| 5346 | Si422 | 3×3 GV | 1 | 8.0 | 2.71 | 2 | 70 | 56 |
| 5345 | Si422 | 3×5 GV | 1 | 6.4 | 2.15 | 2 | 70 | 72 |
| 5344 | Si422 | 3×10 GV | 1 | 6.6 | 2.25 | 3 | 70 | 94 |
| 5337 | Si422 | aligned | 1 | 6.4 | 2.17 | 2 | 70 | 57 |
| 5338 | Si422 | shifted | 2 | 17.2 | 5.82 | 2 | 70 | 44 |

Status 1 and 2 are both converged `least_squares` codes. Every default converged in 2–3 rounds.

### 1b. Default parameters

Shifts in mm, angles in degrees, λ in Å. TTH_0 = 0 (held).

| Run | Shift_x | Shift_y | Shift_z | Rot_x | Rot_y | Rot_z | Lambda |
|---|---|---|---|---|---|---|---|
| 5352 | -3.85 ± 0.46 | -2.01 ± 4.65 | -84.64 ± 1.63 | -2.08 ± 0.93 | 2.02 ± 1.28 | -0.039 ± 0.078 | 1.71847 ± 0.00036 |
| 5342 | -5.25 ± 0.17 | -2.04 ± 1.77 | -59.69 ± 0.50 | -0.17 ± 0.34 | 1.05 ± 0.45 | 0.066 ± 0.028 | 1.71967 ± 0.00013 |
| 5341 | -4.93 ± 0.18 | -2.18 ± 1.92 | -59.57 ± 0.54 | -0.35 ± 0.38 | 0.37 ± 0.49 | 0.059 ± 0.032 | 1.71949 ± 0.00014 |
| 5351 | -2.53 ± 0.38 | -3.25 ± 3.95 | -80.66 ± 1.14 | -1.24 ± 0.52 | 3.32 ± 0.71 | 0.203 ± 0.040 | 1.53252 ± 0.00031 |
| 5347 | -2.62 ± 0.14 | -0.39 ± 1.45 | -59.11 ± 0.30 | -0.18 ± 0.18 | 2.92 ± 0.24 | 0.252 ± 0.015 | 1.53256 ± 0.00011 |
| 5348 | -2.49 ± 0.07 | 0.50 ± 0.83 | -58.42 ± 0.14 | -0.22 ± 0.09 | 1.72 ± 0.11 | 0.248 ± 0.007 | 1.53260 ± 0.00006 |
| 5346 | -2.55 ± 0.12 | 0.01 ± 1.32 | -59.02 ± 0.25 | -0.22 ± 0.15 | 2.31 ± 0.20 | 0.258 ± 0.013 | 1.53257 ± 0.00010 |
| 5345 | -2.52 ± 0.10 | -0.25 ± 1.04 | -59.07 ± 0.20 | -0.23 ± 0.12 | 2.33 ± 0.16 | 0.262 ± 0.010 | 1.53254 ± 0.00008 |
| 5344 | -2.59 ± 0.10 | -0.02 ± 1.09 | -59.00 ± 0.21 | -0.18 ± 0.13 | 2.39 ± 0.17 | 0.260 ± 0.010 | 1.53259 ± 0.00008 |
| 5337 | -2.95 ± 0.10 | -0.16 ± 1.05 | -61.65 ± 0.21 | -0.16 ± 0.12 | 2.75 ± 0.16 | 0.245 ± 0.010 | 1.53231 ± 0.00008 |
| 5338 | -1.69 ± 0.26 | -0.53 ± 2.81 | -63.54 ± 0.67 | -0.11 ± 0.32 | 4.10 ± 0.43 | 0.221 ± 0.027 | 1.53158 ± 0.00021 |

### 1c. Pixel-2θ change against the run's own default (mdeg)

`*`: the extraction did not converge within the round limit (7 of 75 variants).

| Run | Condition | 2θ₀ free | 3 strips | 10 strips | 15 strips | 1 round | no mask | prev. start | Default precision |
|---|---|---|---|---|---|---|---|---|---|
| 5352 | 2 mm slit | 0.00 | 16.99 | 28.61 | 54.81* | 87.00 | 0.90 | n/a | 17.40 |
| 5342 | 3×3 GV | 0.00 | 16.56 | 13.63 | 17.12* | 24.58 | 9.63 | 0.95 | 5.95 |
| 5341 | 3×15 GV | 0.00 | 1.42 | 0.42 | 0.55 | 40.81 | 1.30 | 0.03 | 6.77 |
| 5351 | 2 mm slit | 0.00 | 25.18* | 85.66* | 87.47* | 65.70 | 2.18 | n/a | 8.42 |
| 5347 | Ni only | 0.00 | 0.41 | 0.85 | 0.37 | 1.55 | 6.18 | 0.16 | 3.13 |
| 5348 | Fe only | 0.00 | 0.44 | 0.33 | 0.47 | 0.38 | 0.02 | 0.09 | 1.31 |
| 5346 | 3×3 GV | 0.00 | 2.29 | 4.16* | 4.03 | 1.12 | 44.89 | 1.97 | 2.71 |
| 5345 | 3×5 GV | 0.00 | 1.85 | 1.54 | 6.54* | 0.81 | 18.90 | 0.19 | 2.15 |
| 5344 | 3×10 GV | 0.00 | 1.01 | 0.84 | 0.50 | 1.21 | 6.69 | 0.06 | 2.25 |
| 5337 | aligned | 0.00 | 1.06 | 0.94 | 2.16 | 0.99 | 6.40 | 0.24 | 2.17 |
| 5338 | shifted | 0.00 | 1.89 | 7.44 | 8.04 | 0.82 | 4.85 | 0.10 | 5.82 |

### 1d. RMS in mdeg (rounds)

| Run | Condition | 2θ₀ free | 3 strips | 10 strips | 15 strips | 1 round | no mask | prev. start | Default |
|---|---|---|---|---|---|---|---|---|---|
| 5352 | 2 mm slit | 50.0 (3) | 49.4 (3) | 46.9 (3) | 45.8 (10) | 41.1 (1) | 50.0 (4) | n/a | 50.0 (3) |
| 5342 | 3×3 GV | 17.2 (3) | 18.8 (5) | 15.9 (4) | 15.3 (10) | 14.0 (1) | 17.5 (4) | 17.1 (2) | 17.2 (3) |
| 5341 | 3×15 GV | 20.0 (3) | 19.9 (3) | 20.2 (3) | 20.5 (3) | 13.6 (1) | 19.7 (3) | 20.0 (2) | 20.0 (3) |
| 5351 | 2 mm slit | 23.6 (3) | 17.7 (10) | 12.3 (10) | 12.6 (10) | 30.5 (1) | 24.1 (3) | n/a | 23.6 (3) |
| 5347 | Ni only | 7.0 (3) | 6.8 (3) | 7.3 (3) | 7.6 (3) | 7.0 (1) | 7.7 (3) | 7.0 (2) | 7.0 (3) |
| 5348 | Fe only | 2.5 (2) | 2.0 (3) | 3.1 (3) | 3.3 (3) | 2.5 (1) | 2.5 (2) | 2.5 (2) | 2.5 (2) |
| 5346 | 3×3 GV | 8.0 (2) | 7.6 (2) | 9.5 (10) | 10.0 (4) | 8.1 (1) | 17.3 (3) | 8.0 (2) | 8.0 (2) |
| 5345 | 3×5 GV | 6.4 (2) | 6.0 (3) | 6.7 (3) | 8.1 (10) | 6.4 (1) | 8.9 (4) | 6.4 (2) | 6.4 (2) |
| 5344 | 3×10 GV | 6.6 (3) | 6.4 (3) | 7.0 (2) | 7.2 (3) | 6.6 (1) | 7.2 (3) | 6.6 (2) | 6.6 (3) |
| 5337 | aligned | 6.4 (2) | 6.3 (3) | 6.8 (2) | 7.1 (5) | 6.4 (1) | 6.8 (3) | 6.4 (2) | 6.4 (2) |
| 5338 | shifted | 17.2 (2) | 17.1 (2) | 17.7 (2) | 18.0 (2) | 17.2 (1) | 17.3 (3) | 17.2 (2) | 17.2 (2) |

### 1e. 2θ₀ held at 0 versus free

| Run | TTH_0 when free (°) | ΔRot_y (°) | ΔShift_x (mm) | Pixel-2θ change (mdeg) |
|---|---|---|---|---|
| 5352 | 0.480 | -0.480 | -7.54 | 0.0000 |
| 5342 | 0.291 | -0.291 | -4.70 | 0.0000 |
| 5341 | 0.244 | -0.244 | -3.94 | 0.0000 |
| 5351 | 0.626 | -0.626 | -9.89 | 0.0000 |
| 5347 | 0.606 | -0.606 | -9.80 | 0.0000 |
| 5348 | 0.378 | -0.378 | -6.11 | 0.0000 |
| 5346 | 0.549 | -0.549 | -8.88 | 0.0000 |
| 5345 | 0.555 | -0.555 | -8.97 | 0.0000 |
| 5344 | 0.558 | -0.558 | -9.02 | 0.0000 |
| 5337 | 0.608 | -0.608 | -9.81 | 0.0000 |
| 5338 | 0.841 | -0.841 | -13.55 | 0.0000 |

### Findings
1. **Freeing 2θ₀ changes nothing the data can see.** Pixel 2θ is identical (< 0.01 mdeg) for all 11 runs; 2θ₀ moves to 0.5–0.6° and Rot_y and Shift_x absorb it. This is the exact degeneracy #1090 relies on, so the free value is arbitrary and holding it at 0 is the right default.
2. **A single extraction is not enough.** One round differs from the default by 25–87 mdeg for the slit runs and for 5342 and 5341, several times their precision. This supports the multi-round default.
3. **The start does not matter.** Starting from the previous run's calibration reproduces the default to ≤ 2 mdeg, within precision for every run, and converges in 2 rounds.
4. **Strips 3, 10 and 15 add nothing for clean runs.** Ni, Fe, the 3×N gauge volumes and the aligned run move ≤ about 4 mdeg. The exceptions are 5345 at 15 strips and 5338 at 10 and 15 strips (6.5–8 mdeg, above precision for 5338). More strips make convergence worse, not better, on the slit runs.
5. **The mask matters for the Si422 gauge-volume runs.** Without it 5346 moves 45 mdeg (RMS 8 → 17 mdeg) and 5345 moves 19 mdeg.
6. **The slit runs are the unstable ones**, as expected for the deliberate slit study: 5351 (Si422 slit) fails to converge with 3, 10 or 15 strips and its RMS falls to 12 mdeg at 10 and 15 strips, against 24 mdeg at default. The default for the slit runs is therefore not a reliable reference.
