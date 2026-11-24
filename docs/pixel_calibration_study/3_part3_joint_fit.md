## #1097 study, Part 3: combined Si331AF + Si422 refinement

One shared detector geometry; one λ per run; 2θ₀ = +δ/2 for the Si331AF run and −δ/2 for the Si422 run (mean fixed at 0); `scipy.optimize.least_squares` over the concatenated, unweighted `PixelCalibration.residual` of the two runs. Rings are extracted per run at its own λ and re-extracted at the joint geometry until the step is below half the smaller statistical precision (at most 6 rounds). Errors are `pinv(JᵀJ)` scaled by the joint cost. No parameter ended at a bound. Pair A is the 2 mm slit (5352 + 5351); pair B is the 3×3 gauge volume (5342 + 5346).

|  | A: 5352 + 5351 (slit) | B: 5342 + 5346 (3×3 GV) |
|---|---|---|
| Converged | yes, 5 rounds | no, 6 rounds |
| Last round step (mdeg) | 3.0 | 5.2 |
| δ = 2θ₀(Si331AF) − 2θ₀(Si422) (°) | -0.057 ± 0.043 | -0.173 ± 0.016 |
| λ Si331AF (Å) | 1.71831 ± 0.00028 | 1.71969 ± 0.00012 |
| λ Si422 (Å) | 1.53288 ± 0.00052 | 1.53261 ± 0.00019 |
| Shift_x (mm) | -3.29 ± 0.37 | -3.93 ± 0.14 |
| Shift_y (mm) | -0.06 ± 3.19 | 0.45 ± 1.35 |
| Shift_z (mm) | -82.39 ± 1.07 | -59.34 ± 0.34 |
| Rot_x (°) | -1.88 ± 0.58 | -0.18 ± 0.22 |
| Rot_y (°) | 2.35 ± 0.79 | 1.68 ± 0.29 |
| Rot_z (°) | 0.074 ± 0.045 | 0.173 ± 0.017 |
| RMS joint (mdeg), Si331AF / Si422 | 52.5 / 26.6 | 19.9 / 11.9 |
| RMS single-run (mdeg) | 50.0 / 23.6 | 17.2 / 8.0 |
| Pixel-2θ change vs single-run, Si331AF (mdeg, × precision) | 51.8 (3.0×) | 32.5 (5.5×) |
| Pixel-2θ change vs single-run, Si422 (mdeg, × precision) | 76.4 (9.1×) | 30.3 (11.2×) |
| Jacobian (constrained): smallest relative singular value | 0.060 | 0.062 |
| Jacobian (both 2θ₀ free): smallest / next relative singular value | 2.83e-07 / 0.057 | 2.55e-07 / 0.057 |
| Cost, constrained / free | 0.118941 / 0.118941 | 0.0181698 / 0.0181698 |
| Free fit: 2θ₀_A + 2θ₀_B (°) | 0.0000 | 0.0000 |

### The 2θ₀ constraint is needed
With both 2θ₀ free and unconstrained, the column-scaled Jacobian has one singular value of about 3×10⁻⁷ relative, against 0.057 for the next. The null vector (per degree of Rot_y, natural units: shifts in m, angles in degrees) lies along (2θ₀_A + 2θ₀_B, Rot_y, Shift_x), as expected; every other component is small:

| Parameter | A: 5352 + 5351 | B: 5342 + 5346 |
|---|---|---|
| Shift_x | +0.0158 | +0.0162 |
| Shift_y | +0.0000 | -0.0000 |
| Shift_z | +0.0006 | +0.0005 |
| Rot_x | -0.0014 | -0.0001 |
| Rot_y | +1.0000 | +1.0000 |
| Rot_z | +0.0329 | +0.0032 |
| TTH_0_a | -1.0005 | -1.0000 |
| TTH_0_b | -1.0005 | -1.0000 |
| Lambda_a | +0.0000 | -0.0000 |
| Lambda_b | -0.0000 | -0.0000 |

Freeing the sum adds no fit quality (identical cost) and leaves it at 0.

### Does the second monochromator change either run's pixel 2θ beyond its precision?
Yes, for all four runs: 52 mdeg (3.0×) and 76 mdeg (9.1×) in the slit pair, 33 mdeg (5.5×) and 30 mdeg (11.2×) in the 3×3 pair.

### Caveats
- **One shared geometry fits worse than separate ones.** The joint RMS is 2–4 mdeg above the single-run RMS (pair B, Si422: 8.0 → 11.9 mdeg). A strictly shared geometry does not describe both runs as well as separate ones, so the errors above, which assume the model is right, may be optimistic.
- **Re-extraction noise.** Pair A's step was 12–32 mdeg in rounds 2–4 before settling at 3.0 mdeg in round 5, so its convergence is partly luck. Pair B did not meet the criterion (steps 2–7 mdeg against a threshold of 2.0 mdeg), but δ stayed within ±0.003° over all 6 rounds.
- **δ is consistent with 0 for the slit pair** (1.3σ) and significantly nonzero for the 3×3 pair (−0.173 ± 0.016°, 10.7σ). The slit pair is the least informative, given the slit's artefacts (Part 2).
- **Unweighted.** The noisier run dominates each pair, as the brief specified.
- I did not compare the two single-run calibrations (Si331AF against Si422) directly by pixel 2θ.
