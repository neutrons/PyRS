## #1097 study, Part 2: what the measurement conditions do

Default calibrations (1a) compared within one monochromator. Pixel 2θ is the largest change over the detector grid at arm angles 76, 90 and 103° (L2 = nominal arm length). **Ratio** is that change divided by the compared condition's own statistical precision, and by the quadrature sum of both calibrations' precisions. About 2 or more is significant. **Carrier** is the parameter whose swap alone, from reference to condition, moves pixel 2θ most (mdeg in brackets). One-at-a-time swaps can overstate a parameter in the degenerate Shift_x / Rot_y pair.

| Comparison (reference → condition) | Δ pixel 2θ (mdeg) | Precision (mdeg) | Ratio (own) | Ratio (combined) | Carrier | Δλ (Å) | RMS (mdeg) |
|---|---|---|---|---|---|---|---|
| Si331AF: 2mm slit -> 3x3 GV | 386.0 | 5.95 | 64.9 | 21.0 | Shift_z (234.8) | +0.001203 (+3.2σ) | 50.0 → 17.2 |
| Si331AF: 2mm slit -> 3x15 GV | 375.6 | 6.77 | 55.5 | 20.1 | Shift_z (235.8) | +0.001019 (+2.7σ) | 50.0 → 20.0 |
| Si331AF: 3x3 GV -> 3x15 GV | 21.3 | 6.77 | 3.1 | 2.4 | Shift_x (19.9) | -0.000184 (-1.0σ) | 17.2 → 20.0 |
| Si422: 2mm slit -> 3x3 GV | 242.8 | 2.71 | 89.5 | 27.4 | Shift_z (200.1) | +0.000055 (+0.2σ) | 23.6 → 8.0 |
| Si422: 2mm slit -> 3x5 GV | 240.7 | 2.15 | 112.0 | 27.7 | Shift_z (199.7) | +0.000025 (+0.1σ) | 23.6 → 6.4 |
| Si422: 2mm slit -> 3x10 GV | 245.2 | 2.25 | 109.2 | 28.1 | Shift_z (200.3) | +0.000075 (+0.2σ) | 23.6 → 6.6 |
| Si422: 3x3 GV -> 3x5 GV | 3.9 | 2.15 | 1.8 | 1.1 | Shift_x (1.9) | -0.000030 (-0.2σ) | 8.0 → 6.4 |
| Si422: 3x3 GV -> 3x10 GV | 3.0 | 2.25 | 1.3 | 0.9 | Shift_x (2.4) | +0.000020 (+0.2σ) | 8.0 → 6.6 |
| Si422: 3x5 GV -> 3x10 GV | 5.4 | 2.25 | 2.4 | 1.7 | Shift_x (4.2) | +0.000050 (+0.4σ) | 6.4 → 6.6 |
| Si422: Ni only -> Fe only | 20.6 | 1.31 | 15.7 | 6.1 | Rot_y (33.3) | +0.000043 (+0.3σ) | 7.0 → 2.5 |
| Si422: Ni only -> both (3x3 GV) | 14.5 | 2.71 | 5.4 | 3.5 | Rot_y (17.3) | +0.000016 (+0.1σ) | 7.0 → 8.0 |
| Si422: Fe only -> both (3x3 GV) | 13.3 | 2.71 | 4.9 | 4.4 | Rot_y (16.0) | -0.000027 (-0.2σ) | 2.5 → 8.0 |
| Si422: aligned -> shifted | 113.5 | 5.82 | 19.5 | 18.3 | Shift_x (78.2) | -0.000733 (-3.3σ) | 6.4 → 17.2 |

"Both" means the 3×3 gauge-volume run (5346), taken as the combined Ni + Fe sample: please confirm.

### Findings
1. **Gauge-volume size barely matters.** Within Si422 the 3×3, 3×5 and 3×10 runs agree to 3–5 mdeg (ratios 1.3–2.4). Within Si331AF, 3×3 against 3×15 differs by 21 mdeg (ratio 3.1), carried by Shift_x: marginal.
2. **The slit is a different geometry.** Against any gauge volume it moves Shift_z from about −59 mm to −81 mm (Si422) and −85 mm (Si331AF): 240 mdeg and 380 mdeg, 55–110× the precision. This is the signature of the deliberate slit-artefact study, so a slit run is not usable as a calibration reference. For Si331AF the wavelength also moves by about +3σ, where Shift_z does.
3. **Position matters.** Aligned → shifted changes pixel 2θ by 114 mdeg (ratio 19.5): Shift_x −2.95 → −1.69 mm, Shift_z −61.7 → −63.5 mm, Rot_y 2.75 → 4.10° and λ −0.0007 Å (−3.3σ). The calibration sample must be aligned.
4. **Ni, Fe and both differ at the 13–21 mdeg level** (ratios 5–16), carried by Rot_y with Shift_x partly cancelling. The Fe-only run has only 30 ring points and Ni-only 40, against 70 for the combined sample; Fe-only's very small RMS (2.5 mdeg) and precision (1.3 mdeg) rest on few points, so I would not read much into it.
