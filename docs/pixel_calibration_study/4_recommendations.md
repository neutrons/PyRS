## #1097 study: recommendations

Based on Parts 1–3 above (11 powder runs, IPTS-36131).

**Default settings: keep them.**
- **2θ₀ held at 0** for single-run calibrations (#1090): freeing it changes no pixel's 2θ (< 0.01 mdeg).
- **Multi-round extraction (up to 10 rounds)**: a single round is off by 25–87 mdeg on the harder runs.
- **5 strips**: 3, 10 and 15 give no benefit on clean runs and converge worse on the slit runs.
- **Shared mask**: removing it moves the Si422 gauge-volume runs by 19–45 mdeg.
- **Nominal start**: starting from a previous calibration reproduces the result to ≤ 2 mdeg.

**Measurement conditions**
- **Gauge volume does not matter** within these sizes (≤ 5 mdeg for Si422 3×3 / 3×5 / 3×10; 21 mdeg, marginal, for Si331AF 3×3 against 3×15).
- **Position matters**: shifted against aligned moves pixel 2θ by 114 mdeg (19× precision). Align the calibration sample.
- **The slit run is not a valid reference** (Shift_z moves by 20–25 mm): consistent with the deliberate slit-artefact study.
- **Ni and Fe together** gives more ring points (70 against 30–40) than either alone; the single-material and combined runs differ by 13–21 mdeg in pixel 2θ, so use the same sample as the production calibration uses.

**Joint Si331AF + Si422**
- The 2θ₀ constraint (#1090) is necessary: without it one singular value is ~3×10⁻⁷.
- Joint and single-run calibrations differ by 30–76 mdeg (3–11× precision), and the joint RMS is 2–4 mdeg higher, so a shared geometry is not a free approximation. Whether to use the joint fit as the production calibration is a decision for you, not one these data settle.

**Possible issues, not fixed**
- The brief said production still refines 2θ₀. It no longer does (#1090 holds it by default).
- No bug found in PyRS.
