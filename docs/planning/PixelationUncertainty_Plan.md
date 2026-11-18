# Detector pixelation and intensity uncertainty — project plan

| | |
|---|---|
| **Board** | [neutrons project #5](https://github.com/orgs/neutrons/projects/5); tracking issue [#1111](https://github.com/neutrons/PyRS/issues/1111) |
| **Status** | Phase 0 in progress: 2019–2023 and 2025 (IPTS-36131) data analysed; findings in [ground truths](../ground_truths.md) (2026-11-12) |
| **Production code changes** | None planned in this phase. Any change to reduction or fitting is proposed separately and needs its own approval. |

## Motivation

Measured HB2B data have a **significant fraction of detector pixels with zero
counts**. The HB2B detector has 1024 × 1024 pixels, each 0.293 mm wide, at
0.985 m (about 0.017° of 2θ per pixel). As a result many pixels fall in the
low-count (Poisson) regime, where the usual assumption σ = √N stops
being a good uncertainty estimate. This project asks:

> How does detector pixelation, and the zero-count pixels it produces, affect
> the uncertainty reported for reduced intensities, fitted peak parameters,
> and ultimately strain and stress?

## What PyRS does today

All references are to `main` at `04102098`. See also the ground-truth entry
[Zero-count pixels are given σ = 1 before 2θ histogramming](../ground_truths.md).

1. **Per-pixel uncertainty.**
   [`histogram_by_numpy`](../../pyrs/core/reduce_hb2b_pyrs.py) sets
   σᵢ = √Nᵢ for every pixel and then **replaces σᵢ = 0 with σᵢ = 1**.
2. **Per-bin uncertainty.** The 2θ histogram sums σᵢ², so a bin holding *n*
   pixels, *n₀* of them with zero counts, is assigned

   σ²_bin = Σ Nᵢ + n₀

   The Poisson variance of the bin is only Σ Nᵢ. The extra *n₀* depends on
   **how many pixels land in the bin**. That count depends on pixel size, the
   2θ bin width, the η range, and the mask, not on the neutrons counted.
3. **Empty bins.** Bins with zero total counts get intensity 0 and σ = 1.
4. **Masks.** Masked pixels are removed *before* histogramming, so they do
   not add to *n₀*. Only real, unmasked zero-count pixels do.
5. **Vanadium.** Pixels whose vanadium count is below 0.9 are dropped.
   Vanadium σ uses the same 0 → 1 substitution.
6. **Naming.** The array named `variances` holds **σ, not σ²**. It is passed
   unchanged to Mantid as `DataE`, which Mantid reads as σ, so the fit weights
   are consistent with the name being wrong.
7. **Fitting.** Mantid `FitPeaks` runs with its default least-squares cost
   function, weighting each bin by 1/σ²_bin
   ([`mantid_fit_peak.py`](../../pyrs/peaks/mantid_fit_peak.py)).

### Expected size of the effect (to be verified)

Take a uniform region where each pixel has Poisson mean λ. Then
E[n₀] = n·e^(−λ), and the reported variance divided by the true variance is

R(λ) = 1 + e^(−λ) / λ

| λ (counts / pixel) | R | σ reported / σ true |
|---|---|---|
| 0.1 | 10.0 | 3.2 |
| 0.5 | 2.2 | 1.5 |
| 1 | 1.37 | 1.17 |
| 5 | 1.001 | 1.00 |

So in sparse regions, such as the background between peaks, error bars may be
inflated several-fold, while bins near peak maxima are barely affected. That
changes the *relative* weights in the peak fit. Depending on how Mantid scales
parameter errors, it can also change the reported peak-parameter uncertainties.
This is a prediction from Poisson statistics alone, and the project must check
it against measured data.

## Goals

1. **Characterise.** Measure the zero-count fraction and the per-pixel count
   distribution on real runs, and check whether they follow Poisson statistics.
2. **Quantify.** Measure how much the current treatment changes bin
   uncertainties, fit weights, and reported peak-parameter uncertainties.
3. **Validate.** Use Monte Carlo to compare the *reported* uncertainty with the
   *actual* scatter of fitted values over repeated simulated measurements.
4. **Recommend.** Evaluate treatments that follow from physical law (Poisson
   statistics), and bring any production change as a separate proposal for
   approval.

## Phases and work items

Each item becomes a draft item on the board. Every phase produces a
short write-up and, where a fact is established, a
[ground-truth](../ground_truths.md) entry.

### Phase 0 — Characterise measured data ([#1106](https://github.com/neutrons/PyRS/issues/1106))
Findings: [HB2B detector ground truths](../ground_truths.md) (2019–2023 test data; 2025 runs
5339, 5340, 5344 from IPTS-36131, analysed per sub-run). In short:
- Vanadium-dead pixels are the detector border, and the production mask already covers them
  (2019 and 2025 masks).
- Zeros in live pixels follow Poisson statistics to within 0.1–1 %, provided arm scans are
  analysed per sub-run. Summing an arm scan at one arm angle produced an apparent 3–7 % deficit.
- Along x, the zero fraction follows only the diffraction pattern.
- Along y, efficiency and zero fraction are striped with the 6.56-row (1.92 mm) wire period,
  unchanged from 2019 to 2025. A vanadium from the same day matches rows to about 2 % (the
  floor); older vanadiums mismatch by up to 12 % (drift).

- [x] Zero fraction per run, and observed vs Poisson-expected zeros per detector column and row
      (study script `run_zero_count_study.py`). Done on whole-run sums.
- [x] Separate dead pixels (zero counts in vanadium) from sparse pixels: the vanadium works as a
      dead-pixel mask.
- [x] Repeat on recent runs with a vanadium taken close in time (IPTS-36131: vanadium 5339,
      runs 5340 and 5344): row mismatch ~2 %, the floor.
- [x] Apply a vanadium from another cycle to the 2025 runs, to measure the drift directly. The
      production vanadium 4501 (2024-07) matches rows to 3.3–3.7 % against 1.2–1.9 % same-day;
      time drift and collimator difference (C508 vs C511) not separated.
- [x] Per sub-run rather than whole-run sums (`--per-sub-run`). Needed for arm scans.
- [x] Per 2θ bin and per η wedge: *n*, *n₀*, Σ Nᵢ, and the inflation ratio
      (Σ Nᵢ + n₀) / Σ Nᵢ. On 5344 (autoreduction settings) it equals the Poisson prediction;
      background σ is reported 1.5–1.8× too large, peaks unaffected, the same in 3° η wedges.
- [x] Confirm which wire plane encodes y. The x plane (horizontal) has about half the pitch of the
      y plane: 0.963 mm against 1.933 mm (3.29 against 6.60 px), the same in 2019, 2024 and 2025. The
      x modulation is weak (1.2 %, against 44–49 % along y).

### Phase 1 — Quantify the current treatment ([#1107](https://github.com/neutrons/PyRS/issues/1107))
- [x] Find out whether Mantid `FitPeaks` scales parameter errors by reduced
      χ², and record the answer as a ground truth. **No**: errors follow σ directly. On synthetic
      peaks the inflated σ leaves fitted values unchanged, but over-reports weak-peak centre and
      width errors by up to ~30 % (strong peaks ~5 %).
- [x] Refit selected sub-runs with σ²_bin = Σ Nᵢ, a study-only variant run
      outside production code. Compare fitted centre, FWHM, and intensity,
      their errors, and χ². All 12 5344 sub-runs move the same way: centre +0.65 mdeg, FWHM
      smaller, intensity +0.5–2.1 %, errors 2–29 % smaller; χ² > 1 (model misfit).
- [x] Show how the inflation ratio changes with 2θ bin width and η-wedge
      size. It doesn't: 360/720/1,440 bins and 3° wedges give the same median inflation
      (2.39–2.40); it is set by counts per pixel.
- [x] Measure how much a stale vanadium (wrong vertical stripe pattern)
      biases narrow-η-wedge intensities (texture). Production vanadium 4501 vs same-day 5339:
      wedge-to-wedge intensity ratios spread 0.6–11.7 % per sub-run (fit error 1–4 %).

- [x] Pixel rebinning test: group the 2D detector horizontally (2θ) and vertically (η) and
      measure the effect on 1) zero-count pixels, 2) uncertainties, 3) strain/stress. Vertical
      grouping (≤ 16 rows) removes the zeros and the σ inflation with strain shifts ≤ 8 µε;
      horizontal grouping aliases the 2θ pattern without pixel splitting (33 µε at 2×, fails at 8×).
- [x] Horizontal grouping with 2θ bins at least as wide as the coarse pixel. Works only then
      (2×1 with ≥ 0.045° bins: 3.6 µε RMS; 4×1 with 0.09° bins: 9 µε), and gains nothing over
      vertical grouping.

### Phase 2 — Monte Carlo validation ([#1108](https://github.com/neutrons/PyRS/issues/1108))
- [x] Synthetic generator: peak profile and background mapped onto the real
      HB2B pixel geometry, then Poisson-sampled at chosen count levels.
      Reduction code is used **unchanged**. (λᵢ = k·Vᵢ·R(2θᵢ)/V_max on 5344 sub-run 7, truth from
      its real fit; optional fresh vanadium per realisation.)
- [x] Repeat for N realisations: reduce, fit, and collect the pull
      distributions (fitted − true) / σ_reported for centre, FWHM, and
      intensity. No strain bias in any option; Poisson σ honest (pull 0.96–1.05); production σ
      pull 0.58–0.61 for weak peaks.
- [x] Sweep the count level and pixel grouping: 0.066 and 0.74 counts per pixel; 1×1, 1×4, 1×8.
      Grouping 1×4/1×8 brings production σ to pull 0.85–0.96. *Bin width and horizontal grouping
      were covered on real data in Phase 1.*
- [x] Model misfit (asymmetric peak, curved background): does production vs Poisson σ then shift
      strain, as on real 3489 (+40 µε)? Background curvature: no. On 3489's own peaks, production
      σ makes 24 % of weak-peak fits fail (sub-run 10; Poisson never). Sub-run 7's real +90 µε lies
      far outside the simulated spread (0 ± 7.5 µε), so it is a real-data misfit, not statistics.
- [x] Find the peak-shape misfit in 3489 sub-run 7 (Poisson χ² 2.1): asymmetry, shoulder, or the
      Si422-vanadium efficiency mismatch. A low-angle shoulder (peak asymmetry), not the vanadium;
      the first reduction also used an uncalibrated file. A simulated two-component truth reproduces
      the real production − Poisson shift (+62.0 ± 0.6 against +62.6 µε). Single-peak fits sit
      +150–460 µε from the main component.
- [x] Peak profile: measured on Ni reflections after a fresh `next` ring calibration (5344, 4971).
      Every peak has a high-angle tail (A = +0.005 to +0.15), no sign change at 90°. The asymmetry
      grows towards the high-2θ side of the detector and falls from bottom to top. A symmetric fit
      sits 20–410 µε from the split-profile peak, varying ~350 µε with position: a
      detector-position response, not axial divergence. (Fe is unusable on the Al holder at 1.53 Å.)
- [x] Vertical incident-slit scan (5344 / 5345 / 5346 = 10 / 5 / 3 mm, all else equal): asymmetry,
      symmetric − split offset, FWHM, η gradient and calibration are all independent of the
      vertical gap. Not yet varied: horizontal gap, sample position, receiving optic.
- [ ] Peak profile, next: an asymmetric profile (split or position-dependent) as a Phase 3
      candidate. Check against a measured detector position response (slit scan,
      `calibration-diagnostics` on `next`) and a holder without Al overlaps, so Fe can be used.
- [ ] Other sub-runs or 2θ, several peaks in the window, η wedges.

### Phase 3 — Candidate treatments (study only) ([#1109](https://github.com/neutrons/PyRS/issues/1109))
- [x] Bin-level Poisson variance (σ²_bin = Σ Nᵢ, with an explicit rule for
      empty bins). Scored below; this is the recommended treatment.
- [x] Model-based variance, or a Poisson-likelihood cost function, if
      `FitPeaks` supports it. **It does not:** `FitPeaks`' `CostFunction` offers only
      "Least squares", "Rwp" and "Unweighted least squares". Not feasible with PyRS's fitting engine.
- [ ] **Candidate: 1 × 4 vertical pixel grouping** (4 rows = 1.17 mm, finer than the 1.92 mm wire
      pitch). On 5344 it cut zeros from 61 % to 18 % and σ_rep/σ_Poisson from 1.53 to 1.04, with
      strain shifts of 7.7 µε RMS (see the pixel-grouping ground truth).
- [ ] **Candidate: 1 × 8 vertical pixel grouping** (8 rows = 2.34 mm, close to the wire pitch). On
      5344: zeros 3.7 %, σ ratio 1.00, strain shifts 8.1 µε RMS.
- [ ] For both candidates, before scoring:
  - [x] low 2θ, where rings curve more (e.g. 3489 down to 59.6°): does the −4 to −6 µε offset grow?
        No. With Poisson σ, vertical grouping shifts ≤ 2 µε mean (≤ 5.6 RMS) at 73–108° on 3489;
        the offset was the production-σ artifact (+40 µε mean on weak-peak 3489).
  - [ ] a strained sample, and weak peaks (where the σ inflation matters most). Weak peaks:
        3489 done (production σ_ε 62 µε against Poisson 30; 1 × 4 → 37, 1 × 8 → 32).
  - [ ] η-wedge (texture) intensities;
  - [ ] pull distributions from Phase 2.
- [ ] **Calibration impact of vertical grouping (1 × 4, 1 × 8).** This covers both calibration methods
      on `next`: the ring calibration (`ring_extraction.extract_ring_points`, `PixelCalibration`, the
      GUI default), which fits geometry to ring points located in strips of the raw image, and
      `FitCalibration`, which fits peaks in η-wedge reductions.
  - [x] **C1: a calibration applies unchanged to grouped pixels.** Already shown: coarse-pixel 2θ and
        η from a given geometry equal the block means of the fine pixels to ≤ 10⁻⁵°.
  - [x] **C2: calibrate on grouped data.** Done for the ring calibration on 3510 and 4971 (results in
        the ground truths). *`FitCalibration` and the IPTS-36131 runs: not needed (user, 2026-11-18),
        because calibration runs on the 1 × 1 grid.* Run both methods at 1 × 1, 1 × 4 and 1 × 8 on calibration
        runs. Candidates: the IPTS-36131 Si422 long calibrations (5337, 5344, 5345), HB2B_4971 (Ni)
        and HB2B_3510. Compare:
    - the refined parameters and their errors: Rot_x, Rot_z, Shift_y (already only weakly
      constrained, ±1 mm), Shift_z and λ, plus the degenerate Shift_x / Rot_y / TTH_0 line, compared
      through the pixel 2θ map;
    - the residual RMS, and the ring-point count and scatter;
    - which strips and rings are found. Grouping changes the rows within each strip.
  - [x] **C3: pixel 2θ maps.** Grouping changes the strain map by ≤ 7.5 µε RMS (4971) and ≤ 49 µε
        (3510, 1 × 4), within each calibration's own uncertainty (32–247 µε). For each grouped calibration, take the RMS and maximum Δ2θ over the
        detector against the native calibration, and convert them to strain at representative 2θ.
  - [x] **C4: cross-application.** *Not needed (user, 2026-11-18): calibration runs on the 1 × 1 grid, and
        C1 shows that applying it to grouped reductions is exact.* Planned as: Reduce and fit a strain reference with each combination: native
        calibration + grouped reduction, grouped calibration + grouped reduction, and grouped
        calibration + native reduction. Compare the strains.
  - [x] **Decision (user, 2026-11-18): detector calibration operates on the native 1 × 1 grid**;
        grouped reductions apply that calibration (exact by C1). The study showed no gain from grouped
        calibration: parameter errors unchanged or 1–3 % larger, shifts within the calibration's own
        uncertainty, and non-converged extraction on grouped 3510. The options considered were:
    - **Default: exclude grouping from detector calibration.** Calibrate at native resolution and
      apply that calibration to grouped reductions. C1 makes this exact. Calibration runs are long,
      so zero-count inflation matters less there. Native rows also keep the full vertical
      ring-curvature information that constrains Shift_y, Rot_x and Rot_z.
    - **Apply grouping to calibration too, only if** C2–C4 show that grouped calibration improves
      parameter errors or residuals without moving the pixel 2θ map by more than the native
      calibration's own propagated 2θ-map uncertainty (proposed criterion).
    - **Either way,** the grouping used must be recorded in the calibration file's and the reduced
      file's provenance, and a mismatch reported. That is a production change and part of the
      proposal below.
- [ ] **Candidate: asymmetric peak profile** (highest impact). A symmetric fit sits 20–410 µε from
      the peak of HB2B's asymmetric Ni peaks, varying ~350 µε with the ring's position on the
      detector. That is far more than σ treatment or grouping (≲ 40 µε); see the peak-shape ground
      truth.
  - [ ] **Options.**
    - (a) **Split pseudo-Voigt** (left / right FWHM, one mixing). It describes the data (χ² 1.7–3.0
      → 1.1–1.5) but is not a Mantid function: it needs a PyRS fit function or a Mantid user
      function.
    - (b) **Mantid `AsymmetricPearsonVII`.** `FitPeaks` accepts it with PyRS's arguments, but it has
      one width and left / right *shapes* only. On a noise-free split-width test peak its centre
      came out 0.044° off, so it must be validated.
    - (c) **A position-dependent profile from the detector's response** (e.g. parallax). This needs
      the response measured (slit scan, `calibration-diagnostics` on `next`).
  - [ ] **Calibration consistency.** The ring calibration locates rings with symmetric pseudo-Voigt
        strip fits, so its geometry absorbs part of the offset (symmetric centres −164 to +191 µε
        from Bragg on 5344). An asymmetric profile for sample fits must also be used, or its effect
        checked, in calibration; otherwise calibration and sample fits measure different points of the
        peak.
  - [x] **Monte Carlo.** Use the real split-profile fits (5344, 4971) as the truth, then fit with the
        symmetric pseudo-Voigt and each candidate. Report bias against the true peak position, pull
        widths and failure rates, at the weak and strong count levels. Done on 5344: the symmetric
        bias grows with A (−20 → −414 µε); the split PV is unbiased (|bias| ≤ 7 µε, pulls 0.77–0.96)
        but 2–2.6× noisier; `AsymmetricPearsonVII` removes only 10–15 % of the bias. *That MC used a split-PV truth and is
        not a fair test of other profiles (user).*
  - [x] **Real-data profile comparison** (no synthetic truth; 5344 / 5345 / 5346): split PV and split
        Pearson VII fit equally well (χ²/dof 1.02); split Pearson VII gives the most position-consistent
        d (57 µε spread, against 72 split PV, 91 Mantid AsymmetricPearsonVII, 100 symmetric). Mantid's
        AsymmetricPearsonVII (one width) is in between. Candidates kept: split PV and split Pearson VII.
  - [x] **Calibration with the same profile:** redo the ring calibration with the asymmetric profile
        in its strip fits, then repeat the position-consistency test. Done with free split Pearson VII
        strip fits on 5344. It made things worse: RMS 0.0066 → 0.0101°, d spread 57 → 95 µε (split
        PVII) and 100 → 171 µε (symmetric). The calibration also moved beyond its errors (λ −360 µε,
        Rot_z 0.26 → 0.04°).
  - [x] **Calibration with a per-ring fixed shape** (two-stage: symmetric first pass, fixed-shape
        split Pearson VII fine tune). RMS 0.0084°; λ −250 µε, Shift_z −1.65 mm. Rot_z stays at 0.27°
        (real). The full-detector d spread is still worse than with the symmetric calibration
        (98 against 57 µε), but that test mixes estimators.
  - [x] **Like-for-like consistency test:** apply each calibration (symmetric, two-stage) to 5345 and
        5346 at strip level (ring points against Bragg) and in narrow η wedges, with the same profile
        as the calibration. Matched symmetric is as consistent as two-stage or better (strip RMS
        61–73 against 72–80 µε; wedges 100 µε against noise-limited 130). Mismatched profiles offset d
        by ~±300 µε. A ~55–100 µε systematic position dependence remains with every profile.
  - [ ] **Remaining position dependence (~55–100 µε):** look for its cause in the geometry model
        or the detector position response (pixel pitch / linearity, the slit scan on `next`), not
        in the peak profile.
  - [ ] **Next candidate: split PV with its asymmetry fixed** as a function of detector position and
        η, determined once from the standards (a new instrument parameter, which needs approval). Test
        whether it keeps the zero bias at close to the symmetric profile's precision.
  - [ ] **Real data.** Fit the same reflection at its different ring positions and η wedges. A better
        profile should shrink the spread of d across positions; today it is ~350 µε. Also report
        χ², and the effect on texture-wedge intensities.
  - [x] **Decision (2026-11-18): the asymmetric profile is not proposed for production.** Matched with
        its own calibration, it is no more consistent than the symmetric approach and costs precision.
        The remaining ~55–100 µε position dependence is not a profile effect; it moves to the item
        above (geometry / detector response). Original criterion: choose the profile that minimises
        the position-dependent spread of d with honest pulls and no extra bias. Any change to PyRS's
        peak functions, to calibration extraction or to defaults is a production change and needs
        its own approval.
- [x] Score each treatment by pull-distribution width (target: 1) and bias. Monte Carlo on the real
      geometry (5344 sub-run 7 truth, weak 0.066 counts per pixel, 600 realisations, fixed /
      resampled vanadium), plus 3489 sub-run 10 for fit failures:

      | Treatment | centre pull width | intensity pull width | strain scatter | bias | failed fits (3489 s10) |
      |---|---|---|---|---|---|
      | production σ, 1 × 1 (today) | 0.58 / 0.61 | 0.41 / 0.44 | 20.1 / 21.1 µε | ≤ 1.8 µε | 24 % |
      | **Poisson σ, 1 × 1** | **0.96 / 1.01** | **0.99 / 1.05** | **19.1 / 19.9 µε** | ≤ 1 µε | **0 %** |
      | production σ, 1 × 4 grouping | 0.85 / 0.89 | 0.67 / 0.71 | 19.3 / 20.2 µε | ≤ 1 µε | 8 % |
      | production σ, 1 × 8 grouping | 0.92 / 0.96 | 0.80 / 0.84 | 19.1 / 19.9 µε | ≤ 1 µε | — |
      | Poisson σ with 1 × 4 / 1 × 8 | same as Poisson σ 1 × 1 | | | | |

      At strong counts (0.74 per pixel) every treatment is close to honest (pull width 0.87–0.95).
- [ ] **Proposal for a production change (needs approval).** See "Proposal" below.
- [x] Write a proposal for any production change. **It needs separate
      approval** before any code in `pyrs/` changes. Written below.

#### Proposal: Poisson bin variance in reduction (awaiting approval)

**Change.** In `PyHB2BReduction.histogram_by_numpy` (`pyrs/core/reduce_hb2b_pyrs.py`), stop replacing a
zero-count pixel's σ = 0 by 1 (`pixel_var_array[pixel_var_array == 0.0] = 1.0`). A 2θ bin then gets
σ²_bin = Σ Nᵢ, the Poisson variance. **Empty-bin rule:** unchanged. A bin with Σ Nᵢ = 0 keeps
intensity 0 and σ = 1 (the existing `zero_count_mask` branch), so Mantid never sees σ = 0. The
vanadium σ, normalisation and pixel selection are unchanged. Calibration stays on the native 1 × 1
grid (user decision).

**Expected effect on results** (ground truths; current code, autoreduction settings):
- **Reported uncertainties become honest.** `FitPeaks` does not rescale errors by χ², so σ passes
  straight into them: Monte Carlo pull widths go from 0.58–0.61 to 0.96–1.01.
  - background-bin σ: smaller by a factor 1.5–1.8 (5344);
  - strong-peak strain errors: ~5 % smaller;
  - weak-peak strain errors: ~2.3× smaller (3489 Ni: 85 → 37 µε median);
  - weak-peak intensity errors: ~2.4× smaller.
- **Fitted values** are unbiased in both versions when the peak model fits (Monte Carlo ≤ 1–2 µε).
  On real data, where the pseudo-Voigt does not fully describe the peaks, they move by −5.7 µε mean
  (5344, strong peaks) and +27.6 µε mean (3489 Ni, weak). Precision improves ~5 %.
- **Fewer failed fits on weak peaks:** 24 % of production-σ fits failed in the 3489 sub-run 10 Monte
  Carlo, against none with Poisson σ.
- **Reduced χ²** rises from ≪ 1 (0.19 on weak data) to ≈ 1. It becomes a usable goodness-of-fit
  measure.
- **Stored files:** every newly reduced pattern's σ array changes. Intensities are unchanged.

**Not proposed:**
- vertical pixel grouping. It is no longer needed with Poisson σ (it gives the same pulls), though
  it is a fallback if this proposal is declined (1 × 4 / 1 × 8 safe; 1 × 16 not, with Rot_z ≈ 0.26°);
- the asymmetric peak profile (no gain when matched with its own calibration);
- a Poisson-likelihood cost (not available in `FitPeaks`).

**Tests that would accompany it:** a unit test that a bin of n pixels, n₀ of them empty, gets
σ² = Σ Nᵢ, not Σ Nᵢ + n₀; that empty bins keep σ = 1; and a regression check on a reference reduction
(σ changes, intensity identical).

### Phase 4 — Propagate to strain and stress ([#1110](https://github.com/neutrons/PyRS/issues/1110))
- [ ] Carry the Phase 2 and 3 results through d-spacing, strain, and stress,
      and report how much the stress uncertainty changes on a real sample.

## Where the study code lives

Study scripts are **not** part of the PyRS repository: the repository holds only production code
and tests of production code. The study code (`run_zero_count_study.py`, `zero_counts.py` and their
tests) is kept outside the repository and run in the PyRS pixi environment.
Findings go in [ground truths](../ground_truths.md); result files (figures, JSON, profiles) are kept
with the study code, outside the repository. Nothing under
`pyrs/` changes during Phases 0–3. Environments are run through pixi
(`pixi run …`).

## Open questions

1. Which recent runs and vanadium floods should Phase 0 use?
2. ~~Should zero-count pixels that are also zero in vanadium be added to the
   production mask?~~ Answered on 2019 data: `HB2B_Mask_12-18-19.xml` already covers all of
   them. Re-check against the current production mask.
3. Which quantity matters most for users: peak centre (strain), FWHM, or
   integrated intensity (texture)?
4. How often is a vanadium flood measured? The vertical stripe pattern drifts.
