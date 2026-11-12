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
- [ ] Apply a vanadium from another cycle to the 2025 runs, to measure the drift directly.
- [x] Per sub-run rather than whole-run sums (`--per-sub-run`). Needed for arm scans.
- [ ] Per 2θ bin and per η wedge: *n*, *n₀*, Σ Nᵢ, and the inflation ratio
      (Σ Nᵢ + n₀) / Σ Nᵢ.
- [ ] Confirm which wire plane encodes y. Results are labelled by detector axis.

### Phase 1 — Quantify the current treatment ([#1107](https://github.com/neutrons/PyRS/issues/1107))
- [ ] Find out whether Mantid `FitPeaks` scales parameter errors by reduced
      χ², and record the answer as a ground truth.
- [ ] Refit selected sub-runs with σ²_bin = Σ Nᵢ, a study-only variant run
      outside production code. Compare fitted centre, FWHM, and intensity,
      their errors, and χ².
- [ ] Show how the inflation ratio changes with 2θ bin width and η-wedge
      size.
- [ ] Measure how much a stale vanadium (wrong vertical stripe pattern)
      biases narrow-η-wedge intensities (texture).

### Phase 2 — Monte Carlo validation ([#1108](https://github.com/neutrons/PyRS/issues/1108))
- [ ] Synthetic generator: peak profile and background mapped onto the real
      HB2B pixel geometry, then Poisson-sampled at chosen count levels.
      Reduction code is used **unchanged**.
- [ ] Repeat for N realisations: reduce, fit, and collect the pull
      distributions (fitted − true) / σ_reported for centre, FWHM, and
      intensity.
- [ ] Sweep the count level, 2θ bin width, and effective pixel size
      (software 2×2 or 4×4 pixel grouping) to separate pixelation effects from
      counting effects.

### Phase 3 — Candidate treatments (study only) ([#1109](https://github.com/neutrons/PyRS/issues/1109))
- [ ] Bin-level Poisson variance (σ²_bin = Σ Nᵢ, with an explicit rule for
      empty bins).
- [ ] Model-based variance, or a Poisson-likelihood cost function, if
      `FitPeaks` supports it.
- [ ] Score each treatment by pull-distribution width (target: 1) and bias.
- [ ] Write a proposal for any production change. **It needs separate
      approval** before any code in `pyrs/` changes.

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
