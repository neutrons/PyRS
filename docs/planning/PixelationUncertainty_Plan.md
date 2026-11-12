# Detector pixelation and intensity uncertainty — project plan

| | |
|---|---|
| **Board** | [neutrons project #5](https://github.com/orgs/neutrons/projects/5) |
| **Status** | Scoping (2026-11-12) |
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

### Phase 0 — Characterise measured data
- [ ] Pick representative runs (high and low count rate; powder standard and
      a real sample). *Input needed from the instrument scientist.*
- [ ] Per sub-run: fraction of unmasked zero-count pixels; histogram of pixel
      counts; compare with the Poisson expectation from the local mean.
- [ ] Separate dead or unmasked-bad pixels (zero counts in vanadium too) from
      pixels that are simply sparse.
- [ ] Per 2θ bin: *n*, *n₀*, Σ Nᵢ, and the inflation ratio
      (Σ Nᵢ + n₀) / Σ Nᵢ.

### Phase 1 — Quantify the current treatment
- [ ] Find out whether Mantid `FitPeaks` scales parameter errors by reduced
      χ², and record the answer as a ground truth.
- [ ] Refit selected sub-runs with σ²_bin = Σ Nᵢ, a study-only variant run
      outside production code. Compare fitted centre, FWHM, and intensity,
      their errors, and χ².
- [ ] Show how the inflation ratio changes with 2θ bin width and η-wedge
      size.

### Phase 2 — Monte Carlo validation
- [ ] Synthetic generator: peak profile and background mapped onto the real
      HB2B pixel geometry, then Poisson-sampled at chosen count levels.
      Reduction code is used **unchanged**.
- [ ] Repeat for N realisations: reduce, fit, and collect the pull
      distributions (fitted − true) / σ_reported for centre, FWHM, and
      intensity.
- [ ] Sweep the count level, 2θ bin width, and effective pixel size
      (software 2×2 or 4×4 pixel grouping) to separate pixelation effects from
      counting effects.

### Phase 3 — Candidate treatments (study only)
- [ ] Bin-level Poisson variance (σ²_bin = Σ Nᵢ, with an explicit rule for
      empty bins).
- [ ] Model-based variance, or a Poisson-likelihood cost function, if
      `FitPeaks` supports it.
- [ ] Score each treatment by pull-distribution width (target: 1) and bias.
- [ ] Write a proposal for any production change. **It needs separate
      approval** before any code in `pyrs/` changes.

### Phase 4 — Propagate to strain and stress
- [ ] Carry the Phase 2 and 3 results through d-spacing, strain, and stress,
      and report how much the stress uncertainty changes on a real sample.

## Where the study code lives

Study scripts and notebooks go in `scripts/pixelation_study/`, with any
reusable helpers covered by unit tests under `tests/unit/`. Nothing under
`pyrs/` changes during Phases 0–3. Environments are run through pixi
(`pixi run …`).

## Open questions

1. Which runs should Phase 0 use?
2. Should zero-count pixels that are also zero in vanadium be treated as
   masked, i.e. should they be added to the production mask?
3. Which quantity matters most for users: peak centre (strain), FWHM, or
   integrated intensity (texture)?
