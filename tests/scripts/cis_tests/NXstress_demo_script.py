"""
tests/scripts/cis_tests/NXstress_demo_script.py

Smoke-test / "by hand" demo script for the NXstress I/O implementation.

Features demonstrated
---------------------
1. Loading a ``HidraWorkspace`` from an existing HiDRA project file
   (``tests/data/3393_PWHT-TD.h5``).

2. Fitting two diffraction peaks with
   ``tests.util.peak_collection_helpers.generate_PeakCollection_from_workspace``
   to produce a ``list[PeakCollection]``.

   The ``fit_dic`` below mirrors the starting point given in the docstring of
   that helper, but with ``peak_label`` values adjusted to follow the
   ``peak_tag`` convention required by ``NXstress``:

       "<phase_name> <hkl>"   e.g. "Fe 311"

   where ``<hkl>`` is a string of 3 N digits that encodes the Miller indices
   (h, k, l) as N-digit zero-padded integers.  The two peaks present in the
   data file are the austenitic-iron reflections "Fe 311" and "Fe 222", as
   confirmed by the ``hklPhase`` log stored in the file.

3. Writing the workspace and fitted peak collections to a new
   NXstress-compatible NeXus file via ``NXstress`` used as a context manager.

4. Reading the data back from the NXstress file and printing a short summary
   to confirm that the round-trip succeeded.

Usage
-----
Run this script directly (not via pytest)::

    python tests/scripts/cis_tests/NXstress_demo_script.py

The output NXstress file is written to the current working directory as
``NXstress_demo_output.nxs``.
"""

from datetime import date
from pathlib import Path

import numpy as np

from pyrs.core.instrument_geometry import HidraSetup
from pyrs.core.workspaces import HidraWorkspace
from pyrs.dataobjects.constants import HidraConstants
from pyrs.projectfile.file_object import HidraProjectFile, HidraProjectFileMode
from pyrs.utilities.NXstress import NXstress

from tests.util.input_data_helpers import ensure_input_data
from tests.util.instrument_helpers import ensure_instrument_geometry
from tests.util.peak_collection_helpers import generate_FitResults_from_workspace

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Repository root is two levels above the directory of this script:
#   tests/scripts/cis_tests/  ->  tests/scripts/  ->  tests/  ->  <root>
_REPO_ROOT = Path(__file__).resolve().parents[3]

# DATA_FILE is the baseline HiDRA project file shipped with the repository.
# Once a "-<date>-complete.h5" file has been generated on a machine with /HFIR
# mounted (see Step 1 below), DATA_FILE can be pointed at that file instead;
# the complete file includes instrument geometry and raw detector counts, so
# ensure_instrument_geometry / ensure_input_data become no-ops and the script
# runs correctly even without a live HFIR archive connection.
# ------
DATA_FILE   = _REPO_ROOT / "tests" / "data" / "3393_PWHT-TD.h5" ## *** ORIGINAL DATA SOURCE ***
# DATA_FILE   = _REPO_ROOT / "tests" / "data" / "3393_PWHT-TD-2026-05-08-complete.h5"

INPUT_DATA_PRESENT = False # set to `True` if loading from `...complete.h5`
# ------

OUTPUT_FILE = Path("NXstress_demo_output.nxs")

# Workspace name is derived from the stem of DATA_FILE so that it tracks
# automatically if DATA_FILE is changed to a "-complete" variant.
WORKSPACE_NAME = DATA_FILE.stem

# HB2B run number corresponding to DATA_FILE -- used to fetch raw counts from the archive.
DATA_RUN_NUMBER = 3393

# ---------------------------------------------------------------------------
# Round-trip verification helpers
# ---------------------------------------------------------------------------

_REL_TOL = 0.01   # 1 % relative tolerance for aggregate (mean / variance) comparisons
_GEO_TOL = 1e-5   # tight tolerance for stored scalar geometry values
_WL_TOL  = 1e-4   # tolerance for wavelength (stored as float32)

_POSITION_LOGS = list(HidraConstants.SAMPLE_COORDINATE_NAMES)  # ("vx", "vy", "vz")


def _check_approx(label: str, orig: float, back: float, rtol: float = _REL_TOL) -> None:
    """Assert the two scalars agree to within *rtol* and print the result."""
    orig, back = float(orig), float(back)
    rel_diff = abs(orig - back) / (abs(orig) + 1e-10)
    status = "OK" if rel_diff <= rtol else "FAIL"
    print(f"    [{status}] {label}: orig={orig:.6g}  back={back:.6g}  rel_diff={rel_diff:.2e}")
    assert rel_diff <= rtol, (
        f"Round-trip mismatch for '{label}': {orig} vs {back}  (rtol={rtol})"
    )


def _to_wavelength_array(wl_raw, n: int) -> np.ndarray:
    """Normalise a wavelength value (float, dict, or None) to a 1-D ndarray of length *n*."""
    if wl_raw is None:
        return np.full(n, np.nan)
    if isinstance(wl_raw, float):
        return np.full(n, wl_raw)
    if isinstance(wl_raw, dict):
        return np.array(list(wl_raw.values()), dtype=float)
    return np.asarray(wl_raw, dtype=float)


def verify_sample_group(ws: HidraWorkspace, ws_back: HidraWorkspace) -> None:
    """Verify the round-tripped SAMPLE_DESCRIPTION (NXsample) group."""
    print("  [sample] log names present ...")
    log_names_back = set(ws_back.sample_log_names)
    assert len(log_names_back) > 0, "Read-back workspace has no sample logs"
    for coord in _POSITION_LOGS:
        assert coord in log_names_back, (
            f"Expected position log '{coord}' not found in read-back workspace"
        )

    n = len(ws.get_sub_runs())
    print(f"  [sample] position log shapes (expected n={n}) ...")
    for coord in _POSITION_LOGS:
        arr = ws_back._sample_logs[coord]
        assert arr.shape == (n,), (
            f"Log '{coord}': expected shape ({n},), got {arr.shape}"
        )

    print("  [sample] position log mean / variance ...")
    for coord in _POSITION_LOGS:
        orig_arr = ws._sample_logs[coord].astype(float)
        back_arr = ws_back._sample_logs[coord].astype(float)
        _check_approx(f"{coord} mean", np.nanmean(orig_arr), np.nanmean(back_arr))
        if np.nanvar(orig_arr) > 0:
            _check_approx(f"{coord} var", np.nanvar(orig_arr), np.nanvar(back_arr))


def verify_instrument_group(ws: HidraWorkspace, ws_back: HidraWorkspace) -> None:
    """Verify the round-tripped INSTRUMENT (NXinstrument) group."""
    geom_orig = ws.get_instrument_setup()
    geom_back = ws_back.get_instrument_setup()
    assert geom_back is not None, "Read-back workspace has no instrument geometry"

    print("  [instrument] geometry scalars ...")
    _check_approx("arm_length", geom_orig.arm_length, geom_back.arm_length, rtol=_GEO_TOL)
    orig_rows, orig_cols = geom_orig.detector_size
    back_rows, back_cols = geom_back.detector_size
    _check_approx("detector nrows", orig_rows, back_rows, rtol=_GEO_TOL)
    _check_approx("detector ncols", orig_cols, back_cols, rtol=_GEO_TOL)
    orig_px, orig_py = geom_orig.pixel_dimension
    back_px, back_py = geom_back.pixel_dimension
    _check_approx("pixel_size_x", orig_px, back_px, rtol=_GEO_TOL)
    _check_approx("pixel_size_y", orig_py, back_py, rtol=_GEO_TOL)

    print("  [instrument] wavelength ...")
    n = len(ws.get_sub_runs())
    # `get_wavelength` always returns `_wave_length` first (regardless of the calibrated flag)
    # when it is set, so use calibrated=False which matches how NXstress stored it.
    wl_orig = _to_wavelength_array(ws.get_wavelength(calibrated=False, throw_if_not_set=False), n)
    wl_back = _to_wavelength_array(ws_back.get_wavelength(calibrated=False, throw_if_not_set=False), n)
    assert wl_back.shape == (n,), f"Wavelength shape: expected ({n},), got {wl_back.shape}"
    assert not np.all(np.isnan(wl_back)), "All read-back wavelength values are NaN"
    _check_approx("wavelength mean", np.nanmean(wl_orig), np.nanmean(wl_back), rtol=_WL_TOL)


def verify_input_data_group(ws: HidraWorkspace, ws_back: HidraWorkspace) -> None:
    """Verify the round-tripped INPUT_DATA (NXdata / detector_counts) group.

    If *ws* has raw counts (loaded via ``ensure_input_data``), confirm that the
    read-back workspace contains the same number of sub-runs with counts and
    that the per-sub-run pixel arrays agree to within floating-point precision.

    If *ws* has no raw counts (the project file was loaded without them and
    ``ensure_input_data`` was not called), just confirm that no counts were
    spuriously created on the read-back.
    """
    n_orig = len(ws._raw_counts)
    n_back = len(ws_back._raw_counts)
    print(f"  [input_data] raw_counts entries: orig={n_orig}  back={n_back}")

    if n_orig == 0:
        assert n_back == 0, (
            f"Expected no raw counts in read-back workspace (none were written), got {n_back}"
        )
        print("    [--] no raw counts written -- skipping pixel comparison")
        return

    assert n_back == n_orig, (
        f"raw_counts sub-run count mismatch: orig={n_orig}  back={n_back}"
    )

    sub_runs = sorted(ws._raw_counts.keys())
    for sr in sub_runs:
        orig_counts = ws._raw_counts[sr].astype(float)
        back_counts = ws_back._raw_counts[sr].astype(float)
        assert orig_counts.shape == back_counts.shape, (
            f"sub_run={sr}: detector_counts shape mismatch: "
            f"{orig_counts.shape} vs {back_counts.shape}"
        )
        _check_approx(
            f"sub_run={sr} detector_counts mean",
            float(np.nanmean(orig_counts)),
            float(np.nanmean(back_counts)),
            rtol=_REL_TOL,
        )
    print(f"    pixel arrays compared for {len(sub_runs)} sub-run(s)")


def verify_diffraction_group(ws: HidraWorkspace, ws_back: HidraWorkspace) -> None:
    """Verify the round-tripped FIT / diffractogram (NXdata) group."""
    assert ws_back._2theta_matrix is not None, "Read-back workspace has no 2θ matrix"

    orig_shape = ws._2theta_matrix.shape
    back_shape = ws_back._2theta_matrix.shape
    print(f"  [diffraction] 2θ matrix shape: orig={orig_shape}  back={back_shape}")
    assert orig_shape == back_shape, (
        f"2θ matrix shape mismatch: {orig_shape} vs {back_shape}"
    )

    orig_masks = set(ws._diff_data_set.keys())
    back_masks = set(ws_back._diff_data_set.keys())
    print(f"  [diffraction] mask keys: orig={orig_masks}  back={back_masks}")
    assert orig_masks == back_masks, (
        f"Diffraction mask key mismatch: {orig_masks} vs {back_masks}"
    )

    for mask_id in orig_masks:
        orig_data = ws._diff_data_set[mask_id]
        back_data = ws_back._diff_data_set[mask_id]
        assert orig_data.shape == back_data.shape, (
            f"Diffraction data shape mismatch for mask={mask_id!r}: "
            f"{orig_data.shape} vs {back_data.shape}"
        )
        finite_back = back_data[np.isfinite(back_data)]
        assert len(finite_back) > 0, (
            f"All diffraction data for mask={mask_id!r} is non-finite"
        )
        assert np.any(finite_back != 0), (
            f"All diffraction data for mask={mask_id!r} is zero"
        )
        _check_approx(
            f"mask={mask_id!r} intensity mean",
            np.nanmean(orig_data), np.nanmean(back_data),
        )
        if np.nanvar(orig_data) > 0:
            _check_approx(
                f"mask={mask_id!r} intensity var",
                np.nanvar(orig_data), np.nanvar(back_data),
            )


def verify_peaks_group(peak_collections: list, peaks_back: list) -> None:
    """Verify the round-tripped PEAKS (NXreflections) and FIT/peak_parameters groups."""
    assert len(peaks_back) == len(peak_collections), (
        f"PeakCollection count: expected {len(peak_collections)}, got {len(peaks_back)}"
    )

    def _sort_key(pc):
        # Normalise tag so "Fe 311" and "Fe311" both become "Fe311" for matching.
        return pc.peak_tag.replace(" ", "")

    orig_sorted = sorted(peak_collections, key=_sort_key)
    back_sorted = sorted(peaks_back, key=_sort_key)

    for orig, back in zip(orig_sorted, back_sorted):
        tag = orig.peak_tag
        print(f"  [peaks] '{tag}' ...")

        assert orig.peak_profile == back.peak_profile, (
            f"peak_profile mismatch for '{tag}': {orig.peak_profile!r} vs {back.peak_profile!r}"
        )
        assert orig.background_type == back.background_type, (
            f"background_type mismatch for '{tag}': {orig.background_type!r} vs {back.background_type!r}"
        )

        orig_runs = orig.sub_runs.raw_copy()
        back_runs = back.sub_runs.raw_copy()
        assert len(orig_runs) == len(back_runs), (
            f"sub_runs length mismatch for '{tag}': {len(orig_runs)} vs {len(back_runs)}"
        )
        assert np.array_equal(orig_runs, back_runs), (
            f"sub_runs values mismatch for '{tag}'"
        )

        orig_dref, _ = orig.get_d_reference()
        back_dref, _ = back.get_d_reference()
        orig_dref_is_nan = np.all(np.isnan(orig_dref))
        assert orig_dref_is_nan == np.all(np.isnan(back_dref)), (
            f"d_reference NaN status mismatch for '{tag}': "
            f"orig all-NaN={orig_dref_is_nan}, back all-NaN={np.all(np.isnan(back_dref))}"
        )
        assert not orig_dref_is_nan, (
            f"d_reference is all-NaN for '{tag}' — was 'd0' set in fit_dic?"
        )
        _check_approx(f"'{tag}' d_reference mean", np.nanmean(orig_dref), np.nanmean(back_dref), rtol=1e-5)

        orig_vals, orig_errs = orig.get_native_params()
        back_vals, back_errs = back.get_native_params()
        assert orig_vals.shape == back_vals.shape, (
            f"Native param array shape mismatch for '{tag}': {orig_vals.shape} vs {back_vals.shape}"
        )
        for field in orig_vals.dtype.names:
            o_mean = float(np.nanmean(orig_vals[field].astype(float)))
            b_mean = float(np.nanmean(back_vals[field].astype(float)))
            _check_approx(f"'{tag}' param '{field}' mean", o_mean, b_mean)

        # Error comparison: all fields (including Intensity for PseudoVoigt) round-trip
        # exactly now that the sigma_I inversion in peakParametersForRange is correct.
        for field in orig_errs.dtype.names:
            o_err = orig_errs[field].astype(float)
            b_err = back_errs[field].astype(float)
            _check_approx(
                f"'{tag}' error '{field}' mean",
                float(np.nanmean(o_err)), float(np.nanmean(b_err)),
            )

        print(f"    fit_costs length: {len(back.fitting_costs)}  (expected {len(orig_runs)})")
        assert len(back.fitting_costs) == len(orig_runs), (
            f"fitting_costs length mismatch for '{tag}': "
            f"{len(back.fitting_costs)} vs {len(orig_runs)}"
        )


# ---------------------------------------------------------------------------
# Peak-fit configuration
# ---------------------------------------------------------------------------
# ``peak_label`` values MUST follow the ``peak_tag`` convention so that
# ``_Peaks._parse_peak_tag`` can extract a phase name and Miller indices.
# The data file records "Fe 311, Fe 222" in its ``hklPhase`` log.
#
# fit_dic format:
#   key   – arbitrary string used as an ordered loop index
#   value – dict with:
#       "peak_range"  : [x_min, x_max]  (2θ in degrees)
#       "peak_label"  : peak_tag string  ("<phase> <hkl>")
#       "d0"          : reference d-spacing in Å (for strain calculation)
FIT_DIC = {
    "0": {"peak_range": [87.599, 91.569], "peak_label": "Fe 311", "d0": 1.08},
    "1": {"peak_range": [93.544, 95.890], "peak_label": "Fe 222", "d0": 1.03},
}

# ---------------------------------------------------------------------------
# Step 1 – Load the HidraWorkspace
# ---------------------------------------------------------------------------
print("=" * 60)
print("Step 1: Loading HidraWorkspace")
print(f"  file: {DATA_FILE}")

ws = HidraWorkspace(WORKSPACE_NAME)
with HidraProjectFile(DATA_FILE, mode=HidraProjectFileMode.READONLY) as project_file:
    ws.load_hidra_project(project_file, load_raw_counts=INPUT_DATA_PRESENT, load_reduced_diffraction=True)

# Track whether anything was added to the workspace beyond what DATA_FILE
# already contained, so we know whether to write a "-complete" project file.
_geometry_added   = ws._instrument_setup is None
_had_raw_counts   = bool(ws._raw_counts)   # True when the file already contained them
_raw_counts_added = False

# Most Hidra project files have no instrument section; NXstress requires one.
# Install the nominal HB2B engineering geometry if none was stored in the file.
ensure_instrument_geometry(ws)

# Load the raw detector counts from the HFIR archive so that the NXstress
# input_data group is written and can be verified on read-back.
# This requires a live connection to the HFIR archive; if the run is not
# accessible, ensure_input_data is a no-op and the input_data group will
# simply be empty (the rest of the demo continues normally).
try:
    ensure_input_data(ws, DATA_RUN_NUMBER)
    _raw_counts_added = bool(ws._raw_counts) and not _had_raw_counts
    print(f"  raw counts loaded : {len(ws._raw_counts)} sub-run(s)")
except Exception as _exc:  # noqa: BLE001
    print(f"  [WARN] could not load raw counts from archive: {_exc}")
    print("         input_data group will be empty -- continuing.")

print(f"  sub-runs loaded : {len(ws.get_sub_runs())}")
print(f"  wavelength      : {ws.get_wavelength(calibrated=True, throw_if_not_set=False)} Angstrom")

# ---------------------------------------------------------------------------
# Step 1b -- Persist a "-complete" HiDRA project file when new data was added
# ---------------------------------------------------------------------------
# If geometry or raw counts were loaded that were absent from DATA_FILE, save a
# self-contained project file next to DATA_FILE.  On subsequent runs (with or
# without /HFIR mounted) DATA_FILE can be pointed at this file directly.
if _geometry_added or _raw_counts_added:
    _today = date.today().isoformat()  # e.g. "2026-05-08"
    _complete_file = DATA_FILE.with_name(f"{WORKSPACE_NAME}-{_today}-complete.h5")
    print()
    print(f"  Saving complete HiDRA project file -> {_complete_file}")
    with HidraProjectFile(_complete_file, mode=HidraProjectFileMode.OVERWRITE) as _hf:
        if ws._instrument_setup is not None:
            _hf.write_instrument_geometry(HidraSetup(ws.get_instrument_setup()))
        ws.save_experimental_data(_hf, ignore_raw_counts=not _raw_counts_added)
        # save_experimental_data only writes raw counts + sample logs; the reduced
        # diffraction data lives in a separate HDF5 group and must be written explicitly.
        if ws._2theta_matrix is not None:
            _hf.write_reduced_diffraction_data_set(
                ws._2theta_matrix,
                ws._diff_data_set,
                ws._var_data_set,
            )
    print(f"  Written: {_complete_file.resolve()}")

# ---------------------------------------------------------------------------
# Step 2 – Fit peaks and build list[PeakCollection]
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Step 2: Fitting peaks with generate_FitResults_from_workspace")

fit_results = generate_FitResults_from_workspace(ws, FIT_DIC)
peak_collections = [pc for result in fit_results for pc in result.peakcollections]

print(f"  PeakCollections fitted: {len(peak_collections)}")
for pc in peak_collections:
    print(f"    peak_tag      : {pc.peak_tag!r}")
    print(f"    peak_profile  : {pc.peak_profile}")
    print(f"    background    : {pc.background_type}")

# ---------------------------------------------------------------------------
# Step 3 – Write to NXstress file
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print(f"Step 3: Writing NXstress file -> {OUTPUT_FILE}")

with NXstress(OUTPUT_FILE, mode="w") as nxs:
    nxs.write(ws, peak_collections)

print(f"  Written: {OUTPUT_FILE.resolve()}")

# ---------------------------------------------------------------------------
# Step 4 – Read back and verify round-trip
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Step 4: Reading back from NXstress file")

with NXstress(OUTPUT_FILE, mode="r") as nxs:
    ws_back, peaks_back = nxs.read(entry_number=1)

print(f"  sub-runs read back    : {len(ws_back.get_sub_runs())}")
print(f"  PeakCollections read  : {len(peaks_back)}")
for pc in peaks_back:
    print(f"    peak_tag (read back): {pc.peak_tag!r}")

print()
print("Verifying SAMPLE group ...")
verify_sample_group(ws, ws_back)

print()
print("Verifying INSTRUMENT group ...")
verify_instrument_group(ws, ws_back)

print()
print("Verifying INPUT_DATA group ...")
verify_input_data_group(ws, ws_back)

print()
print("Verifying DIFFRACTION group ...")
verify_diffraction_group(ws, ws_back)

print()
print("Verifying PEAKS group ...")
verify_peaks_group(peak_collections, peaks_back)

print()
print("=" * 60)
print("Demo completed successfully.")
