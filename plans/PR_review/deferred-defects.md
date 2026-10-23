# Deferred defects — PyRS PR review

Defects found during the PR review of NXstress subspecs 04, 04b and 04c that are
**real and reproduced**, but deliberately not fixed in that PR — because the fix
belongs to code the PR does not own, or to a change that deserves its own review.

This file is not a backlog of ideas and not a place for open questions. Every
entry is a defect that was measured, with the measurement attached, so that
picking one up later does not mean re-deriving it.

**Each entry records where it was found** (`subspec <list>: part <batch>`,
matching the review's commit headers) and why it was deferred. Entries are
append-only; when one is fixed, say so in place rather than deleting it.

---

## DD-1 — `validate_config` cannot be enforced after a runtime override

**Found during:** subspec 04b, 04c: part 7
**Owner:** `neutrons_standard.Config`, not PyRS
**Status:** open, not patched here

### What happens

`pyrs/utilities/config.py` calls `validate_config()` once, at import, so a
malformed configuration fails at the first import rather than at the first I/O
call site. That is the right posture and it works — for the configuration that
exists at import.

It cannot cover an override applied afterwards. `Config.loadEnv(<path>)` installs
a new configuration at any later point, and nothing re-validates:

```console
>>> Config.loadEnv("bad.yml")            # nxstress: {use_production_names: "yes"}
>>> Config["nxstress.use_production_names"]
'yes'                                     # a truthy string, never checked
```

`validate_config()` is a public function and can be called again by hand, but
nothing in `Config` invites it: there is no validation hook, no post-refresh
callback, and `_Config.validate` is a documented no-op. So the guarantee the
module advertises — that a malformed config is rejected before it can be acted
on — holds only for the import-time path.

### Why it is deferred

The gap is in the external package. Closing it properly means
`neutrons_standard.Config` growing a validation hook that `refresh`/`loadEnv`
invoke, which PyRS cannot add from the outside, and which is the kind of change
that wants its own review with the package's owners.

The PyRS-side workarounds are all worse than the gap:

- calling `validate_config()` from every consumer — the scattering this module
  exists to prevent;
- wrapping `Config` in a PyRS proxy that re-validates on access — a second
  `Config`, and the module docstring is explicit that there must be exactly one
  way to reach it;
- re-validating inside `_discriminator.field_names()` and friends — validates
  some keys, at some call sites, which reads as coverage and is not.

### Why it is tolerable for now

The runtime-override path is **test-only** by decision (subspec 04b, part 2: the
`env` OS variable is the documented production mechanism, and it is read at
import). Production never calls `loadEnv`. What is missing is that nothing says
so, and nothing stops a future caller.

### If picked up

Ask `neutrons_standard` for a validation hook on `refresh`. Failing that, the
narrow PyRS fix is for the `config_override` test fixture to call
`validate_config()` after merging, so at least every overridden configuration a
test installs is one production would have accepted — which is where a malformed
override would otherwise first mislead someone.

### Update, 2026-10-16 — partially mitigated in tests; the gap itself is unchanged

Re-verified before acting, rather than recalled. The reproduction still holds
exactly as recorded above:

```console
before override: False
after loadEnv  : 'yes'   (type str, truthy=True)
validate_config(): RuntimeError -- so it WOULD catch it, if called
```

`_Config.validate` is still `pass`, `refresh` still mentions no validation, and
nothing re-validates after `loadEnv`. **The deferral stands.**

**Done — the narrow fix this entry proposed.** `config_override`
(`tests/unit/pyrs/utilities/conftest.py`) now calls `validate_config()` after
each merge, so a test cannot install a configuration production would have
rejected. 104 call sites; a malformed override is refused by default and
accepted only under the opt-out below.

> **Correction, same day.** This paragraph originally claimed the guarantee
> "was checked by measurement rather than assumed". That was true of a
> standalone script and **false of the suite**: an independent test review
> deleted the hook and all 675 tests stayed green, so the guarantee was real
> but unconstrained. The hook had also made `test_config.py:354`'s own
> `validate_config()` unreachable, leaving the 8 tests in
> `TestValidateConfigRejectsMalformedValues` asserting against the *fixture's*
> exception rather than the subject's — passing, and no longer testing what
> their names said. Both closed:
> `test_config.py::TestConfigOverrideRefusesMalformedConfig` now pins the hook
> and the dead line is gone; deleting the hook fails 11 tests. Recorded in full
> in `04b-comments.md`'s `RE-REVIEW RECOMMENDED` section as D10. **Measuring a
> guarantee once is not the same as constraining it**, and only the second
> survives the next refactor.

**Still to do — the actual fix.** Ask `neutrons_standard` for a validation hook
that `refresh`/`loadEnv` invoke, so a runtime override is checked by the package
that performs it. PyRS cannot add this from outside, and the three PyRS-side
workarounds rejected above are still worse than the gap. Recorded here so the
request is not re-derived; **not yet raised with the package's owners.**

### What the fix uncovered: three consumer checks that production cannot reach

Adding validation broke 3 of the 104 uses, and the reason is worth more than the
fix was.

All three install a malformed configuration in order to test that a *consumer*
rejects it — `_discriminator.field_names()` on a blank and on a repeated
discriminator name, `_Instrument._instrument_names()` on a blank instrument
name. Measured against `ConfigSchema` directly:

```console
blank discriminator   : rejected by schema -> consumer check NOT reachable in production
repeated discriminator: rejected by schema -> consumer check NOT reachable in production
blank instrument_name : rejected by schema -> consumer check NOT reachable in production
```

`NonBlankText` and `_distinct` already refuse all three at import, and the `env`
OS-variable override is merged *before* `validate_config()` runs — so no
production path can deliver such a configuration to a consumer. **The three
tests passed only because nothing validated the override**, which is the very
gap this entry is about.

This is the review's recurring finding inverted: not an assertion that cannot
fail, but a test that can only pass by installing a state production forbids.
One of the three already said so in its own docstring — *"reachable only by
bypassing `validate_config`"* — so it was known at one site and not generalised.

**Resolved by opt-out, not by deletion.** `override(..., validate=False)`, used
at exactly those three sites, each carrying a comment saying why. Whether that
defence-in-depth should remain is a layering question — the schema validates
once at a boundary, the consumers re-check at point of use — and settling it
means touching `_discriminator.py` and `_instrument.py`, which is not something
to decide inside a PR about append. Recorded here as the open half.

---

## DD-2 — The sub-run ordering requirement is arbitrary, and should be dropped

**Found during:** subspec 04b, 04c: phase-2 PR assembly (post-review probe D7, 2026-10-20)
**Owner:** `pyrs.dataobjects.sample_logs.SubRuns`, and the sites below that treat
a sub-run number as an array position

> **Recommendation: drop the requirement that a workspace's sub-runs be in
> increasing order.** Strongly recommended. What PyRS needs is that sub-run
> numbers are **unique within a workspace**, and **disjoint between
> workspaces** that share an NXstress entry. Both are already checked, and
> neither depends on order. The increasing order is enforced in exactly one
> place, protects exactly one lookup that can be rewritten without it, and is
> otherwise never used. Meanwhile it makes a correct NXstress round trip
> impossible for workspaces that share a discriminator value and are not
> already in increasing order, and it hides a separate class of real bugs (§4) behind the
> appearance of a sorted, contiguous axis.

### What happens

An entry whose workspaces share one discriminator value is read back as **one**
workspace with the rows in on-disk order, which is the inputs' order. So was a
merged write (`nxstress.merge_workspaces`), until that key was removed (Decisions
Log row 45 in `plans/NXstress-prod/README.md`). Writing `[4, 5, 6]` then
`[1, 2, 3]` produced a file that `read` refused, from `SubRuns.set`. This was
measured with the merged write:

```console
D7 write merged [4,5,6]+[1,2,3] -> [4, 5, 6, 1, 2, 3]
D7 read RAISES RuntimeError subruns are not sorted in increasing order
D7 increasing order (control) -> (1, [1, 2, 3, 4, 5, 6])
```

The same happened on an append of lower scan points under a discriminator value
the entry already held.

### What was done in this PR — refuse, don't re-sort

`NXstress._refuseUnreadableScanPointOrder` now refuses, with
`NotImplementedError`, any write or append that would read back a workspace
whose scan points are not strictly increasing. It groups the rows the way `read`
does (`_workspaceSelections`), so interleaved inputs separated by a
discriminator are unaffected. Nothing is written, and an appending instance
stays usable, as for Case B.

The rows are deliberately **not** re-sorted. The on-disk order is the inputs'
order, and sorting it would break the round trip: there would then be no way to
re-assign the rows to the individual workspaces on read. The refusal is the
interim. Removing it is the first step once this entry is fixed, and the strict
xfail `test_multi_workspace.py::TestReadBackScanPointOrder::test_shared_value_in_decreasing_order_round_trips`
will say so.

### The analysis: what actually depends on the order

From a survey of the whole codebase (2026-10-20). The key sites were re-read
against the code, and each defect was reproduced on a synthetic workspace.

**1. Where the order is actually used — one place.**

- `SubRuns.get_indices` ([sample_logs.py:229](../../pyrs/dataobjects/sample_logs.py#L229))
  finds the positions of several sub-runs with `np.searchsorted`, a binary
  search, which is only correct on a sorted axis.
  - **Defect:** on an unsorted axis it returns wrong positions without raising.
    `SubRuns.append` skips the ordering check, so this is reachable today:
    `[4,5,6]` appended with `[1,2,3]`, then `get_indices([5, 2])` → `[6, 4]`.
  - **Defect, even when sorted:** it checks only that a query's first and last
    values exist. `SubRuns([1,2,4]).get_indices([1,3,4])` → `[0, 2, 2]`: the
    missing 3 maps to the position of 4.
  - Single-value lookups use `np.nonzero(value == x)` and need only uniqueness.
  - It is reached through `SampleLogs.__getitem__((name, subruns))`,
    `get_pointlist(subruns)` (including NXstress `_Peaks._sample_positions`),
    `HidraWorkspace.get_sample_log_values`,
    `save_experimental_data(sub_runs=subset)` and the peak-fitting plot.

**2. Where the order is checked, but not used.**

- `SubRuns.set` ([sample_logs.py:167](../../pyrs/dataobjects/sample_logs.py#L167)):
  `value[:-1] < value[1:]`. Every `SubRuns` is built through it: a project-file
  read, `HidraWorkspace._set_subruns`, `PeakCollection.set_peak_fitting_values`
  and NXstress `_Sample.sampleLogsFromNexus`, which is where the D7 read failed.
  What it guarantees downstream is that values do not repeat.
- `_Peaks.peakCollectionRanges` ([_peaks.py:524](../../pyrs/utilities/NXstress/_peaks.py#L524))
  checks that scan points increase within each peak-collection block. It exists
  only because the `PeakCollection` built from the block would fail `SubRuns.set`
  anyway.
- `HidraWorkspace._set_subruns` ([workspaces.py:1121](../../pyrs/core/workspaces.py#L1121))
  sorts only so that its input passes the check.
- The checks that enforce the **real** invariant are already order-free:
  - `NXstress._validateMultiWorkspace` (a dict of seen scan points);
  - `_rejectScanPointCollisions` and `_classifyCompoundKeys` (set
    intersections);
  - `_Peaks.validateNoDuplicatePeaks`;
  - the reader's partition check in `_workspaceSelections` (`np.isin`, plus the
    overlap and unclaimed checks).

**3. Places that need only one consistent order.** These pair a sub-run array
with data rows by position, and work in any order that all the arrays share:
- reduced-data rows and sample-log columns;
- `PeakCollection` parameter rows and its exclude list;
- the Mantid FitPeaks spectrum index;
- NXstress's `input_data` axis check, diffractogram rows, wavelength-dict order,
  and the block order shared by PEAKS, `peak_parameters` and
  `background_parameters` (`IndexedPeaks.sort_key` never sorts by scan point);
- `SubRuns.__eq__` and `matching_subruns` compare element-wise, which requires
  the same order, not sorted order.

One exception would surface once the requirement is dropped:
`HidraWorkspace.set_wavelength` ([workspaces.py:1297](../../pyrs/core/workspaces.py#L1297))
compares `sorted(dict_keys)` to the sub-runs element-wise, so it would refuse a
valid dictionary for unsorted sub-runs. It fails loudly. Comparing as sets fixes
it. There is no test, because it is unreachable while `SubRuns` is sorted.

**4. Places that assume sub-runs are exactly 1..N.** These treat the sub-run
number as a position. They break on a gap, or a first sub-run other than 1,
**even when sorted**, so they are defects whether or not the requirement is
dropped. Sortedness never protected them; a 1..N test workspace hides them.

| Location | What goes wrong | Test |
|---|---|---|
| [workspaces.py:1258-1262](../../pyrs/core/workspaces.py#L1258) `save_reduced_diffraction_data` | Selects rows as `sub_run - 1`. With no raw counts loaded it always takes this branch, so saving any workspace whose sub-runs do not start at 1 raises `IndexError`. | `test_workspaces.py::TestSubRunNumbersAreNotPositions` |
| [workspaces.py:1081](../../pyrs/core/workspaces.py#L1081) `set_sample_log` | Discards the result of `matching_subruns`, so values given for other sub-runs, or in another order, are stored by position | same class, two tests |
| [reduction_manager.py:421](../../pyrs/core/reduction_manager.py#L421) | Decides whether to rebuild the 2θ map by comparing with sub-run `n - 1`, not the one reduced last. A gap or a start above 1 raises; iterating out of order would reuse the wrong engine silently. | `test_reduction_manager.py` |
| [mantid_peakfit_calibration.py:247](../../pyrs/calibration/mantid_peakfit_calibration.py#L247) | `get_diff_peaks(sub_run - 1)` against position-indexed powder tables | `tests/unit/pyrs/calibration/test_mantid_peakfit_calibration.py` |
| [detector_calibration_model.py:141](../../pyrs/interface/detector_calibration/detector_calibration_model.py#L141) | `fitted_ws[sub_run - 1]`, while `fit_peaks` fills `fitted_ws` by position | `test_detector_calibration_model.py` |
| [detector_calibration_crtl.py:77](../../pyrs/interface/detector_calibration/detector_calibration_crtl.py#L77) | `exclude_list[sub_run - 1]`, the same pattern | `test_detector_calibration_crtl.py` |
| [peak_fitting_crtl.py:164](../../pyrs/interface/peak_fitting/peak_fitting_crtl.py#L164) | One typed sub-run is used as a 1-based position in the sub-run list | `tests/ui/test_peak_fitting_crtl.py::TestSubRunNumbersAreNotPositions` |
| [peak_fitting_crtl.py:180](../../pyrs/interface/peak_fitting/peak_fitting_crtl.py#L180) `parse_sub_run_list` | Typed sub-runs are used as 0-based positions, and the range `[1, N)` drops the first position and sub-run N | same class |
| [texture_fitting_crtl.py:28](../../pyrs/interface/texture_fitting/texture_fitting_crtl.py#L28) `trim_data` | Included sub-runs are used as 0-based positions, and a single included sub-run is ignored (`len(...) > 1`) | `test_texture_fitting_crtl.py` |
| [texture_fitting_crtl.py:119](../../pyrs/interface/texture_fitting/texture_fitting_crtl.py#L119) | The first sub-run number is used as a spectrum index | `test_texture_fitting_crtl.py::test_fit_peaks_sub_runs_not_starting_at_one` |

Correction to the survey: `parse_sub_run_list`'s `(x >= 1) == (x < N)` reads like
a typo for `&`, but it is equivalent for `N >= 1`, since both conditions cannot be
false together. The defect is the number-as-position use.

**5. Display order only.** `rstables.py:45` and `manual_reduction_viewer.py:224`
sort for display. The browse order of sliders and tables is also display-only.

Also noted, outside the ordering question: `nexus_conversion` would silently
overwrite a repeated sub-run's counts, which is a uniqueness issue.

### The xfail tests

Each test is `strict=True`, so a fix turns it into a failing XPASS until its
marker is removed. Each one names DD-2 in its `reason` and pins `raises=` to the
exception the defect produces (`AssertionError`, or `pytest.fail.Exception` for
"did not raise"), so a broken arrangement errors rather than passing as an xfail.
None needs Mantid or a widget: the collaborators are plain stand-ins, each checked
to pass with 1..N sub-runs.
- `tests/unit/pyrs/dataobjects/test_sample_logs.py::TestSubRunOrdering`:
  - unique unsorted values are accepted (the recommendation itself);
  - `get_indices` with a missing interior value;
  - `get_indices` on an unsorted axis;
  - plus a passing test pinning that repeated values are still refused.
- `tests/unit/pyrs/core/test_workspaces.py::TestSubRunNumbersAreNotPositions`
- `tests/unit/pyrs/core/test_reduction_manager.py`
- `tests/unit/pyrs/calibration/test_mantid_peakfit_calibration.py`
- `tests/unit/pyrs/interface/texture_fitting/test_texture_fitting_crtl.py`
- `tests/unit/pyrs/interface/detector_calibration/test_detector_calibration_model.py` and `test_detector_calibration_crtl.py`
- `tests/ui/test_peak_fitting_crtl.py::TestSubRunNumbersAreNotPositions`
- `tests/unit/pyrs/utilities/NXstress/test_multi_workspace.py::TestReadBackScanPointOrder::test_shared_value_in_decreasing_order_round_trips`

### Why it is deferred

`SubRuns` is used across the whole data model, so relaxing it touches every
loader and every view, and it deserves its own review. The phase-2 NXstress PR
contains it with the refusal above instead.

### If picked up

1. In `SubRuns.set`, check uniqueness only (`np.unique(value).size == value.size`).
2. Replace the `searchsorted` in `get_indices` with an order-free lookup, such as
   a value-to-position dictionary, that raises `IndexError` for **any** missing
   value, not only the first and last.
3. Compare as sets in `set_wavelength`. Make `set_sample_log` act on
   `matching_subruns`: either refuse a mismatch, or reorder the values to the
   workspace's axis.
4. Remove `_peaks.py`'s within-block ordering check, `_set_subruns`'s `sort`, and
   NXstress's `_refuseUnreadableScanPointOrder`, then remove the xfail markers
   above.
5. Separately: fix the §4 sites to look sub-runs up through
   `get_indices` rather than by `sub_run - 1`. These are independent of steps
   1-4, and each has its own xfail test above.

See also [docs/ground_truths.md](../../docs/ground_truths.md), "Two data-model
limits that bite any multi-record NeXus write", which first recorded that
`SubRuns` cannot hold a non-monotonic array.

---

## DD-3 — Strain/stress viewer: a direction with no fitted peaks crashes on load

**Found during:** phase-2 PR: checking the strain/stress GUI steps in `plans/PRs/phase-2.md`
**Owner:** `pyrs/interface/strainstressviewer/model.py`, which phase 2 does not otherwise change here
**Status:** open, not patched here

### What happens

Loading a project file with no peak collections as **e22** or **e33** raises
`IndexError`. `Model.check_peak_for_direction`
([model.py:136](../../pyrs/interface/strainstressviewer/model.py#L136)) answers
"does every file carry the selected peak?" by looping over the files' peak
dictionaries, so with none it returns True. The `e22`/`e33` setters
([model.py:61](../../pyrs/interface/strainstressviewer/model.py#L61),
[model.py:75](../../pyrs/interface/strainstressviewer/model.py#L75)) then call
`create_strain` ([model.py:125](../../pyrs/interface/strainstressviewer/model.py#L125)),
which builds an empty strain list and indexes its first element. The `selectedPeak`
setter takes the same path for every loaded direction. **e11** is not affected,
because its setter does not build a strain.

Measured on `tests/data/HB2B_1320.h5` with its `peaks/peak0` group deleted,
through `Controller.filesSelected("e22", [path])`:

```
e22 -> IndexError: list index out of range
  File "/home/ux0/workspaces/PyRS/pyrs/interface/strainstressviewer/model.py", line 134, in create_strain
    setattr(self, f"_e{direction}_strain", sum(strain_list[1:], strain_list[0]))
                                                                ~~~~~~~~~~~^^^
IndexError: list index out of range
```

In the viewer the exception leaves the **Browse** button's slot uncaught.

### Why it is deferred

It predates phase 2, and phase 2 does not change what it depends on. Input to the
strain/stress viewer is a fitted project file in practice, so a file with no peak
collections is unusual. It was found while exercising the phase-2 direction check,
not caused by it.

### If picked up

1. In `check_peak_for_direction`, return False when the direction has no peak
   dictionaries (or none holding `selectedPeak`), so no strain is built.
2. Decide whether such a file should be refused with a `failureMsg`, since a
   direction with no peaks cannot contribute a strain.
3. Regression test beside `tests/integration/test_strainstress_direction.py`,
   loading a no-peaks file as e22.

---

## DD-4 — Strain/stress viewer: `Controller.fileSelected` is dead, and broken

**Found during:** phase-2 PR: fixing `Controller.filesSelected`'s success check
**Owner:** `pyrs/interface/strainstressviewer/controller.py`
**Status:** open, not patched here

### What happens

`Controller.fileSelected(name, filename)`
([controller.py:5](../../pyrs/interface/strainstressviewer/controller.py#L5))
calls `self._model.set_workspace(...)`, which `Model` does not define
(`hasattr(Model(), "set_workspace")` is False), so any call raises
`AttributeError`. Nothing calls it: a `grep` over `pyrs/` and `tests/` finds only
its definition. The plural `filesSelected` is the one the viewer uses. The
method dates from af206107 (2020-06-25).

### Why it is deferred

Dead code has no symptom, and removing it is unrelated to phase 2.

### If picked up

Delete `fileSelected`.
