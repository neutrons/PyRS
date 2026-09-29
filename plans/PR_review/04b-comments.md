# PR review — subspec 04b (multi-workspace NXstress I/O)

Review of `eb5457b1..cda22352`, per
[`plans/PR-review-process/review-process.md`](../PR-review-process/review-process.md).

Each entry: the comment, the agreed resolution, and the change actually made.
My summaries of the diff are not recorded (§6e); defects I flagged are, whether or
not they were acted on.

---

## Batch 2 — `pyrs/utilities/NXstress/_discriminator.py`

### D4 — a read-only property breaks the discriminator round trip, silently

**Flagged by me.** `resolve` uses a `HidraWorkspace` property whenever one exists;
`apply` used it only when `fset is not None`, otherwise writing a sample log. All
six `HidraWorkspace` properties are read-only, so a field naming one was read from
the property and written back to a log: the value reached the file correctly and
came back as the constructor's default, with nothing raising.

**Reviewer:** *"Since `HidraWorkspace` may need to have readonly properties for
'user facing' reasons, I think the correct solution is probably to add an
internal-use (i.e. non private) accessor method to `HidraWorkspace` for use by
`NXstress`. We cannot simply omit the property, or raise if it is readonly, and the
round-trip verification is definitely a requirement."* Then, after reviewing the
options: *"What I want to work out, is a well-designed method to allow properties
which are used as discriminators to be settable, but without making them read/write
properties."*

**Why it could not wait.** `strainstressviewer/model.py` constructs
`HidraWorkspace(direction)` — the viewer already carries direction in a read-only
property, and spec 05 is the viewer. A `set_<name>()` method was rejected on the
reviewer's criterion: it makes the value writable in all but spelling.

**Resolution: a `restorable_property` descriptor**, declared on the property
itself, in its own module.

**Changes:**
- **New** [`pyrs/utilities/restorable_property.py`](../../pyrs/utilities/restorable_property.py)
  — `restorable_property` (a `property` subclass naming a backing attribute),
  the `restorable()` decorator, and `is_restorable()` / `restore()`. Assignment
  still raises `AttributeError`; `isinstance(…, property)` still holds, so
  `resolve`'s read path is untouched. A `setter()` call raises rather than
  silently producing a writable property.
- `pyrs/core/workspaces.py` — new `_direction` field; `direction` as a
  `restorable_property`; `hidra_project_file` marked restorable (it was settable
  only as a side effect of `load_hidra_project`, so a workspace rebuilt from any
  other source could not record its origin).
- `pyrs/interface/strainstressviewer/model.py` — one line in
  `load_hidra_project_file`, the single funnel all three directions and both entry
  points pass through.
- `_discriminator.apply` — a branch for `is_restorable`, between the settable
  property and the sample-log fallback.
- `_discriminator.field_names` — rejects, **at configuration time**, a field
  naming a read-only property that is neither settable nor restorable. This is
  what makes the asymmetry unreachable rather than merely fixed.

**Two decisions inside the resolution.** `direction` is backed by `_direction`,
**not** aliased to `_name` — aliasing would make `ws.direction` answer
`"Combined Project Files"` for `CombineRunsModel`'s workspace. And its getter
falls back to the `direction` sample log when `_direction` is unset, a deliberate
compatibility shim that kept every existing test working unchanged; spec 05
retires it (05's Follow-up 2).

### D6 — `_discriminator` was the only module bypassing the documented `Config` access

**Flagged by me, and I got the diagnosis backwards first.** `_discriminator` did
`from pyrs.utilities import config as _config` and reached `_config.Config[...]`,
with a comment explaining that a bound name goes stale when the `default_config`
fixture reloads. I verified the comment's claim was *true* and concluded the
deviation was therefore correct.

**Reviewer:** *"this override capability is a test-only capability, so what
`_discriminator.py` is doing should definitely not be viewed as correct access!"*
— which is the deciding point: production shape must not be bent around a test
capability. `config.py`'s docstring says *"Import `Config` from this module"*, and
every other consumer does.

**What the investigation then found**, in order:
- There is **no config-based property decorator** in the installed
  `neutrons_standard` — the capability the fix might have leaned on does not exist.
  The absence is the finding.
- `Config.__getitem__` → `_find` reads `_config` **live on every access**, with no
  cache, so a test can override in place with no rescan. Three caveats: a helper
  must deep-merge (a shallow assignment drops siblings — verified by `KeyError`);
  `_fix_directory_properties` only matters for `instrument.home`/`samples.home`;
  and nothing re-validates, since `_Config.validate()` is `pass`.
- **`default_config` leaked.** Its teardown reloaded to yet another new singleton,
  so any test using it orphaned every consumer module for the **rest of the
  session**. That is why the first version of the override fixture worked in
  isolation and failed in a full run.

**Changes:**
- `_discriminator.py` → `from pyrs.utilities.config import Config`.
- `tests/unit/pyrs/utilities/conftest.py` — new `config_override` fixture that
  deep-merges into the live singleton's `_config` and restores a deep copy, never
  swapping the instance; **and** `default_config` now restores the pre-test
  singleton instead of reloading to a third one.
- The three NXstress test files switched from `default_config` + `loadEnv` to
  `config_override`.

**Side effect worth naming:** `_instrument.py` was unoverridable in tests —
`_instrument_names()` returned the shipped default whatever a test configured, so
any test asserting a configured instrument name was passing vacuously. It works now.

### D5 (nit) — a field name with surrounding whitespace is accepted

**Flagged by me.** `field_names` checks `name.strip()` is non-blank, then uses
`name` unstripped, so `' direction '` becomes the column `__20direction__20`.

**Not resolved.** `OPEN DEFECT` — see below.

---

## `OPEN DEFECT` — **closed in batch 7, and the diagnosis was wrong**

**D5 — whitespace in a configured discriminator field name.** Recorded in batch 2
as "a config typo produces a valid but surprising column", with two proposed
fixes: reject a name differing from its own `.strip()`, or normalise it.

**Both proposals were wrong, and so was the framing.** See batch 7 (D25) in this
document: the encoding is not the problem, whitespace is incidental, and
restricting field names would have broken a case that works today. Kept here
unedited as the record of what was believed.

---

## `RE-REVIEW RECOMMENDED`

**Files:** `pyrs/utilities/restorable_property.py` (new),
`pyrs/core/workspaces.py`, `pyrs/interface/strainstressviewer/model.py`,
`tests/unit/pyrs/utilities/conftest.py`, `pyrs/utilities/NXstress/_discriminator.py`.

**Why:** §5's *"a fix agreed during review outgrew its estimate"*. Batch 2 began as
a read of one 322-line file and produced a new module, a change to a core data
class, a change to a viewer, and a rework of shared config test infrastructure —
four of those outside the PR's original 66-file diff. Per §5 this is also the
weakest configuration the process has: I implemented it and then verified it with
tests I wrote in the same pass.

**Scope for a second pass:** the `restorable_property` semantics (does read-only
really hold everywhere it is used?), and the `config_override` / `default_config`
interaction, which is subtle enough that it failed in a full run after passing in
isolation.

### Second pass, 2026-10-16 — both named concerns answered; one defect found

**Read-only holds everywhere it is used.** Verified by execution, not reading,
on both real properties (`HidraWorkspace.direction`,
`HidraWorkspace.hidra_project_file`) and on synthetic cases: assignment raises
`AttributeError`, `.setter()` raises `TypeError`, and `restore()` writes the
correct backing attribute in both the derived and the explicit-`restores` form.
`_discriminator.field_names()`'s guard also fires as intended — a discriminator
naming a plain read-only property is refused at configuration time, so the
`resolve`-reads-property / `apply`-writes-log divergence cannot arise.

**The `config_override` / `default_config` interaction holds**, and this
mattered more than when the concern was written: `config_override` gained a
`validate_config()` call in the same session (DD-1), and it binds
`config_module` at fixture setup while `default_config` *reloads* that module —
so a stale binding was plausible. Probed directly: an override applied after a
`default_config` test still reaches the bindings production modules hold, the
validation call reads the same object the merge wrote to, and cleanup restores.
Passes in isolation **and** in a full run, which is the exact split the original
concern named.

#### REQUIRED FIX — `restorable_property.deleter()` silently corrupts `restores`

**Not yet fixed.** Recorded here so it is not lost.

`restorable_property` overrides `getter` and `setter` — the former carrying the
comment *"must not silently drop `restores`"*, so the hazard was understood —
but **not `deleter`**. `property.deleter` reconstructs through
`type(self)(fget, fset, fdel, doc)` without passing `restores`, so `__init__`
falls back to re-deriving it from the getter's name:

```console
declared          : _file_path
after .deleter    : _source_file   <-- re-derived, not preserved
restore() wrote   : _file_path='a.h5'  source_file='a.h5'
stray attribute   : ['_source_file']
round trip LOST?  : True
```

No exception. The wrong attribute is written, a stray one is created, and the
value is lost — which is exactly what the module docstring says `restore()`
exists to prevent: *"restoring something that merely looks assignable is how a
reconstruction quietly writes to the wrong place."*

**Live hazard, not hypothetical.** `workspaces.py:118` is
`@restorable(restores="_project_file_name")` — a property whose backing name
does not correspond to its getter, which is precisely the case this breaks.
Nothing calls `.deleter()` today, so it is latent; the fix is a three-line
override mirroring `getter`, plus a test.

**Also:** `is_restorable()` accepts a class or an instance — documented, and
used with a class at `_discriminator.py:166` — while `restore()` accepts only an
instance. The module docstring presents them as a pair
(`if is_restorable(ws, name): restore(ws, name, value)`), so the asymmetry is an
invitation to write a call that fails only for the class form.

#### Independence — and it was worth it

The two findings above came from the same author who wrote the code, which is
the weakness this note exists to name and is not resolved by looking harder.
`design-reviewer` and `test-reviewer` were run over all five files as an
independent pass. **Both found more than I did, and one of them found a defect
I introduced during this very session.**

Process error worth recording: I launched both reviewers while still editing
`conftest.py`. The design reviewer caught the tree changing under it, re-ran
everything and said so — but that was avoidable noise, and a reviewer that had
*not* noticed would have reported artefacts as findings.

##### The mechanism was wrong, and only a type checker could see it

`restorable_property` subclassed `property` so a value could stay read-only
while an I/O reader restored it. **mypy special-cases the literal `@property`
decorator syntactically** and does not recognise a subclass used as a
decorator, so:

```
t.plain       -> "str"   @property       t.plain      = "x" -> error: read-only
t.viafactory  -> "Any"   @restorable()   t.viafactory = "x" -> (no error)
```

The construct whose entire purpose was to keep a value read-only was the one
option on the table that **removed the static guarantee**, and degraded every
annotated return type to `Any` at the same time. `ws.direction = "22"` was a
mypy error before this module existed and silent after it.

Replaced: the mark now goes on the **getter**, under an ordinary `@property`
(`@restores("_direction")`). Verified on the real class — `ws.direction` reveals
`str`, assignment is an error again. This also dissolves two other findings by
construction: `deleter` and `setter` carry `fget` across unchanged, so the mark
cannot be lost by a rebuild that someone forgot to override.

##### What the two reviewers found between them

| | Finding | Status |
|---|---|---|
| D1 | `property` subclass defeats mypy's read-only enforcement and return types | **fixed** — mark moved to the getter |
| D2 | `.deleter()` silently re-derived `restores` | **dissolved** by D1's fix; pinned by test |
| D3 | `restore()` never checked the backing attribute exists — a typo in `@restores(...)` wrote a second, unread attribute and lost the value silently | **fixed** — raises `AttributeError` naming the spelling |
| D4 | `setter()` guard bypassable: `__init__` still accepted `fset` positionally | **dissolved** — no subclass any more |
| D5 | `apply()` fell through to a sample log for a read-only unmarked property, while `resolve()` read the property — the exact silent divergence the module exists to prevent, still reachable | **fixed** — `apply` now raises where the violation happens, not only in a different function at config-read time |
| D6 | `apply()` did not normalise through `_as_text` while `resolve()` did, so `numpy.str_` landed on a property annotated `-> str` | **fixed** |
| D7 | `restore()`/`is_restorable()` disagreed about accepting a class, while the module docstring pairs them as one idiom | **fixed** — `restore` rejects a class with a message naming the real problem |
| D8 | **No test file for `restorable_property.py` at all**; coverage 87%, the three uncovered lines being its three `raise` guardrails | **fixed** — `tests/unit/pyrs/utilities/test_restorable_property.py`, 18 tests |
| D9 | The whole mechanism could be deleted with the suite green | **fixed** — see below |
| D10 | The `config_override` validation hook was unpinned, and had silently killed 8 assertions | **fixed** — see below |
| D11 | `_validate_round_trippable`'s `prop.fset is not None` clause never exercised; the test's name promised more than it checked | **fixed** — settable half covered explicitly on a stub |
| D12 | `test_multi_workspace.py:27` read *"`config_override` rather than `config_override`"* | **fixed** |

##### D9 — the mechanism could be removed with everything green

The test reviewer's mutation pass found that `direction` is **self-masking**:
delete `apply()`'s restorable branch and `apply` writes a `direction` sample
log, which `workspaces.py`'s legacy log fallback then answers from, so
`resolve` still returns the right value. Only `hidra_project_file` constrained
the branch — and *its* marking was unpinned too, because the `vars()` sweep is
self-adjusting and an un-marked property simply moves buckets.

Compound mutant — un-mark `hidra_project_file` **and** delete the branch:
**entire suite green**. Two edits, neither failing anything on its own,
removing the feature this PR exists to add.

Closed by a test that defeats the mask: a workspace with **no** `direction`
log, so the fallback has nothing to answer with and only the property can.
Re-run of that compound mutant now: **13 failures**.

##### D10 — I introduced this one today, while fixing DD-1

I added `validate_config()` to `config_override` and wrote in DD-1 that the
guarantee "was checked by measurement rather than assumed". True of a
standalone script; **false of the suite** — deleting the hook left all 675
tests green. Worse, the hook made `test_config.py:354`'s own `validate_config()`
unreachable, so the 8 tests in `TestValidateConfigRejectsMalformedValues` were
capturing the *fixture's* exception instead of the subject's. They passed, and
had stopped testing what their names said.

This is the review's recurring finding — an assertion that cannot fail —
committed by me in the same session I spent writing about assertions that
cannot fail, and found by an independent reviewer rather than by me.

Closed: `TestConfigOverrideRefusesMalformedConfig` pins the hook directly, and
the dead line is gone. Re-run of that mutant now: **11 failures**.

##### Checked and not acted on

- `default_config` at `test_instrument.py:225` reported as a no-op: **does not
  apply** — there is no `default_config` anywhere in that file. The reviewer saw
  a transient tree state. Verified rather than "fixed".
- `restore()` as a public back-door setter, where the codebase's other readers
  use ordinary public setters (`NXstress.py:889-899`): a real API-scope
  judgement, not a defect, and a larger change than this pass. Left as written.
- The `config_override` / `default_config` fixtures were independently cleared
  by both reviewers, agreeing with the finding above.

---

## Batch 3 — `pyrs/utilities/NXstress/_peaks.py`

### D7 — a string discriminator column was created as `float64` when the entry has no peak collections

**Flagged by me.** `discriminator_dtypes` infers each column's dtype from the
values present. With no collections there are none, and `np.asarray([]).dtype` is
`float64`, whose kind is not in `("U","S","O")`, so the string branch was skipped.
Reachable in production: `CombineRunsModel` exports via `write([ws], [[]])`
(Decisions row 28), and spec 05 configures `discriminator_fields: ["direction"]`.

Measured before the fix — the file round-tripped through PyRS, so the harm was a
wrong on-disk type for any other reader:

```console
  peaks/direction   dtype=float64  shape=(0,)     <-- should hold strings
  peaks/phase_name  dtype=object                  (for comparison)
  read back: 1 workspace(s), [0] collection(s)
```

**Reviewer:** *"Emit no discriminator columns when there are no collections."*

**Change:** `discriminator_dtypes` returns `{}` for an empty `indexed`, so `_init`
emits no column. Verified: the entry now carries only the 15 reserved columns and
still reads back as one workspace with zero collections. The docstring records why
the empty case is not a detail.

### Reviewer question — is such a file then non-appendable?

*"I think then that such a NXstress file is non-appendable, except via the
multiple NXentry route."* **Correct, and already enforced.** Measured:

```console
  append to a column-less entry -> RuntimeError:
    "the target entry carries no discriminator columns, so an appended
     workspace could not be told apart from the one(s) already there."
  new NXentry route still available? yes
```

**No change** — 04c's Case-A precondition 2 already covers it.

### Raise earlier than "interleaved blocks detected"

**Reviewer:** *"When `discriminator_names` is `()` (AND should this be `None`
instead of `()`?) — there is no possible merge allowed? Is this correct? My point
is we can raise a RuntimeError earlier than 'interleaved blocks detected ...'."*

**The case for raising is stronger than it looked.** Omitting the names does not
produce a confusing error — it produces a **silent wrong answer**:

```console
  peakCollectionRanges(peaks) with the argument FORGOTTEN:
    no error; returned 1 range -- WRONG, should be 2 workspaces
```

Both workspaces shared `('Fe',1,1,0,'_DEFAULT_')`, so their adjacent blocks merged
into one range, and the monotonicity check passed because the concatenated scan
points were still increasing. "Interleaved blocks detected" is only the failure
mode when the data happens to be shaped differently — the failure varies with the
data.

**On `None` vs `()`: kept `()`.** Once the group is authoritative via the check
below, `None` and `()` behave identically, so the extra state buys nothing. The
alternative use for `None` — "derive the names yourself" — would make
`peakCollectionRanges` read configuration, turning a pure function over a group
into one with a hidden config dependency, and would mask the caller's mistake
rather than surface it.

**Change:** new `_Peaks._validateDiscriminatorNames`, called first in
`peakCollectionRanges`, comparing the caller's names against the columns actually
present and naming both sets. Distinct from `_discriminator.names_for_read`, which
compares *configuration* against the file.

### `merge_workspaces` does not make NXstress merge anything

**Reviewer:** *"the NXstress implementation does not merge workspaces -- that's
something that is external (in PyRS itself)."* **Correct**, and my phrasing had
attributed a merge to NXstress. `merge_workspaces` is a config key only — it
appears in no method signature — and its whole effect is permissive: it lifts
`_validateMultiWorkspace`'s refusal so rows may be concatenated with no boundary
recorded. Merging that produces a genuinely merged `HidraWorkspace` happens in
PyRS first (`HidraWorkspace.append_hidra_project` via
`CombineRunsModel.combine_project_files`, which then passes a length-1 list).

**Change:** phrasing corrected in all three places I had introduced or inherited it
— `_discriminator.names_for_read`, `_peaks._validateDiscriminatorNames`, and the
new test's comments. `()` now reads as "the entry records no workspace boundary"
rather than "the entry is merged".

Four other sites use "merged write/entry" as shorthand without attributing an
operation (`_peaks.py:79`, `:458`, `_discriminator.py:175`, `:300`); left as
shorthand now that `names_for_read` carries the authoritative note.

### Reviewer question — does `merge_workspaces: false` with N>1 and no discriminators raise?

**Yes.** Measured across all three configurations:

```console
  fields=[]            merge=False -> ValueError: given 2 workspaces, but
                                      'nxstress.discriminator_fields' names no field
  fields=[]            merge=True  -> ValueError: Duplicate PeakCollection detected
                                      at ('Fe', 1, 1, 0, '_DEFAULT_')
  fields=['direction'] merge=False -> ValueError: sample log 'direction' not found
```

The middle row is worth recording: **even with `merge_workspaces: true`, two
workspaces sharing a compound key are still refused**, by `validateNoDuplicatePeaks`
— with no discriminators their sort keys are identical. So the escape hatch only
works when the inputs' compound keys already differ, and the ordinary
multi-direction case (two workspaces measuring the same peak) cannot use it at all.
**No change**; recorded because it narrows what the setting is for.

---

## Batch 4 — `pyrs/utilities/NXstress/_fit.py`

### D8 — a round trip added an all-NaN mask to a workspace that never reduced it

**Flagged by me.** One `DIFFRACTOGRAM` group spans the whole entry, so when one
input reduced a mask and another did not, `_concatenated_diffraction` NaN-fills
the second's rows to keep the scan-point axis aligned — deliberately, and
documented. But the read side handed every group in the entry to every workspace,
so the second got the mask back:

```console
  before write:      ws0 [None, 'mask_a']    ws1 [None]
  after round trip:  ws0 [None, 'mask_a']    ws1 [None, 'mask_a']   <-- all-NaN
```

`reduction_masks` is *counted* by `texture_fitting_crtl.py:142`
(`if len(...) == 2:`) and `mantid_peakfit_calibration.py:240`, and drives
`setup_out_of_plane_angle` / `enable_polar_plot` in the texture viewer. Latent
today — it needs a multi-workspace file with uneven reduced masks opened in that
viewer — and live once spec 05's multi-direction files exist.

**Reviewer:** *"Drop the all-NaN mask on read. HOWEVER, I'm wondering if we should
just warn and drop the mask with no reductions associated with it on write... a
user setting up a set of masks, and just not bothering to perform a specific
reduction; that might not necessarily be an error."*

**On the write-side proposal: that case is already handled, and is a different
case.** Measured, for a single workspace defining `mask_a` but never reducing it:

```console
  _mask_dict keys       : ['mask_a']            <- defined
  _diff_data_set keys   : ['None']              <- never reduced
  DIFFRACTOGRAM groups  : ['DIFFRACTOGRAM']     <- no DIFFRACTOGRAM_mask_a written
  instrument/masks/names: ['_DEFAULT_', 'mask_a']  <- the mask array IS preserved
  round-tripped reduction_masks: [None]         <- unchanged
```

`_Fit.init_group` builds `mask_keys` from `_diff_data_set`, not `_mask_dict`, so
an unreduced mask already produces no diffractogram, while the mask array itself
still reaches `instrument/masks` and round-trips into `_mask_dict`. Nothing to
change, and no warning is warranted: it is not an anomaly, just a mask that was
defined and not used.

**And a write-side drop would be actively wrong for D8.** There, `mask_a` *does*
have reductions — the first input's. Dropping it at write would discard real data
to tidy up the second input's absence. Whether a workspace has a mask is a
per-workspace fact, so it can only be resolved per workspace, which is the read
side.

**Change:** `NXstress._workspaceFromNexus` omits a mask whose data is entirely NaN
across *this workspace's* rows, logging at INFO. The two-theta matrix is taken
before the check, so dropping the first group does not lose the axis. The comment
records the accepted cost: a reduction that genuinely produced only NaN is
indistinguishable from one that never ran and is dropped too — preferred over
silently changing a workspace's mask set across a round trip.

### Observations, no change

### D9 — `_2theta_matrix`'s row count was never validated against the sub-run count

**Flagged by me as an observation; escalated by the reviewer.** NXstress writes
`scan_point` from the sub-runs and the diffractogram rows from `_2theta_matrix`.
A workspace whose two disagree produced a file whose rows did not correspond to
its scan points, with nothing saying so. Pre-existing, but N workspaces amplify
it: one short input shifts every later input onto the wrong rows.

**Reviewer:** *"The following must be guarded; we should not allow a silent error
like this!"* and *"I would probably guard it in both places; or at least add an
additional test ensuring that the guard does not regress (and pointing to all of
its consumers in a comment at the test)."*

**Change — guarded in both places, because one is not enough.**
`HidraWorkspace.set_reduced_diffraction_data_set` now rejects a `two_theta_matrix`
that is not 2-D, a mask array whose shape disagrees with it, and a row count that
contradicts the sub-runs when those are known. But that setter is only **one of
five** paths that write `_2theta_matrix`; the other four bypass it entirely:

| path | reaches `_2theta_matrix` |
|---|---|
| `_load_reduced_diffraction_data` | reading a `HidraProjectFile` |
| `_append_reduced_diffraction_data` | `append_hidra_project` |
| `set_reduced_diffraction_data` | row-wise; `reduction_manager`, `mantid_peakfit_calibration` |
| direct assignment | anywhere |

So `_Diffractogram._concatenated_diffraction` carries the second guard, at the
write boundary every one of those funnels through, naming **which input** is
short — which matters precisely because the damage is not local.

Four tests in `test_fit.py::TestReducedDataRowCountIsGuarded`, whose class
docstring carries that table so a future reader knows what each guard covers and
why removing either leaves a silent route to a misaligned file. Two exercise the
setter; two force the mismatch *past* it, as the other four paths can.
- **`_concatenated_diffraction` runs twice per append** — once in `validateAppend`,
  once in `init_group`. The cost of pre-flighting rather than discovering
  mid-mutation, and the largest array work on that path.

### Cross-reference — `_fit.py` also changed in batch 5

Batch 5 (`_instrument.py`, written up in
[04-comments.md](04-comments.md#batch-5--pyrsutilitiesnxstress_instrumentpy-brought-forward-during-batch-4),
04's Follow-up 8) separated the detector-mask namespace from the diffractogram
keyspace, and three of its edits land in this file:

- `_Fit.init_group` and `_Fit.validateAppend` now key on
  `nxstress_diffractogram_keys` instead of `nxstress_mask_names`, so the default
  tag is no longer forced into the keyspace.
- `_Fit.init_group` validates that each diffractogram key references a detector
  mask that is actually registered, via the new
  `diffractogram_detector_mask(key, known_masks)`.
- `validateWorkspaceAndPeaksData` lost its mask-namespace check on
  `PeakCollection.mask`; its `_diff_data_set` check was the correct one all along.

Noted here rather than re-argued: the defect is 04's, the file is 04b's.

---

## Batch 3 reopened — `pyrs/utilities/NXstress/_peaks.py` (static typing)

Raised after batches 4 and 5 were committed (`602ec3b5`), by the reviewer
challenging the phrase "mypy at the 11-error baseline" in that batch's
verification report.

### D14 — `mypy` was measured against the wrong referent, and the errors are this PR's

The 11 errors were real and were being reported each batch as a stable
*baseline*. They are not inherited:

```console
next   : Success: no issues found in 8 source files
HEAD   : Found 11 errors in 1 file (checked 10 source files)
```

All 11 blame to `7799a6738` — subspec 04b, one of the three subspecs under
review. **The measurement was correct and the word was wrong**: "baseline" means
the merge target, and I had taken it to mean the state at review start, which for
this review is the thing being reviewed. Nothing about the numbers would have
revealed it; only re-anchoring them did.

### The defect itself

Ten of the eleven come from `IndexedPeaks = IndexedPeaks` inside the `_Peaks`
class body. Harmless at runtime, but it binds a class variable that shadows the
module-level `NamedTuple` for the rest of the body, so every later
`list[IndexedPeaks]` annotation named a variable, not a type — meaning the one
type this subspec introduced was also the one type `mypy` could not check.

The accompanying comment justifies the *module-scope definition* (`@validate_call_`
resolves annotations while the class body is still executing), which is correct
and now sits on the definition. It does not justify the alias. Nothing consumed
the alias: `_Peaks.IndexedPeaks` occurred twice repo-wide, both docstrings in the
same file.

The eleventh: `indexed(..., logs: SampleLogs = None)` against a field declared
`SampleLogs | None`.

Verified before and after by experiment, not inference — commenting out the one
line dropped `mypy` from 11 errors to 1 with 236 NXstress unit tests still
passing, which is what established that the alias had no consumer.

### `mypy` is still not enforced

Per the reviewer's direction it was **not** added to `.pre-commit-config.yaml`.
It is in `pyproject.toml`'s dependencies with no hook and no pixi task, so
nothing runs it; turning it on would surface the rest of `pyrs/`, which this
review is not scoped to. Recorded so the absence reads as a decision.

---

## Batch 27 brought forward — `probes/a5_nxstress_roundtrip.py`, `probes/README.md`

Raised when the reviewer challenged a phrase I had been repeating in every
batch's verification line — "all probes run except the deliberately-retired
`a5_nxstress_roundtrip`" — and then corrected a claim I made while explaining it.

### What I got wrong

I said the probes "are run by nothing". **They are run by the full audit** —
`process.md` §7.5 makes them audit evidence, re-run on each pass. What is true is
narrower and was the real point: nothing runs them *between* audits, which is why
`a5_transformations_chain.py` was broken from 04b until batch 5.

That correction is what makes the rest a defect rather than tidiness. If an audit
re-runs the directory, a probe that is *expected* to fail must say so **in the
file**, because a traceback is indistinguishable from a regression.

### D15 — the retirement was recorded in one place, and that place is not the file

The disposition lived only in `probes/README.md`'s table. The probe itself opened
as a live A5 probe and closed with `Run: pixi run python …`. Two further things in
it had gone stale:

- **Claim 4** — "`NXstress.py:151-152` raise on any operation that would extend an
  existing `NXentry`" — is stated as a claim under test, and 04c **inverted** it
  (Decisions row 33). A reader takes it as current behaviour.
- The *"Deliberately NOT probed"* paragraph said 04b's round trip, discriminator
  resolution and 04c's append path "cannot be probed: none of that code exists
  yet… uncovered until 04b and 04c land." All three have landed and all three are
  now probed.

And the one statement that still executes, `inspect.signature(NXstress.write)`,
now prints the **post**-04b signature under a claim heading asserting the pre-04b
one. So the file's live measurement contradicts the claim it is filed under.

### D16 — the successor was named as a test, and only for one of five claims

`process.md` §7.5 item 6 is *supersede, do not delete* — "keep the probe and say
so in **both files**." Neither file did. The index named `test_multi_workspace.py`,
which is a test, not a probe, and covers one claim. The actual map:

| claim | successor |
|---|---|
| 1 — I/O types `(HidraWorkspace, list[PeakCollection])` | `a5_discriminator_resolution.py`, `a5_subruns_nonmonotonic.py` |
| 2 — read-back completeness, raw counts only with `input_data` | **none** |
| 3 — multiple `NXentry` per file | `a5_append_preconditions.py` |
| 4 — mode `"a"` cannot extend an entry | `a5_append_preconditions.py` — *and the claim is now false* |
| 5 — pre-04b `write` signature | nothing; it is history, which is the part that justifies keeping the file |

**Claim 2 is the one that matters beyond bookkeeping.** Retiring this probe
dropped read-back completeness out of probe coverage entirely — `read()` is now
called by exactly one other probe (`a5_append_preconditions.py`), and only for the
append round trip. The matrix is not over-claiming: `README.md`'s A5 cell is
already `~`. But the *reason* it is `~` now includes something it did not before,
and that belongs in `review/findings.md` §3 — **left for batch 28**, which owns
that file.

### Agreed fixes (three, all applied)

1. **The probe is self-describing.** A `RETIRED` block opens the docstring: why,
   the exact failure to expect (`ValidationError` at `nxs.write(ws, peaks)`), the
   sentence *any other failure is a real finding*, and the successor map. Claim 4
   is marked **NO LONGER TRUE** at the claim; the stale "cannot be probed"
   paragraph is kept with a note that all three have since landed. No claim is
   rewritten — §7.5 item 6, a log preserves what was believed.
2. **`AUDIT_STATUS = "retired"`**, module level, so a sweep honours the retirement
   without parsing prose. Absent means live. It sits below the imports because
   `plans/` is linted and a statement above them trips ruff `E402`.
3. **The index row** gains the per-claim successor map and the expected failure,
   and its Verdict now says which two claims have since become false.

Plus a new convention, since this is not specific to one probe:
`process.md` **§7.5 item 9** states what a retired probe must carry, and
`probes/README.md` gains a *"Running them all, during an audit"* section with a
sweep that keys on the marker rather than the filename — so a second retirement
is not hard-coded into whatever ran the last one.

### Verification

The documented sweep, run as written: **19 OK, 1 SKIPPED (retired), 0 FAIL.**

---

## Batch 5 completed — `_input_data.py`, `_sample.py`

**Process note first.** Appendix A's batch 5 is three files — `_input_data.py`,
`_instrument.py`, `_sample.py`. Only `_instrument.py` was walked: it was pulled
forward during batch 4's mask discussion, written up under the heading "Batch 5",
and the walk then moved to batch 6 on the label rather than on the file list.
`_input_data.py` and `_sample.py` were skipped and are reviewed here, after batch
6. Between them they own the scan-point family, the retained-log columns,
`_writable`'s dtype coercion and the `detector_counts` append path — not files to
let through on a heading.

**Reviewer:** *"You need to deal with '_input_data.py and _sample.py remain
unwalked' -- we are beyond 'batch 5' at this point!"*

### D23 — `stress field direction` round-tripped 1-D into 2-D, silently

The direction is held as a per-scan-point log but the schema defines it as a
scalar attribute on `stress_field`. The writer stored the whole array:

```python
direction = sampleLogs[direction_key] if direction_key in sampleLogs else "x"
sff.attrs["direction"] = direction
```

and `sampleLogsFromNexus` broadcast it again. Measured on three scan points:

```console
WRITE  direction attr value : ['z', 'z', 'z']        (TODO says: scalar in {x,y,z})
READ   recovered dtype/shape: <U1 (3, 3)
       recovered value      : [['z' 'z' 'z'] ['z' 'z' 'z'] ['z' 'z' 'z']]
       original was         : ['z' 'z' 'z']
```

Nothing raised: `SampleLogs.__setitem__` checks the first axis only, and
`len((3, 3)) == 3`. The same line also used `logss[0]` with **no cross-input
agreement check**, where `name` and `chemical_formula` raise through `_scalar`.

**The existing test documented the defect instead of catching it.**
`test_Sample_stress_field_present` asserted:

```python
# The direction attribute gets the array value
if isinstance(direction_val, list):
    assert direction_val[0] == "z"
else:
    assert direction_val == "z"
```

— an accommodation for the array case, with a comment naming it. Third instance
of this pattern in the PR, after `N_SCAN = 7` (D18) and the `|S` fixture (D20): a
hazard understood precisely enough to be written around in a test, and not
reported against the code.

**Fixed.** The direction goes through `_scalar`, so it is a scalar and must agree
across inputs; the read side takes `to_text(np.atleast_1d(attr)[0])`, which also
recovers a file written before the fix as a 1-D log rather than a 2-D one. The
test now asserts a bare scalar, and `TestStressFieldDirectionIsEntryWide` adds
the round trip, cross-input disagreement, and variation within one input.

**Left as it was, deliberately:** the `else "x"` default still invents a
direction when no log is present, rather than omitting the attribute. Changing it
alters what the read side recovers, and the writer's own TODO records that there
is no real example of these entries. Flagged rather than guessed at.

### D24 — `_scalar` compared normalised values and returned the raw one

```python
distinct = {to_text(v) for v in values}  # comparison normalised
return values[0]  # un-normalised value written
```

Harmless once logs are `str` in memory, but it is the pattern the centralisation
(D19) had just removed everywhere else. Now returns the agreed `str`.

**And it only compared the first element of each input.** Since these are
entry-wide values held as per-scan-point logs, a log that *varies within* one
input is as wrong as two inputs disagreeing — and `[0]` silently kept one. Every
value is now compared.

### Growable inconsistency — fixed

`_input_data` wrote `maxshape=(None, None), chunks=CHUNK_SHAPE(2)` and
`_sample`'s `scan_point` wrote `chunks=CHUNK_SHAPE(1), maxshape=(None,)`.
Literally equivalent to `**growable(2)` / `**growable(1)`, so no behaviour
change — but these were the last two sites not using the helper that exists to be
the single spelling. `CHUNK_SHAPE` is no longer imported by either module.

### O5-O7 — fixed, at the reviewer's direction

*"Let's go forward with 05, 06, and 07, before we call this batch complete?"*

**O5 — the assertion-message match.** `if "some coordinates do not have finite
values" in str(e)` depended on upstream wording: a rewording would stop the
branch matching and let the `AssertionError` escape mid-write.

New `_Sample._coordinates_are_finite(logs)` asks instead of provoking, and
`get_pointlist` is now called only when it will succeed. `PointList`'s other two
assertions still propagate — they report a *malformed* coordinate set, not an
absent one. A **missing** coordinate log deliberately still raises `ValueError`
naming the log, rather than being NaN-filled, since that would hide a workspace
with no positions at all. No `sample_logs.py` change was needed.

`TestNonFiniteCoordinatesAreAsked` covers NaN, `inf`, the finite case, and the
missing-log case, and its last test states the property directly: the NaN path
must survive a reworded assertion.

**O6 — the full `detector_counts` read.** Measured before changing it, on a
64 MB dataset with a 5% row selection:

```console
  full .nxdata then mask     peak=   67.2 MB   result=(20, 20000)
  field[mask] then .nxdata   peak=    3.2 MB   result=(20, 20000)
```

So the selection does reach h5py — a 21x reduction, and the saving scales with
how small one workspace is relative to the entry. `readSubruns` now indexes the
`NXfield` with `np.flatnonzero(rows)` before `.nxdata`.

`TestRowSelectionHappensInH5py` pins what can silently go wrong in that swap: an
index list must pick the same rows as the boolean mask, in the same order,
**including when the selected rows are not contiguous**.

**O7 — the `break`.** Now `next(iter(ws._raw_counts), None)` with the loop gone,
which says "the first scan point is representative" in the code rather than
leaving it to be inferred from a loop that exits on its first iteration. The
`None` case — a workspace carrying no counts — is stepped over explicitly and
covered by `TestPixelCheckHandlesAWorkspaceWithNoCounts`; the previous `for`
simply never entered, which worked but by accident.

### Verification

507 unit (was 443 at the start of batch 6) / 104 integration / 16 GUI; `ruff` and
`mypy` clean; probe sweep 19 OK, 1 SKIPPED (retired), 0 FAIL.

---

## Batch 7 — `pyrs/resources/application.yml`, `pyrs/utilities/config.py`

Keys, then their reader. +18 and +35 lines — the smallest batch so far, and the
one where every later module's behaviour is decided.

### D25 — closes D5, by disproving it

I opened this batch by re-raising D5 as a whitespace defect and measuring it:

```console
fields after override : [' direction ']
validate_config()     : ACCEPTED
field_names()         : (' direction ',)
column_name()         : '__20direction__20'
write                 : ValueError: Sample log name  direction  is not found in
                        allowed value list ['vx', 'vy', 'vz', ...]
```

**Reviewer:** *"This isn't quite right. The issue has not much to do with
whitespace. Any field-name that is not an identifier would trigger this. Is the
correct solution to not allow non-identifier names to be used as discriminator
columns?"*

The first half is right and the second does not hold. Measured with a realistic
non-identifier PV log name:

```console
encode/decode round trip: True -> BL7__3ACS__3ASample__3ADir
on-disk column(s)       : ['BL7__3ACS__3ASample__3ADir']
read back               : 2 workspace(s), ['11', '22']
```

`BL7:CS:Sample:Dir` writes, splits and recovers correctly. Requiring
`column_name(n) == n` would **reject that**, which is the case the `__` escape
introducer exists for (Decisions row 23) — a PV log would be usable as a
retained log in `logs/*` but not as a discriminator, for no reason the format
imposes.

**What actually failed** is that no property or sample log of that name exists.
`direciton` — an ordinary misspelling, and a perfectly good identifier — fails
identically. Whitespace was incidental and the identifier distinction is a red
herring; I carried batch 2's framing forward instead of testing it.

**And it cannot be checked at configuration time.** Whether a field resolves
depends on the workspace, which does not exist when the config is read.
`field_names()` already does everything that *is* checkable there: blank,
duplicate, reserved-column collision, and an unrestorable property.

**So the defect is diagnosis, not acceptance**, and the fix is in
`_discriminator.resolve`, where the failure occurs and the context exists. It now
reports the configuration key, quotes the name with `repr` so whitespace is
visible without anyone guessing at it, says which of the two lookups were tried,
notes when the name matches a *method* rather than a property, and lists the logs
that are present.

**Not added, at the reviewer's direction:** a "did you mean" suggestion.
*"Every time I've seen that attempted, the developers get it wrong, so it ends up
being extra code that isn't helping anyway."* Quoting the configured name exactly
shows the whitespace case without guessing at intent.

One implementation note: `resolve` tests `name in ws.sample_log_names` rather than
catching `get_sample_log_value`'s `ValueError`. The other failure that method
raises — a log varying across scan points — is an `AssertionError` that must keep
propagating, and re-describing one of two failures by catching an exception type
means depending on which is which. Same reasoning as O5 in batch 5.

Pinned by `TestUnresolvableFieldIsReportedAgainstTheConfiguration`, whose
docstring records that a non-identifier name is legal, so the disproved premise
is not re-adopted later.

### D26 — `validate_config`'s docstring claims more than it checks

> *"Raise RuntimeError unless **every typed key** holds the type it declares"*

It checks four of nine. Measured:

```console
instrument_name       : 123     -> validate_config() ACCEPTED
use_production_names  : 'yes'   -> validate_config() ACCEPTED
```

Severity differs per key, which is the useful part:

- `instrument_name` / `instrument_short_name` **are** defended at the point of
  use — `_Instrument._instrument_names` checks `isinstance(value, str)`, warns,
  and falls back to `"HB2B"`. The gap is real; the docstring is what is wrong.
- `extension` is unchecked and reaches `impose_extension` in three viewers, so a
  non-string there is a user-visible filename failure.
- `use_production_names` has **no Python consumer at all** — `grep` finds it only
  in plan documents. Declared, documented, unvalidated and unread, so a typo in
  it stays silently inert until spec 10 flips it.

**Fixed — the validator now covers every key the file ships, from a schema.**

First attempt used four hand-maintained tuples (`BOOL_KEYS`, `TEXT_KEYS`,
`EXTENSION_KEYS`, `LIST_KEYS`) combined into `VALIDATED_KEYS`.

**Reviewer:** *"As it stands, 'VALIDATED_KEYS' is a bit messy. Could
'VALIDATED_KEYS' be specified somehow using a schema for the YAML config file?"*

It can, and it should: a key list maintained *beside* the thing it describes is
what decayed in the first place. `ConfigSchema` is now a pydantic model
mirroring `application.yml` -- one section model per top-level section, one
field per key -- and `VALIDATED_KEYS` is **derived** from it by walking
`model_fields`.

**`strict=True` is load-bearing, not tidiness.** Pydantic's default would have
made this worse than the hand-rolled checks:

```console
  Lax     flag='false' -> False      <-- the exact defect, now a silent conversion
  Lax     flag='yes'   -> True
  Lax     flag=1       -> True
  Strict  flag='false' -> rejected
```

A permissive schema would have turned the check into a coercion. Measured on
pydantic 2.13.5 and recorded in the model's comment.

Errors are rendered from `ValidationError.errors()` into one `RuntimeError`
naming each offending key, so a config with two mistakes reports both -- pinned
by `test_every_offending_key_is_named_at_once`. The cross-section rule ("at
least one output enabled") stays outside the schema: it is a rule *between*
sections, and a caller catching it should not have to dig through a
`ValidationError` to learn which rule failed.

**A schema introduces its own silent-pass mode, and it bit immediately.**
`ConfigSchema` was first written with only a docstring and no fields. A pydantic
model with no fields validates anything, so `VALIDATED_KEYS` came back empty and
`validate_config()` passed vacuously -- caught only because the measurement
printed the key list rather than just the verdict. Now asserted directly by
`test_the_schema_describes_some_keys_at_all`.

**A blank `instrument_name` is now rejected at import**, where it previously
warned and fell back to `"HB2B"` at the point of use. The fallback stays as a
backstop for a config-read failure; silently writing `HB2B` into a file prepared
for another beamline is the outcome worth preventing.

### Observations

### D27 — the configured extension carried its separator

`application.yml` shipped `extension: ".nxs"`, and the first version of the
schema *required* the leading dot, on the grounds that `Path.with_suffix` raises
`ValueError: Invalid suffix 'nxs'` without one.

**Reviewer:** *"Please do not put '.' in the extension; this needs to be
implicit."*

Right -- the dot is a `pathlib` separator, not part of the extension, and
leaving it in the value means a reader cannot tell which. The configuration now
reads `extension: "nxs"` / `"h5"`, the schema **rejects** a value that carries a
dot (naming the value to write instead), and `impose_extension` supplies it:

```console
  impose_extension("run.dat", 'nxs' ) -> 'run.nxs'
  impose_extension("run.dat", '.nxs') -> 'run.nxs'
```

It still tolerates a dotted argument, because it is also called with literals
elsewhere; only the *configured* value is constrained. All five call sites pass
`Config[...extension]`, and every other `.nxs`/`.h5` in the viewers is a
hard-coded literal unaffected by the change.

**O8 — `validate_config` is never re-run after a runtime override.** Deferred, as
**DD-1** in [deferred-defects.md](deferred-defects.md): the gap is in
`neutrons_standard.Config`, which offers no validation hook on `refresh`/
`loadEnv` and whose `_Config.validate` is a documented no-op. Every PyRS-side
workaround is worse than the gap — scattering `validate_config()` across
consumers, wrapping `Config` in a second accessor the module docstring forbids,
or re-validating some keys at some call sites, which reads as coverage and is
not. Tolerable because the runtime-override path is test-only by decision (batch
2); what is missing is that nothing says so.

**O9 — the `discriminator_fields` comment was stale. Fixed.** It now states that
a name matching a `HidraWorkspace` property must match a **restorable** one
(`pyrs.utilities.restorable_property`), and why a plain read-only property is
refused when the configuration is read: its value would be written to the file
and silently lost on the way back, with the reconstructed workspace answering
whatever its constructor defaulted to. It also records that a non-identifier
name is legal and encoded reversibly — the premise D25 disproved, now stated
where a user configuring the key will actually read it.

### Verification

526 unit (was 507) / 104 integration / 16 GUI; `ruff` and `mypy` clean.

---

## Batch 8 — the model classes

Four files, not the three Appendix A names: `strainstressviewer/model.py` joined
during batch 2. 14 insertions, 5 deletions — the smallest batch in the walk, and
the only one that touches what a user clicks.

Three of the four are purely 04b's signature change. The defects are in what that
change had to assume.

### D28 — both reading viewers silently discarded everything but the first workspace

```python
# A viewer holds one workspace; an entry written by this viewer holds one.
wss, peakss = nx.read()
ws, peaks = wss[0], peakss[0]
```

The comment's second clause is true of files *this viewer wrote*, and a user may
open any file. Measured:

```console
entry holds 2 workspace(s): ['11', '22']
the viewer takes wss[0] -> '11'; the rest are discarded silently
```

A multi-workspace entry — what 04b exists to produce, and what every append
creates — loaded with no error and part of its content missing. The next save
would have made that permanent.

### D29 — and the same for entries

Neither reader passed `entry_number`, so `read()` took entry 1:

```console
file holds entries: ['entry', 'entry_2']
the viewer calls read() with no entry_number -> entry 1 only
```

Same silent truncation, one level up, and 04c made multi-entry files routine to
write.

### Resolution — refuse by name, and leave the application usable

**Reviewer:** *"Hopefully for all of these changes we have the option to make the
user aware of what's going on, without terminating their action possibilities
(i.e. allowing them to continue)?"*

New `pyrs/interface/nxstress_load.py`: `read_single_workspace(path)` and
`UnsupportedNXstressFile`. It checks the entry count *before* reading and the
workspace count after, and the message names what the file actually holds and
states that nothing was loaded.

Refusing rather than loading part of the file is the deliberate half: a partial
load can be fitted and saved back, and the remainder would be gone. Nothing is
loaded, so nothing is lost, and the user can pick another file.

`UnsupportedNXstressFile` subclasses `RuntimeError` specifically so it lands in
the viewers' existing "unable to load" handling — pinned by a test, since that
inheritance is load-bearing and not obvious.

**Two things had to be fixed for "aware but not terminated" to be true:**

- `peak_fitting_viewer.load_and_plot` caught the failure, popped a dialog, and
  then **fell through** to code assuming a loaded workspace — turning a reported
  failure into a second, unreported one against whatever was loaded before. It
  now returns. Pre-existing, and reachable far more often now.
- `texture_fitting`'s load path had **no error handling at all**: a raise escaped
  into Qt, leaving a traceback on stderr and a window that looked as though
  nothing had happened. It now pops the same dialog and returns, matching
  `loadRunNumber` directly above it.

`NXstress.entry_numbers()` is new and public, extracted from the scan
`_resolveTarget` was doing inline: a caller that can present only one entry has
to be able to ask how many there are.

### Observation 3 — direction mismatch now raises

`strainstressviewer` records the user's direction choice on the workspace, and
the project file may record one of its own. If they disagree, one is wrong and
neither the model nor the viewer can tell which, so it is reported instead of
silently preferring either. The log is read directly rather than through
`ws.direction`, which would answer with the value just restored.

This model already had the shape the other two needed — `except` →
`failureMsg.emit(...)` → return `(None, {})` — so raising gets the user a message
and leaves the viewer usable, with no new handling.

Three tests cover it: disagreeing, agreeing, and **no `direction` log at all**,
which is the ordinary case for a Hidra project file and must keep working.

### Stale comment — fixed

`combine_runs_model.py` justified its empty peak list by citing
`_Peaks.init_group([], ...)` and `_Fit.init_group(ws, [], ...)`, both pre-04b
signatures. The reasoning held; the symbols did not. It now describes what
actually happens, including that such an entry records the `UNDEFINED_PEAK_TAG`
sentinel and therefore cannot be appended to — the branch batch 6 found in
`_rejectFitModelMismatch`.


### `entry_numbers()` — why a method, and why both refusals sit inside the `with`

Raised by the reviewer after the fix landed, as a design-consistency question:
*"How does the following method work, design-consistency wise, with the NXstress
interface, which is generally accessed solely in terms of its context manager?"*

It is not an exception to that interface. It reads `self._root`, which only
`__enter__` sets, and carries the same guard and message as `read` and `write`:

```console
inside the context : (1, 2)
outside it         : RuntimeError: Usage error: only usage as context manager is supported!
```

It exists because `read`/`write` already *take* an `entry_number` and nothing
exposed which numbers are valid — so it completes an existing parameter rather
than introducing an idiom. The scan was already there, inline in `_resolveTarget`.

**Not a property**, on the measurement rather than on taste: the answer changes
within one instance's lifetime, so attribute syntax would imply a stability it
does not have.

```console
entry_numbers before write: (1,)
entry_numbers after  write: (1, 2)
```

Two lesser reasons point the same way — it raises outside the context, and a
property that raises makes an object awkward to inspect in a debugger or `repr`;
and `read`/`write` are methods, so a property would read as a different category
of thing.

**The question also exposed a documentation gap.** `entry_numbers` had `Returns:`
and no `Raises:`, so nothing stated the context requirement — and `read()` and
`write()` had the **same omission**, both raising that `RuntimeError` first thing
and documenting neither it nor (for `write`) the already-invalidated case. All
three now state it.

**And one about access.** Only the first of the helper's two refusals needs the
open file: everything `read` returns is detached, because every read path ends in
`.nxdata`, which materialises into numpy — measured by reading sub-runs, raw
counts, diffraction and peak parameters after the close.

**Reviewer:** *"Accessing within the with block is more consistent in any case,
regardless of whether or not we actually need to do that."* Taken as the rule:
the access *pattern* is the thing to be consistent about, not whether an
individual statement strictly requires the file. Both refusals sit inside.

`TestReadReturnsObjectsDetachedFromTheFile` pins the detachment anyway, since
the helper and both viewers depend on it and `NXstress` nowhere states it — and
the read path does index `NXfield`s lazily before materialising, so the
assumption is not idle.

### Verification

526 unit / **112 integration** (was 104) / 16 GUI; `ruff` and `mypy` clean. The
seven new integration tests are `test_nxstress_load.py` (4) and
`test_strainstress_direction.py` (3); both are integration by `CLAUDE.md`'s
dividing line — interface code driving a second component — despite synthetic
fixtures under `tmp_path`.

---

## Batch 10 — `tests/unit/pyrs/utilities/NXstress/conftest.py`

The fixtures everything else uses. 18 insertions, 4 deletions; the second hunk is
pure reformatting of `minimal_PeakCollection`'s signature, no behaviour.

`minimal_HidraWorkspace` gains a `sub_runs` parameter overriding the default
`1..N`, for a stated reason: two workspaces sharing one entry must cover disjoint
scan points, and the interesting case — values that *interleave*, `[1,3,5]`
against `[2,4,6]` — cannot be built from `1..N` at all. That is the shape probe
`a5_subruns_nonmonotonic.py` and Decisions row 31 rest on.

### D32 — the last implicit `Optional` in the repository

`sub_runs: np.ndarray = None`. Same defect class as 04b's Follow-up 6 F6.2
(`_peaks.indexed`'s `logs: SampleLogs = None`). `mypy` sees it perfectly well
when pointed at the file — nothing runs `mypy` over `tests/`. Swept: this was the
**only** remaining instance across `pyrs/` and `tests/`. Now
`np.ndarray | None = None`.

### D33 — an explicitly passed `n_subruns` was silently discarded

```console
n_subruns=6 AND sub_runs=[1,2,3] -> [1, 2, 3]
```

The comment said `sub_runs` "overrides the default `1..N`", which is true of the
*default*; it also overrode an explicit `n_subruns`, with no indication, in a
fixture used by essentially every NXstress test. A test reading
`minimal_HidraWorkspace(n_subruns=6, sub_runs=(1, 2, 3))` asked for six and got
three.

Now refused. `_DEFAULT_N_SUBRUNS` is a named constant so the fixture can tell
"passed 3 explicitly" from "did not pass it":

```console
sub_runs only        -> [1, 3, 5]
n_subruns only       -> [1, 2, 3, 4]
both, agreeing       -> [2, 4, 6]
both, disagreeing    -> ValueError: n_subruns=6 disagrees with sub_runs=[1, 2, 3] (3 values).
```

### O10 — the sub-run axis was assigned in three places **(outside this PR's scope)**

Found via the new parameter: a non-monotonic `sub_runs` produced
`"Cannot change subruns when non-empty (previous=[1 2 3], new=[3 1 2])"` on a
freshly built workspace, which is not what went wrong.

Three sites assigned `self._sample_logs.subruns`, disagreeing about sorting:

| site | assigned | effect |
|---|---|---|
| `HidraWorkspace.set_sub_runs` | `sorted(sub_runs)` | silently reorders |
| `HidraWorkspace.set_sample_log` | `np.atleast_1d(sub_runs)` | caller's order |
| `HidraWorkspace.load_hidra_project` | `hidra_file.read_sub_runs()` | file's order |

with `SubRuns.set` beneath them *refusing* non-increasing input. One caller
pre-empted that guard by sorting, the others did not, and the disagreement
surfaced as the wrong error — by the time the second assignment ran, the first
had already happened, so it reported an attempted *change*.

**Two options were weighed.** Deleting the sort would centralise on `SubRuns.set`,
the layer that owns the invariant and states it in its own words — but would
raise for any caller currently relying on the sort, and such callers are exactly
the ones a test suite might not reach. **The stakeholder chose the safer option:**
*"It doesn't completely fix the issue, but it's less likely to break code that we
might not see."*

**And the sort cannot be unconditional**, which is what shaped the helper.
`set_sample_log` pairs its values with `sub_runs` **positionally**, so sorting the
axis alone reattributes every one of them, silently, because the lengths still
agree:

```console
axis as given    : sub-runs [3, 1, 2] <-> vx [30.0, 10.0, 20.0]
axis sorted only : sub-run 1 -> 30.0, which was measured at sub-run 3
```

So `_set_subruns(sub_runs, *, sort: bool)` is the single assignment point, and
`sort` is a **required keyword with no default** — writing it out at each call
site is the point, since the three callers genuinely differ. The measurement
above is in its docstring, as the reason.

This touches `pyrs/core/workspaces.py`, a core data class outside the plan
series, and is called out in the commit message accordingly.

### Verification

526 unit / 112 integration / 16 GUI; `ruff` and `mypy` clean; probe sweep 19 OK,
1 SKIPPED (retired), 0 FAIL.

---

## Batch 11 — `tests/util/peak_collection_helpers.py`

The other shared helper. 14 insertions, 1 deletion: `createPeakCollection` gains
`sub_runs` (overriding the default `1..N`, for the same disjoint-coverage reason
as batch 10) and `mask`, exposing `PeakCollection`'s own argument — which is what
let batch 5's texture tests build a collection keyed to an eta reduction.

### No defects in the change

It is the sibling of batch 10's, and it got the contested case **right**: it
raises when `sub_runs` and `N_subrun` disagree, where `minimal_HidraWorkspace`
silently discarded `n_subruns`. Two helpers, one PR, same conflict, opposite
handling — D33 brought the workspace one into line with this one rather than the
reverse.

Two things checked rather than eyeballed:

- **`HidraConstants.DEFAULT_MASK` and NXstress's `DEFAULT_TAG`** are both
  `'_DEFAULT_'`, but not duplicated: `_definitions.py:336` is
  `DEFAULT_TAG = HidraConstants.DEFAULT_MASK`. Derived, so no drift — worth
  confirming, since batch 5's whole mask-namespace separation turns on that
  string.
- **What the parameter accepts.** The guards are tighter than the explicit check
  suggests, because `SubRuns` catches the rest, and *names the real problem*:

```console
ascending  [1,3,5]   -> [1, 3, 5]
descending [5,3,1]   -> RuntimeError: subruns are not sorted in increasing order
duplicates [1,1,1]   -> RuntimeError: subruns are not sorted in increasing order
length mismatch      -> ValueError: sub_runs has 2 entries but N_subrun is 3
```

Contrast O10, where the same class of input produced *"Cannot change subruns when
non-empty"*. Identical underlying guard; legible here only because nothing
pre-empts it by assigning a sorted copy first.

### The abandoned `permutations` block — deleted

A triple-quoted **string literal** (not `#` comments, which is why `islice` and
`permutations` appear in no import and `ruff` said nothing) sat above the new
code:

```python
all_runs = [n for n in range(3 * N_subrun)]
subruns = next(islice(permutations(...), 2 * N_subrun, 2 * N_subrun + 1))
```

Asked whether it was a preserved alternative. It is not — it is an **earlier
attempt at exactly what the new parameter does**, abandoned at the point of
hitting the increasing-order constraint, with the author's note directly beneath
it: *"Assume subruns are supposed to be in order. Why would that be the case?"*

It would have run, but only by luck. At the chosen index it is increasing for
every realistic `N` — because `2*N` is still near the start of the lexicographic
sequence, where only the last element varies — and one step further it is not:

```console
N=3 ... N=8 at index 2*N : increasing? True   (always)
index 6: (0, 1, 8)       : increasing? True
index 7: (0, 2, 1)       : increasing? False  <-- SubRuns.set would raise
```

It also yields one deterministic value per `N` despite the comment saying
"random", and `all_runs` is assigned and never used. Deleted.

**The author's question now has an answer, recorded in its place:** sub-runs must
be strictly increasing because `SubRuns.set` enforces it — duplicates included,
since it tests `value[:-1] < value[1:]` — and the reason *that* rule exists is
that every per-scan-point array is paired with the axis positionally, so a
different order reattributes values silently rather than failing (measured in
O10). The comment points at where it is enforced, and says why no check is
repeated in the helper.

### O11 — mixed annotation style, fixed

`peak_tag: str` and `N_subrun: int` were annotated; `sub_runs` and `exclude_list`
were bare. Not an implicit `Optional` — `mypy` has nothing to object to without
an annotation — but `sub_runs` is the parameter D32 had just annotated in the
sibling fixture, so the two differed for no reason. Both are now
`Sequence[int] | None`.

### Verification

526 unit / 112 integration / 16 GUI; `ruff` and `mypy` clean.

---

## Batch 13 — `tests/unit/pyrs/utilities/NXstress/test_discriminator.py`

Pairs with batch 2. Entirely new: 315 lines, nothing deleted.

Three things in it are better than routine, and worth naming because they are
the shapes this review has been asking for elsewhere.
`test_reserved_columns_matches_peaks_init` asserts the guard's reserved set
equals what `_Peaks._init` actually writes, by iterating the real group rather
than keeping a second list. `test_only_a_reserved_name_encodes_onto_a_reserved_column`
pins *why* checking the encoded name suffices — it only catches everything
because the encoding is injective. And `TestEveryPropertyRoundTripsOrIsRefused`
sweeps `vars(HidraWorkspace)` with a guard against a vacuous pass.

### D34 — a docstring naming the same fixture twice

```python
"""Goes through `config_override` rather than `config_override`: the latter swaps
the `Config` singleton, ..."""
```

Both halves say `config_override`, so the sentence conveys nothing — and what it
is warning about is the batch 2 finding that cost real time to diagnose: a module
binding `Config` the documented way keeps reading the previous instance after a
*reload*. "The latter" is `default_config`. Corrected, and the mechanism spelled
out rather than left to "see that fixture's docstring".

### O15 — the partition was claimed exhaustive and not asserted

The class docstring said *"The partition is exhaustive: every property must be in
exactly one branch."* It was not quite: the refusal branch excludes a property
whose **encoded** name collides with a reserved `NXreflections` column, because
`field_names` checks that collision first and raises a differently-worded error:

```console
'mask'   Discriminator field 'mask' collides with the reserved `NXreflections` column
'name'   Discriminator field 'name' is a read-only `HidraWorkspace` property, so its...
```

Both are refusals; only the wording differs. But the exclusion is a third bucket
that no test covered, and measured it is **currently empty only by luck** — all
seven properties classify cleanly today.

Now three shared classifiers (`_accepted`, `_refused`, `_reserved_named`) with
`test_the_partition_covers_every_property` asserting they account for every
property with none counted twice, and `test_no_property_is_named_like_a_reserved_column`
asserting the third stays empty — with a message telling a future reader to add
a case rather than widen the exclusion. The two existing branch tests now use the
same classifiers, so the partition cannot drift from what they test.

**Confirmed able to fail**, by subclassing the test with a `ghost` property the
branches do not classify: the partition assertion breaks, as it must.

### O16 — six unused `tmp_path` parameters

`test_default_is_empty(self, config_override, tmp_path)` and five others never
used it, implying a filesystem dependency that is not there. Removed.

### Verification

532 unit (was 530) / 112 integration / 16 GUI; `ruff` clean.

---

## Batch 14 — `test_peaks.py`, `test_peaks_read.py`

Pairs with batch 3. 422 insertions across the two.

### No defects in the code under test

`test_peaks.py` replaces the `sx`/`sy`/`sz` tests that pinned **NaN
placeholders** with ones asserting the fields carry `PointList.(vx, vy, vz)` and
checking `local_name` provenance, plus a subset case covering the defect that
kept the block commented out — reading the log wholesale gives one value per
*workspace* scan point where the slice needs one per *collection* scan point.
`TestPeaksTailAppend` adds per-family append coverage including that an appended
`Al` block stays *after* `Fe` rather than re-sorting into it.

`test_peaks_read.py` promotes `TestSplitterEnforcesOnlyContiguityAndMonotonicity`
from probe `a5_peakcollection_ranges.py`, so the premise both 04b and 04c rest on
is re-checked per commit rather than per audit, and adds
`TestDiscriminatorNamesAreCheckedAgainstTheGroup` for the batch 3 defect where
omitting the names returned 1 range where 2 were correct, **with no exception**.

Two pieces of judgement worth recording as right: `expected_block_count` restates
the block rule locally rather than importing it, with the reason given — a test
that asks the module for its own rule agrees with whatever rule it later adopts.
And the final test is careful that one range for an undiscriminated group is
*not* "merging" in any NXstress sense, the distinction corrected during batch 3.

### O17 — the source-scan test flagged its own explanatory comments

`test_no_global_ordering_machinery_in_the_module` asserted
`source.count(name) == 0` for four sort-related words. The claim holds, but raw
text cannot tell code from prose:

```console
after adding a COMMENT containing the word: {'searchsorted': 0, 'argsort': 1, ...}
```

A maintainer writing *"deliberately no argsort over the index here"* — the
comment this test invites — would break it, so it punished the documentation
written because of it. Now `ast.parse` plus a walk over `ast.Call`, matching
`tests/unit/pyrs/projectfile/test_file_object.py`, which already uses that idiom.
Verified against all three cases:

```console
real module                    -> clean
+ a COMMENT mentioning argsort -> clean
+ a real np.argsort CALL       -> {'argsort'}
```

`test_the_scan_would_notice_such_a_call` was added alongside: the subject is the
*absence* of something, so a walk that silently matched nothing would pass for
the wrong reason.

### O18 — `sorted()` omitted from the scan, deliberately but silently

`_peaks.py` calls `sorted(` four times: `init_group` sorts the incoming batch,
which is not a search over the on-disk index and is what makes each written block
contiguous to begin with. Correct to exclude, but it read as an oversight beside
four sort-related words that *are* listed. The call set is now a named constant
with that reasoning attached.

### O19 — the splitter stand-in was a bare `dict`

`peaks_group()` assembled six `NXfield`s into a dict. It worked, because the
splitter only subscripts and iterates — but it carried 6 of the group's 15
columns, so it diverged from the thing it stood for and would fail with a bare
`KeyError` the moment the splitter read a seventh. `probes/README.md`'s own
warning applies: *a test that reimplements its counterparty is testing the
reimplementation*.

Now built through `_Peaks._init` — the real writer — and then filled by hand,
since the point is to present orderings the writer would never emit. The sibling
class in the same file already did it this way.

### Verification

533 unit (was 532) / 112 integration / 16 GUI; `ruff` clean.

---

## Batch 15 — `tests/unit/pyrs/utilities/NXstress/test_fit.py`

Pairs with batch 4. 336 insertions: `TestFitTailAppend` (per-family append
coverage for the two parameter groups and the diffractogram),
`TestReducedDataRowCountIsGuarded` (batch 4's D9) and
`TestTextureWorkspaceRoundTrip` (batch 5's D13).

### D35 — the "five paths" claim is wrong, in two places

The docstring whose whole job is to justify **why two guards are needed** says
`set_reduced_diffraction_data_set` "is only one of five paths that write
`_2theta_matrix`; the other four bypass it", and lists a fourth bullet: *direct
assignment to `_2theta_matrix`*.

The first three check out — `workspaces.py` lines 204, 271 and 983/1051, with
real callers in `reduction_manager` (x2) and `mantid_peakfit_calibration`. The
fourth does not: **no production code outside `workspaces.py` writes that
attribute at all**, and `_fit.py` only reads it. The sole direct assignments are
in `test_fit.py` itself (lines 265 and 742) -- the helper that simulates the
bypass.

So it is four writers and three bypass paths. The same wrong count is duplicated
in production, at `_fit.py:449`.

The argument survives: three bypass paths still make the second guard necessary.
But a count stated twice and wrong twice, in the text justifying a duplicated
guard, is how the guard gets removed later. Both corrected, and the test
docstring now says explicitly that `_inconsistent` assigning the attribute is
*the test*, not a fifth production path.

### O20 — an assertion that cannot fail

```python
indexed = _Peaks.indexed([], ws._sample_logs)
with pytest.raises(RuntimeError, match=r"... has reduced data for 5 scan"):
    _Diffractogram._concatenated_diffraction([ws], DEFAULT_TAG)
assert indexed == []
```

`indexed` is never passed to the call under test, and `_Peaks.indexed([], ...)`
returns `[]` by construction -- the assertion tests the fixture. Third instance
of this shape in the review, after O12 and the `argsort` text scan. Removed,
along with an unused `createPeakCollection` fixture in the same signature.

### O21 — a private attribute poked where a public parameter now exists

Both texture tests did `peaks._mask = "eta_0.0"`. Batch 11 added `mask` to
`createPeakCollection` -- partly for this -- and `minimal_PeakCollection`
forwards `**kwargs`, so the supported route gives the same result:

```console
createPeakCollection(mask=...) -> 'eta_0.0'
pc._mask = ...                 -> 'eta_0.0'
```

Equivalent today, but the assignment bypasses whatever `PeakCollection.__init__`
may later do with the argument. A third instance at line 497 (pre-dating this
PR) was converted with them, so nothing in the suite now reaches into `_mask`.

### Verification

533 unit / 112 integration / 16 GUI; `ruff` and `mypy` clean.

---

## Batch 16 — `test_input_data.py`, `test_instrument.py`, `test_sample.py`

Pairs with batch 5. The diff against the PR base is 593 lines, but **375 of
those were added by this review** in batches 5, 10 and 11 and were reviewed as
they were written. The PR's own contribution is 219 lines, and that is what was
walked (`git diff eb5457b1 cda22352`).

### Two judgement calls that were already right

`test_instrument.py` corrects an assertion that was wrong in an instructive way.
`detector["depends_on"]` was expected to be `translation_x`, the **first** link;
traversal follows each field's own `depends_on` back to `"."`, so the entry point
must be the **last**. As written, the chain terminated immediately and seven
transformations were written but unreachable -- with the test asserting that
state was correct.

`test_sample.py`'s identifier expectation is spelled out as the literal
`HB2B__3ACS__3ACustomValue` rather than computed from `allowed_identifier`, with
the reason given: the test should fail if the encoding changes rather than
silently track it. Same reasoning as `expected_block_count` in batch 14.

### D36 — two tests stubbed `Config` with a dict

```python
monkeypatch.setattr(
    _instrument_module,
    "Config",
    {"nxstress.instrument_name": "HB2A", "nxstress.instrument_short_name": "PG3"},
)
```

The only two such monkeypatches in the suite. They replace the module's `Config`
binding with a plain dict, so `Config[key]` becomes a dict lookup: the real
`__getitem__` and the real deep-merge are never exercised, and a stub has no
"unset" state to test against.

Both now use `config_override`, and a third test --
`test_Instrument_name_defaults_when_not_overridden` -- becomes expressible.
Together the three show the value *tracks* configuration, which no one of them
shows alone. `_instrument.py` imports `Config` the documented way, and a real
override reaches it:

```console
a REAL config override reaches _instrument.py -> ('HB2A', 'PG3')
```

**A correction to my own justification, recorded because the overstatement was
in a comment that shipped before it was checked.** I claimed this would catch
`_instrument.py` reverting to the forbidden `from neutrons_standard.config
import Config`. Mutation says otherwise: the mutant **passed**. Both names reach
the same singleton once `init("pyrs")` has run, and `config_override` mutates
that instance, so the tests are blind to the import path. Probing it directly
reproduced the hazard the module docstring describes --
`ModuleNotFoundError: No module named 'None'` -- which is the point: that import
is a load-*order* fault, raising only when it runs before `init()`. It is
guarded by import order and by `config.py`'s docstring, not by these tests. The
comment in the test now says so.

### Verification

534 unit (was 533) / 112 integration / 16 GUI; `ruff` clean.

---

## Batch 19 — `tests/unit/pyrs/utilities/test_config.py`

Pairs with batch 7. The PR added **74 lines**; the other 222 in the current diff
were added by batches 2 and 7 of this review and reviewed as written.

`TestDiscriminatorConfigKeys` covers the two 04b keys through the real `env`-file
mechanism, and its docstring gives the reason the typing matters:
`discriminator_fields: direction` parses as a *string*, and a string is iterable,
so an unchecked value would be read one character at a time as five one-letter
field names.

### No defects

### O26 — seven redundant local `import pytest`

The module docstring states a local-import rule, but that rule is about
`pyrs.utilities.config` specifically: importing it at module level runs its side
effects (writing under `~/.pyrs/`) at collection time, before the fixture has
redirected `HOME`. It does not apply to `pytest`, and repeating it seven times
invites a reader to think everything in the file must be imported locally. The
docstring now says which import the rule is about; the seven are gone.

### O27 — two override mechanisms, both correct, with nothing saying when each applies

The file uses both, a few hundred lines apart:

| | reaches | use when |
|---|---|---|
| `default_config.loadEnv(path)` | `pyrs.utilities.config` only -- it **reloads** the singleton, so any module that bound `Config` the documented way still holds the previous instance | the override *file* is the subject |
| `config_override({...})` | every binding -- deep-merges into the live singleton without replacing it | some *other* module must read the override |

Both are right where they are used. The `validate_config` tests are correct to
use `loadEnv`: what a user can write in a `.yml` is exactly what is being
rejected, and `config_override` does not go through that path at all. They
survive the reload only because the module under test *is* the module being
reloaded.

**Choosing wrongly fails silently.** Point `loadEnv` at a test of
`_instrument.py` and it reads the stale instance, so the assertion runs against
the shipped default and proves nothing -- which is the defect batch 2 measured
and batch 16 fixed. Recorded in the module docstring with that consequence
spelled out, and in `TestDiscriminatorConfigKeys` with the condition that makes
`loadEnv` safe there.

### Verification

534 unit / 112 integration / 16 GUI; `ruff` clean.

---

## Batch 21 — `tests/integration/test_nxstress_viewer_roundtrip.py`

42 lines from the PR: one unpacking fix for 04b's list return, and one new test
covering the CombineRuns call-site decision.

### No defects

The new test's docstring is the valuable part. It records **why** CombineRuns
keeps its in-PyRS pre-merge rather than passing N workspaces to `write`:
`append_hidra_project` already discards per-run boundaries exactly as
`nxstress.merge_workspaces` would, and the unaffected `.h5` export path needs
the single merged workspace regardless. That is the resolved Q3, written at the
call site it governs.

### O30 — the test could not see a dropped run

Its stated point is *"three runs pre-merged into one workspace, not three"*, and
its assertions were:

```python
assert len(workspaces) == 1, "three runs pre-merged into one workspace, not three"
assert peakss == [[]]
assert len(workspaces[0].get_sub_runs()) == len(model._hidra_ws.get_sub_runs())
```

The first two hold however many runs survived. The third compares the read-back
count against **the model that produced it**, so dropping runs 2 and 3 leaves
both sides reading 3 and the test green -- "one workspace out" being exactly
what dropping two runs also produces.

Now counted from the inputs: `expected_subruns = sum(len(ws.get_sub_runs()) for
ws in runs)`, which is 9 for this fixture. **Confirmed able to fail** by
mutating `combine_project_files` to `break` after the first run:

```console
FAILED ...::test_export_project_files_wraps_the_merged_workspace_in_a_length_one_list
```

### O31 — `model._hidra_ws` private access

Reaching into the model's internals, and only needed because the expected count
came from the model. Counting the inputs removed both at once.

### Verification

534 unit / 112 integration / 16 GUI; `ruff` clean.

---

## Batch 25 — `04b-multi-workspace-nxstress.md` + `open-questions/04b-multi-workspace-nxstress.md`

The batch plan's Appendix A row 25 names these `04b-nxstress-multi-workspace.md`.
The real filenames reverse the middle terms. A third row-vs-reality mismatch,
alongside the two already recorded under "Corrections found while walking it" --
and the same lesson: a file list drawn from a diff goes stale, so each batch's
membership should be re-derived from the range when it opens.

### Coverage

A1, A2, A3, A7. **A4/A5 exempt** -- neither document asserts third-party
behaviour or a cross-module runtime contract not already covered by a landed
probe. Written down rather than left implicit, since afterwards "no probes were
needed" and "no probes were written" are indistinguishable.

All 61 code citations in both documents were extracted mechanically and then
each one opened and read. That combination is the point: the tool says which
citations exist and that they resolve, and only reading says whether the line
means what the prose says it means.

### D44 -- thirteen stale body pointers; two of them actively misleading

Every one into a function this PR rewrote *after* Follow-up 2's correction table
was written. Detail and the full table in 04b's **Follow-up 11** (F11.1).

The two worth separating: `workspaces.py:517` resolved inside `get_wavelength`
while attached to the `append_hidra_project` claim, and `_input_data.py:95-97`
resolved onto `appendableDatasets` while described as the exact-match reader.
**A citation that lands on something plausible is worse than one that lands on
nothing**, because following it does not feel like a failed check.

Follow-ups 1 and 2 were left untouched at the time, per the policy this
document set at :860-864. **That deferral was overturned in batch 26**: the
reviewer ruled that citation values are pointers, not beliefs, and so are not
subject to the append-only rule -- a stale value is worthless because it points
at something that makes no sense, and the previous value is in git history.
Follow-ups 1 and 2 were re-resolved in batch 26. Recorded as Decisions Log
item 41.

### D45 -- a claim that had come to assert the opposite of its referent

Both documents said `_peaks.py`'s docstring "currently states" global
lexicographic sorting and "should be corrected during implementation". The
correction had been made: that docstring now says a global sort is *not*
guaranteed, and cites this spec's own Decisions item 17 for it. The citation
resolved, passed `check_citations`, and contradicted the prose quoting it.
Recorded as F11.2.

### O34 -- not fixed, raised for the series

Two batches running, the finding has been stale line pointers, and the reason is
structural: `landing_trigger.py` flags *every* citation into a changed file, so
one genuine drift arrives indistinguishable from forty correct neighbours and the
list gets skimmed. A tool that diffed old-vs-new line content and reported only
citations whose target text actually moved would separate them. Toolkit work,
not this PR's.

### Verification

`check_links` 0 across 25 documents; `check_citations` 3 of 425 (known
`open-questions/06` residue, none here). Each corrected pointer re-resolved
against current source and asserted to contain the symbol its prose names -- 21
checks, all passing, one off-by-one caught that way. No code changed; tiers not
re-run.
