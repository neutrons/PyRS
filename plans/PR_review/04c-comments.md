# PR review — subspec 04c (NXstress append mode)

Review of `eb5457b1..cda22352`, per
[`plans/PR-review-process/review-process.md`](../PR-review-process/review-process.md).
This document also carries the work belonging to **no** subspec — commit `61389b28`,
the integration-test fixups — per §4 of that process.

Each entry: the comment, the agreed resolution, and the change actually made.
My summaries of the diff are not recorded (§6e); defects I flagged are, whether or
not they were acted on.

---

## Batch 1 — `pyrs/utilities/NXstress/_definitions.py`

### D1 — `tail_append`'s error named the dataset `'unknown'`

**Flagged by me.** The whole value of that message is naming which dataset cannot
grow, and `NXfield.nxpath` returns the literal string `'unknown'` for a field not
attached to a tree, so the message read *"dataset 'unknown' was written at a fixed
size"*. `.nxname` is `'unknown'` too — there is genuinely no name to report.
Unreachable from the append path, which is always file-backed; reachable from a unit
test or a future in-memory caller.

**Resolution:** fix.

**Change:** new `_definitions._field_label(field)`, used by all three `tail_append`
messages. Returns `dataset '<nxpath>'` when there is a path, and
``an unattached (3,) `NXfield` `` when there is not, so the message never presents
`unknown` as a name.

### D2 — `tail_append(field, <scalar>)` raised `IndexError: tuple index out of range`

**Flagged by me.** `np.asarray(5).shape` is `()`, which passes the trailing-axis
check vacuously (a 0-d array has no trailing axes to disagree about); `values.shape[0]`
then raises an `IndexError` naming neither the field nor the problem. No current
caller passes a scalar — an unguarded contract rather than a live bug.

**Resolution:** fix.

**Change:** a `values.ndim == 0` guard before the trailing-axis check, raising
`RuntimeError` that names the field and says to pass a 1-element array. The comment
at the guard records why it must precede the trailing-axis check rather than follow it.

### D3 — `growable` advertises flexibility the writer refuses

**Flagged by me.** `growable(rank)` makes *every* axis unlimited, so `diffractogram`
goes to disk as `maxshape=(None, None)` — telling any NeXus reader the two-theta axis
can grow, when `tail_append` and `_Fit.validateAppend` both refuse exactly that.

**Reviewer:** *"I think we need to give a meaningful error message when a 'trailing
axis' append is rejected? (Assuming we don't have one already.) Also, it wasn't obvious
if we could just have the non-appendable items (e.g. 'diffractogram') NOT use the
'maxshape=(None, None)' form — what would be the implication of this?"*

**On the error message:** one already existed, in `tail_append`, and
`_Fit.validateAppend` has a domain-specific one that fires first in the append path
(*"the incoming reduced diffraction for mask `_DEFAULT_` has 25 two-theta bin(s), the
target entry 20"*). `tail_append`'s was extended to say *why*, not only *what*: "Only
the first axis grows on an append; the trailing axes are fixed when the entry is
written."

**On pinning the trailing axes — probed, not reasoned about.** Pinning works and HDF5
enforces it: `maxshape=(None, 20)` writes, an axis-0 grow succeeds, and an axis-1 grow
is refused by the library (`NeXusError: Shape incompatible with current NXfield`).
**But `(None, None)` turns out to be load-bearing for `CHUNK_SHAPE` as written** —
h5py accepts a chunk wider than the data only when that axis is unlimited:

```console
  maxshape=(None, None)  chunks=(1, 100)  data (3, 20):  OK
  maxshape=(None, 20)    chunks=(1, 100)  data (3, 20):  ValueError:
        Chunk shape must not be greater than data shape in any dimension
  maxshape=(None, 20)    chunks=(1, 20)   data (3, 20):  OK
```

So pinning drags in a shape-aware `CHUNK_SHAPE`, and two datasets must stay unlimited
regardless: `DIFFRACTOGRAM/fit` and `fit_errors` are `(0, 0)` placeholders that **spec
09** resizes on *both* axes when it fills them in.

**Resolution: keep `(None, None)`, document why.** No second change to on-disk layout
in one PR, no scope growth inside a review.

**Change:** the `growable` docstring now records the asymmetry, the h5py constraint
that causes it, the three measured cases above, and the spec-09 placeholder reason —
stating plainly that the trailing axes are *"a chunking artifact, not a contract"* and
naming the three places that enforce the real rule.

### Noted, not a defect — `__` in a log name expands ~2.5×

**Reviewer:** *"a minor question about what actually happens if an incoming identifier
includes `__` — how would this actually be encoded?"*

Each `__` becomes `__5F_`: the first underscore is escaped because its successor is an
underscore, the second passes through. Round-trips exactly —

| input | encoded | decoded |
|---|---|---|
| `a__b` | `a__5F_b` | `a__b` |
| `__foo` | `__5F_foo` | `__foo` |
| `__` | `__5F_` | `__` |
| `HB2B__CS__X` | `HB2B__5F_CS__5F_X` | `HB2B__CS__X` |
| `a____b` | `a__5F__5F__5F_b` | `a____b` |

**The durable point is the expansion against the 63-character cap.** *n* consecutive
underscores become ~2.5× their length, so a name that is comfortably short can still
fail to encode. Not live — zero of the 185 real log names in `tests/data` need an
escaped underscore at all — and it fails *loudly*, with `allowed_identifier` raising a
`ValueError` that quotes both the original and the encoded form. Recorded because the
condition is not obvious from reading the encoder. **No change.**

---

## Batch 6 — `pyrs/utilities/NXstress/NXstress.py`

The orchestrator: 832 insertions, 83 deletions, the largest single diff in the PR.
Two defects, one of them found by the reviewer reading my own summary back to me.

### D17 — every texture append was refused, and the instance was poisoned with it

`_rejectMaskMismatch` was the last call site in the package feeding `_diff_data_set`
keys into `nxstress_mask_names`. Batch 5 split those namespaces and converted
`_fit.py`; this one was missed. Measured:

```console
incoming via nxstress_mask_names        : ['_DEFAULT_', 'eta_-5.0', 'eta_0.0', 'eta_5.0']
incoming via nxstress_diffractogram_keys: ['eta_-5.0', 'eta_0.0', 'eta_5.0']
on disk DIFFRACTOGRAM groups            : ['eta_-5.0', 'eta_0.0', 'eta_5.0']

APPEND: RuntimeError: ... reduced-diffraction masks do not match the target entry's.
```

The injected `DEFAULT_TAG` is a key the writer never emits for an eta-only
reduction. It fires in `_classifyAppend`, so the file is untouched — but it routes
through `self._reject`, so the instance dies too.

It bites **only** when nothing reduced under the bare default mask. Every one of
the ~40 tests in `test_append.py` reduces under `{None: ...}`, where the two
functions agree, which is why the module passed over it. `_Fit.validateAppend`'s
docstring already assigns the set check here, so the owner was right and only the
function it called was wrong.

**Fixed**: `nxstress_diffractogram_keys`, and the method renamed
`_rejectDiffractogramKeyMismatch` — the defect was a name that let the wrong
function look right. Three regression tests in `TestTextureEntryAppends`.

### D18 — the length heuristic rejected legal appends (reviewer-found)

I summarised `_perScanPointDatasets` as identifying scan-point datasets by length
rather than by a maintained list, and repeated its docstring's claim that
over-matching is "the safe direction, since the only consequence is demanding
resizability of something that didn't need it."

**Reviewer:** *"that almost certainly does not work; as a counter-example, consider
the case where we have a 3-unit array of directions … and also 3 scan-points …
Obviously there aren't going to be any new 'directions' appended, nor should this
be required!"*

Correct, and the claim is the defect. The swept set feeds a resizability
**requirement**, so a false match is a rejection, not a no-op:

```console
n_scan = 3;  43 dataset(s) swept in
  FIX  entry/instrument/masks/names   shape=(3,) maxshape=(3,)
APPEND: RuntimeError: ... 1 dataset(s) in this entry were written at a fixed size
        and cannot be extended: ['entry/instrument/masks/names'].
```

Three detector masks and three scan points. `masks/names` is entry-wide, is never
grown, and is *required* by `_Instrument.validateAppend` to stay constant — so the
sweep demanded growability of the one thing the design forbids growing, and blamed
an out-of-date writer for it.

The sweep **under**-matched too, which neither of us had noticed. The peak index is
one row per (compound key, scan point), so it equals `n_scan` only while each
workspace fits exactly one peak. Fit a second phase — ordinary multi-peak work —
and `peaks/*` and both parameter groups left the pre-flight entirely, restoring
exactly the mid-mutation refusal 04c's Follow-up 3 F3.1 had hoisted out.

**The tell was in the tests.** `TestWriterEmitsResizableDatasets` carries:

> `n_scan` must not collide with any entry-wide array length, or an entry-wide
> array would be swept in and wrongly demanded to be resizable. … 7 collides with
> none of them; 3 is also safe today but only by two of those being small.

The hazard was understood precisely enough to be designed around *in a test*, and
not reported as a defect in the code.

**Fix — option 2 of three, per the reviewer.** Each module declares the datasets it
writes row-aligned (`appendableDatasets`), and `NXstress._appendableDatasets`
composes them. Exact rather than heuristic *because the caller owns the group's
layout*: inside a group NXstress writes, a field is either row-aligned — and so
`growable` — or an entry-wide scalar. `_Instrument` enumerates explicitly rather
than walking, since its non-scalar fields are mostly entry-wide arrays, which is
how this started.

Declared set: **49** datasets, against 43 swept of which one was false and the peak
index was included only by coincidence. The collision case now appends.

Rejected alternatives: excluding `instrument/masks` by location (patches the
instance, not the class), and keeping the sweep but treating non-appendable as
"not in the family" (keeps the runtime check honest about foreign files, but can
silently excuse a real field that lost `growable`).

**The property given up, and where it went.** The sweep covered a newly added field
automatically. `TestAppendableDatasetsAreDeclaredByTheirOwner::test_every_growable_dataset_has_exactly_one_owner`
now states that directly — every dataset written `growable` is declared by exactly
one module, and every declared dataset is resizable on disk — which is a stronger
guarantee than the sweep gave, and is checked against a written entry rather than
at runtime.

### O1 — `@validate_call_` restored on `_validateWorkspaceAndPeaksData`

It had the decorator before 04b; `init_group` and `_init` kept theirs. Re-added and
measured rather than assumed: the full NXstress suite passes with it. Flagged as a
genuine question first, since 04c's Follow-up 3 had already recorded one
`@validate_call_` surprise — but as an undeclared absence it was indistinguishable
from an oversight, which it appears to have been.

### O3 — withdrawn

I raised that nothing checks the per-scan-point time arrays against the scan-point
count. `SampleLogs.__setitem__` already does:

```console
ValueError: Number of values (or value rows)[2] isn't the same as number of subruns[3]
```

Raised without checking the upstream guard.

### D19 — a string log's type depended on where the workspace came from (O2, resolved)

`_entryTimes` decoded every time value unconditionally inside a `try` that
catches only `ValueError` — the clause that substitutes `NO_LOG` for an
unparseable timestamp — so a `str` log raised `AttributeError` straight past the
fallback. On the append path `_appendEntryTimes` is the **first** mutation step,
so the instance was invalidated for what was only a type mismatch.

**Reviewer:** *"why is there any ambiguity causing O2 — there doesn't seem to be a
reason to allow both bytes and str — what's causing that issue?"*

Nothing normalised. The type was decided by provenance:

| source | string log value |
|---|---|
| `HidraProjectFile.read_sample_logs` — `data_set[()]`, straight from h5py | `bytes` |
| `_Sample.sampleLogsFromNexus` | `bytes`, except `name`/`chemical_formula` decoded one-off at `:473-483` |
| built in memory from a `<U` array | `str` |

So six policies downstream: four tolerant (`summary_generator`,
`peak_profile_utility`, `_discriminator._as_text`, `HidraWorkspace.direction`),
two not (`_entryTimes`, `_fit.py:662`). Even the NXstress conftest carries the
split — it `.encode()`s `start_time`/`end_time`/`Filename` with a comment saying
it must match h5py, then writes `direction` as a `<U` array four lines later.

**Reviewer's answer to "adjust both readers?":** *"it should be bytes when at
hdf5, but probably string, when in memory. What happens if we require that?"*

Measured, both directions, all three tiers:

| | `bytes` in memory | **`str` in memory** |
|---|---|---|
| unit | 2 failed | **passed** |
| integration | 1 failed | **passed** |
| nature | one real regression: the CSV exporter wrote `# string1 = b'a constant string'` | all 83 initial failures were the *same* defect — the unconditional `.decode()` |

The asymmetry decided it: writers that must encode are few and already do it;
readers that must decode are many and mostly untested. The `str` direction also
ratifies the two deliberate normalisations already in the tree rather than
reversing them.

**Implemented (option 1, per the reviewer), then corrected twice.**

**Reviewer:** *"those sections need to go through the same conversion mechanism —
it must be centralised."* So the conversion is one pair of functions,
`to_text`/`to_text_array` in `pyrs/utilities/convertdatatypes.py` beside the
existing `to_int`/`to_float`, called at the store boundary
(`SampleLogs.__setitem__`), at the HDF5 boundary
(`HidraProjectFile.add_sample_log`) and at the **nine** hand-rolled sites across
five modules. Those nine had drifted into four behaviours — some handling
`numpy.bytes_`, some not; some coercing the non-bytes branch to `str`, some
returning it untouched. `_definitions.as_text` is gone; `_discriminator._as_text`
delegates.

### D20 — the first version checked the wrong dtype, and every tier was green

I claimed the one-off decodes in `_sample.py:473-483` were redundant, having
removed them and watched 465 tests pass.

**Reviewer:** *"Wouldn't the 'nxdata' in the h5py be in bytes? Why did removing
those work?! Something is not working as expected!"*

Right on both counts. `.nxdata` is bytes — but it arrives as an **object** array
of Python `bytes`, and the check was `dtype.kind == "S"`, which catches only
fixed-width arrays. Object arrays (`kind == "O"`) passed straight through:

```console
AFTER the first "fix", a round-tripped workspace:
    start_time    array-dtype=object   first=bytes    b'2024-01-15T10:00:00'
    direction     array-dtype=object   first=bytes    b'11'
    SampleName    array-dtype=<U5      first=str_     'steel'
```

Only the two fields built via `np.array([b"steel"] * n)` — which *is* `|S` —
converted, which is exactly why removing their decodes appeared to work. Three
things to record:

1. **The green tiers proved nothing here.** They passed because the intolerant
   consumers had been made tolerant, not because logs had become text.
2. **The unit test passed because its fixture was wrong.** It built
   `np.array([b"11", ...])` — the dtype the literal syntax gives, not the one the
   reader yields. A normalization test has to construct the reader's dtype.
3. **The reviewer found it by reading the claim, not the results.** Nothing in
   the output would have surfaced it.

### D21 — the encode produced fixed-width columns

`numpy.char.encode` gives `|S`, whose width is fixed by the longest value present
at creation; a longer value written later truncates **silently** — the hazard
this series had already documented for NXstress. And since `HidraProjectFile`
re-saves a workspace it previously read, it would have converted an existing
file's variable-length columns to fixed-width on every save. Now
`h5py.string_dtype(encoding="utf-8")`, matching `_Sample._writable`.

Found by writing the assertion the measurement deserved (`stored[()].dtype.kind
== "O"`) rather than the one that would pass.

### D22 — the conversion passed misuse through instead of raising

`to_text_array` returned a numeric array unchanged, because `SampleLogs` calls it
for every log and most logs are numbers.

**Reviewer:** *"Why would `to_text_array` leave numeric types untouched? In that
case it's obviously being abused and should raise an exception -- that would be a
developer 'usage' error!"*

Same mistake as D20 in miniature: a pass-through makes a misdirected call look
like a working one, and the symptom surfaces later and elsewhere. Now
`is_text_array` is public and carries the definition of "array of strings"
(including the element-wise inspection an `object` array needs), `to_text_array`
raises `TypeError` naming the dtype that arrived, and the two call sites that may
hold either kind guard with the predicate -- `SampleLogs.__setitem__` and
`_Peaks._decoded`, whose discriminator column may legitimately be numeric.
`TestIsTextArray::test_it_agrees_with_what_to_text_array_accepts` pins the two
together so they cannot drift.

**Except for an empty `object` array**, which the reviewer flagged as a
legitimate case the strict version would break — *"I can imagine a sample-log
that will be an array of bytes (on disk), but is presently empty."*

Checked, and the ambiguity is not there: an empty HDF5 dataset keeps its own
dtype on read, and only a variable-length string comes back as `object`
(`float64` stays `float64`, `|S4` stays `|S4`). So empty + `object` is
unambiguously an empty string column; it has no values to misclassify, and `<U`
is both the invariant's dtype and the data's. Empty + `float64` still raises.

It is also already reachable, and **both** previous behaviours were wrong.
`write([ws], [[]])` — the documented no-peak-fits shape — leaves `phase_name` and
`mask` as empty `object` columns, and the hand-rolled `_decoded` turned them into
`np.array([])`, i.e. **`float64`**: an empty array of numbers standing in for a
string column. Invisible because the caller returns on `len == 0` two lines
later. `to_text_array` now passes `dtype=np.str_` for precisely that case.

**Out of scope, and flagged for explicit review.** `pyrs/dataobjects/sample_logs.py`
and `pyrs/projectfile/file_object.py` are not in this plan series. Taken here on
the reviewer's judgement that *"there probably would not ever be a separate PR
about the core change"*, on condition that the commit message calls it out.

**Incidental finding, not fixed:** `summary_generator.py:169` calls
`value.decode()` and **discards the result** (`value.decode()`, not
`value = value.decode()`), which is why bytes reached that CSV header at all.
Pre-existing, unreachable under the new policy, and left for its own change.

### Also found: nothing runs the doctests

`restorable_property.py` (batch 2) and now `as_text` carry doctests, and there is
no `--doctest-modules` in `addopts` and no collector anywhere in `tests/`. They
are documentation, not tests. Recorded rather than fixed — turning them on is a
repo-wide change with its own fallout.

### Verification

**494 unit** (was 443) / 104 integration / 16 GUI; `ruff` and `mypy` clean; probe
sweep 19 OK, 1 SKIPPED (retired), 0 FAIL.

---

## Batch 17 — `test_NXstress.py`, `test_append.py`, `test_multi_workspace.py`, `test_workspace_read.py`

Pairs with batch 6. **1409 lines from the PR** -- the largest test batch -- plus
388 added by earlier parts of this review and reviewed as written.

`test_append.py` is 927 of those and the byte-neutrality discipline in it is
real rather than decorative: **17 refusal tests assert the file's `sha256` is
unchanged**, and the three that do not are about *instance* state, where file
state is not the subject.

### D37 — the one refusal that did not assert byte-neutrality

```python
with pytest.raises(ValueError, match=r"skips past the next free entry"):
    with NXstress(written, "a", entry_number=12) as nx:
        nx.write([ws], [peaks])
```

A refusal *after* the file is opened for append -- so 04c's no-op guarantee
applies, exactly as it does to the seventeen siblings that check it. The other
two raising tests in that class raise in `__init__`, before any open, which is
why they legitimately do not.

Worth closing rather than tolerating: this guarantee has broken twice already
(04c Follow-up 3 F3.1), both times silently, with the right exception and a
changed file.

### D38 — an `autouse` fixture whose entire body was a string

```python
@pytest.fixture(autouse=True)
def setUp(self, load_HidraWorkspace, createPeakCollection):
    """
    self.sampleLogs = self.ws._sample_logs
    ...
    self.peak0 = createPeakCollection(...)
    """
    yield
```

Found by the unused-fixture scan, which reported both parameters unused -- they
are: every statement that would have used them is inside a triple-quoted string.
The same shape as the `permutations` block in batch 11, and here it is
`autouse`, so it ran for all 22 tests in the class.

It could not be revived either: `self.ws` is assigned nowhere, so the first line
would raise `AttributeError`. All five references to `self.ws` / `self.peak0` /
`self.sampleLogs` in the file are inside that string; no live test uses them.

Removed rather than repaired. Its one piece of real content -- a comment
explaining that **no single available project file carries everything an
NXstress entry needs**, which is why each test builds its own source -- is kept,
promoted to the class docstring where it is read.

### O22 — two unused `tmp_path` fixtures

`test__Peaks_fields_and_subgroups`, and the `discriminated` fixture in
`test_append.py` on which most of that module depends.

Distinguished from the eleven other "unused" parameters the scan found:
`discriminated` and `config_override` are **side-effecting** -- requesting them
without naming them is how they take effect, and `config_override`'s teardown
provides isolation even when never called. Those are correct and were left
alone. So was `failing(field, values)`, a monkeypatch stand-in whose parameters
must match the signature it replaces.

### O23 — an unused unpacking target

`roundtrip_named_masks` bound `peaks_readback` and yielded only the workspaces.
Ruff's `F841` does not see tuple-unpacking targets, so it passed lint. Now `_`,
with a note that the fixture is about mask naming.

### What this file did not catch

Both batch 6 defects trace to gaps here, and are recorded because *"the tests
were thorough and still missed these two"* is the useful fact: every
diffractogram in the module is keyed `{None: ...}`, where the mask and
diffractogram namespaces agree, so **D17** was invisible; and `N_SCAN = 7` was
chosen to dodge the length collision that **D18** turned out to be, under a
comment reasoning about it. Both now have tests, with the reasons written into
the module.

### Verification

534 unit / 112 integration / 16 GUI; `ruff` clean.

---

## Batch 26 — `04c-nxstress-append.md`

`open-questions/04c-nxstress-append.md` is untouched by the PR -- Appendix A
row 26 is correct -- and was checked for A7 freshness anyway: all six questions
are RESOLVED/OBSOLETE and none is falsified by what landed.

### Coverage

A1, A2, A3, A7. **A4/A5 satisfied, not exempt** -- 04c rests on
`a4_h5py_nexusformat_append.py`, `a4_growable_string_fields.py`,
`a4_nxfield_maxshape_introspection.py` and `a5_append_preconditions.py`, all
committed with real output pasted.

### The reviewer's ruling, which re-scoped the batch

Asked whether citations inside append-only `## Follow-up` sections could be
corrected:

> it is OK to update the citations in a follow-up section; that is, the citation
> specifics themselves are *not* subject to the "do not edit follow-up" rules.
> Stale citation *values* are worthless because they will point to something
> that makes no sense; when required, the actual previous values can be
> recovered from the git history!

Recorded as **README Decisions Log item 41**. It withdraws **O35** (batch 24's
edits to 04c's Follow-ups were correct) and reverses batch 25's deferral on 04b,
which has been applied retroactively as 04b's Follow-up 12.

### D47 -- fixed: every 04c citation re-resolved by symbol

33 citations, 30 assertions, all passing. Detail in 04c's **Follow-up 6**.

The method is the finding. **Batch 24 remapped four of 04c's pointers by diff
position, and positional remapping is only as good as the pointer it starts
from.** `_sample.py:312` was cited for "the variable-length dtype" while landing
on `_present_in_all`; the remap moved it to `:529`, where it landed on
`_present_in_all` again. A positional remap preserves where the *text* went;
only a symbol lookup preserves what the *claim* meant. This is the concrete
answer to O34.

Worst single case: `_instrument.py:317` was cited for the *mask* group's
extend-by-name behaviour and had come to land on the **detector** `depends_on`
assignment. Real, related, plausible -- and wrong.

### Also fixed, found while sweeping

The viewer call-site enumeration in Follow-up 2's verification was wrong in
three of five entries, and omitted `nxstress_load.py:55` -- the only `"r"` in
the set, and so the one the claim most depends on. The conclusion ("every call
is `"w"` or `"r"`") survives; the evidence for it did not. Re-derived with
`grep -rn "NXstress(" pyrs/interface/`, which is how it should have been taken:
a hand-written enumeration goes stale silently, a command can be re-run.

### D46 -- fixed: the family table omitted two of its own five members

`## Scope: all position-aligned groups` is where the spec says what append
covers, and its central promise -- "append never leaves one group's rows grown
without the corresponding rows in every other group in its family" -- is
quantified over that table. The table listed three members; the code grows five.
Missing: `instrument/monochromator/wavelength` (04b had already established it
is per-scan-point, and 04c then filed `_instrument.py` under *name-keyed, no
insertion needed*, which is true of `_Masks` and false of the module),
`SAMPLE_DESCRIPTION/transformations/*`, and `entry/start_time`/`end_time` --
the section had no entry-level row at all.

Fixed by enumerating the family exhaustively **and** naming the real authority
beside it: each module's `appendableDatasets`, aggregated by
`NXstress._appendableDatasets` and consumed by `_validateAppendableShapes`.
A prose table is a restatement and goes stale; pointing at the code that is
actually consulted means the next member shows up in one place rather than
having to be remembered in two. Recorded as F6.4.

### D48 -- fixed, and it released the last stuck pointer

The Overview's "`NXstress.write` already supports ... each call adds a new,
auto-numbered `NXentry` **today**" was true of the pre-04c starting point and
false of the code this spec shipped: under Decisions Log item 33 an omitted
`entry_number` now *appends* to the highest existing entry. Rewritten into the
past tense, labelled as the starting point, with the current behaviour stated
after it and pointed at `write()`'s own docstring table.

Worth separating from ordinary drift: this also freed `NXstress.py:151-152`,
the one pointer the D47 sweep had to leave alone. **A stale pointer attached to
a stale claim cannot be fixed as a pointer** -- repointing it would have made
the citation contradict the sentence citing it -- yet a mechanical sweep reports
it as merely out of date, indistinguishable from the forty that were. Recorded
as F6.5.

### Verification

`check_links` 0 across 25 documents; `check_citations` 3 of **504** (known
`open-questions/06` residue; the total rose as the new Follow-ups added
resolvable citations). 35 of 35 symbol assertions pass. A final sweep of all 69
citations in 04c and 84 in 04b leaves none pointing at a blank or structural
line except deliberate "was X" values inside correction records. No code
changed; tiers not re-run.
