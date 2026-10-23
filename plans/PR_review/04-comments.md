# PR review — subspec 04 (NXstress internal cleanup)

Review of `eb5457b1..cda22352`, per
[`plans/PR-review-process/review-process.md`](../PR-review-process/review-process.md).

Each entry: the comment, the agreed resolution, and the change actually made.
My summaries of the diff are not recorded (§6e); defects I flagged are, whether or
not they were acted on.

---

## Batch 1 — `pyrs/utilities/NXstress/_definitions.py`

`_definitions.py` carries work from both `04` (the identifier policy and the mask-key
correspondence) and `04c` (the growth helpers). Per the process document §4, the batch
is split across documents by which subspec's change is under discussion: the
`allowed_identifier` exchange is here, and `D1`–`D3` on `growable`/`tail_append` are in
[`04c-comments.md`](04c-comments.md).

### Reviewer question — how is a literal `__` encoded?

Answered in full in [`04c-comments.md`](04c-comments.md) (kept together with the rest
of that exchange rather than duplicated). Summary: `__` encodes to `__5F_` and
round-trips exactly; the durable point is that *n* consecutive underscores expand
~2.5×, which interacts with the 63-character `MAX_IDENTIFIER_LENGTH` cap. No real log
name triggers it, and the failure is loud. **No change.**

### No defects flagged against `04`'s part of this file

The identifier encoding was checked rather than read: all three docstring examples
produce exactly what they claim, and the encoding is **injective and round-trips over
7380 adversarial inputs** drawn from `_ . : a A 0 u $` and space at lengths 1–4, with
every output matching `VALID_ITEM_NAME` and fitting the 63-character cap.

Worth recording one non-obvious property the correctness rests on, because it reads
like an optimisation and is not: a literal `_` is emitted only when the next character
is neither `_` nor illegal, so **two literal underscores are never adjacent in the
output** and a literal `_` never precedes an escape. Without that one-character
lookahead, `_` followed by `:` would encode to `___3A` and the decoder would attempt
`int("_3", 16)`.

---

## Batch 5 — `pyrs/utilities/NXstress/_instrument.py` (brought forward during batch 4)

Opened early at the reviewer's direction, because batch 4's D8 discussion ran into
the mask handling and the context was loaded. Four defects, **one root cause**.

### The root cause: two namespaces conflated into one

| | what it is | always has | lives in |
|---|---|---|---|
| **detector mask** | `_DEFAULT_` + named masks from `_mask_dict` | an array | `masks/detector/` |
| **diffractogram key** | a `_diff_data_set` key — *references* a detector mask, optionally ∩ an eta ROI | no array of its own | the `FIT` group's names |

`reduction_manager` builds the key as `"{mask_id}_eta_{eta_cent}"`, dropping either
part that is absent, so the four shapes are `None`, `"mask_a"`, `"eta_-5.0"` and
`"mask_a_eta_-5.0"`. `_Masks.mask_keys` unioned those keys into the *mask*
namespace, and everything below follows from that.

**Reviewer, correcting me twice on the way:** *"'eta ROI intersected with mask_vec'
actually does imply that the default-mask is being used, when no other detector
mask has been specified"*, and *"the default detector mask is always being used.
There is no reduction that 'has no mask'."* Both corrections mattered: the first
removed my objection that a link to the default would be *false*, the second showed
that `None` means "the default mask alone", not "no mask" — which is what makes the
diffractogram keyspace and the mask namespace genuinely different sets rather than
the same set with a quirk.

### D11 — `masks/names` listed masks that had no array

Measured before the fix, on a texture-shaped workspace:

```console
  names          : ['_DEFAULT_', 'eta_0.0', 'eta_-5.0', 'eta_5.0']
  detector/      : ['_DEFAULT_']
  solid_angle/   : []
      eta_0.0    -> NOWHERE  <-- name with no array
```

The `NXlink` meant to patch this was never serialised. **And making it work would
have been worse**: `masksFromNexus` would then return the default array under each
composite key, and `set_masks_from_dict` would add entries to `_mask_dict` that the
original workspace never had — the D8 defect in a different property.

**Change:** `_Masks.mask_keys` returns the mask namespace only. The `NXlink` branch
and its warning are gone. `masks/names` now carries `['_DEFAULT_']` for that
workspace, and every name resolves.

### D13 — a texture workspace could not be written at all

Two independent rejections, both from the same conflation:

```console
  eta-only reductions      -> ValueError: Reduced data required for mask '_DEFAULT_'
  eta-keyed PeakCollection -> ValueError: Mask 'eta_0.0' required by `PeakCollection`
```

The first because `nxstress_mask_names` injected `DEFAULT_TAG` into the
*diffractogram* keyspace, demanding a bare `DIFFRACTOGRAM` for data never reduced.
The second because `validateWorkspaceAndPeaksData` checked `PeakCollection.mask` —
a **diffractogram key** — against the *mask* namespace, where `eta_0.0` is correctly
absent.

**Change:** a new `nxstress_diffractogram_keys` that does **not** inject the
default, used by `_Fit.init_group` and `_Fit.validateAppend`; and the peak check
now validates against `_diff_data_set`, which its *second* check was already doing
correctly — so fixing it meant deleting the first, not adding a third.

### D10 — the solid-angle heuristic guessed, and guessed wrong

```console
  detector mask, float, 16 px -> solid_angle=True    MISCLASSIFIED
  solid-angle, 3 floats (odd) -> solid_angle=False   MISCLASSIFIED
```

Nothing in PyRS loads a solid-angle mask into a `HidraWorkspace` —
`read_mask_solid_angle` exists on `HidraProjectFile` but is called only in one test,
and `_load_masks` reads detector masks only. So the branch has never run.

**Change, per the reviewer:** *raise `NotImplementedError`* rather than classify,
and fold in `_generate_default_mask`'s dead `detector_mask=False` branch (which
returned a plausible-looking `[-180.0, 180.0]`). The `solid_angle` group stays as the
placeholder, and `masksFromNexus` still reads it, so a future writer needs no reader
change.

### D12 — a workspace with no mask wrote a file PyRS could not read

```console
  masks SET (default exists):        on disk (16,)  -> read back OK
  masks ABSENT (default generated):  on disk (4, 4) -> READ FAILED
                                     RuntimeError: Mask array with shape (4, 4) is not acceptable
```

`_generate_default_mask` returned `np.ones(detector_size)` — 2-D — while
`set_detector_mask` refuses any 2-D array whose second axis is not 1. **All 42
round-trip tests use `with_masks=True`**, so the generated-default path had never
been round-tripped.

**Change:** return 1-D `np.ones(nrows * ncols)`, plus the missing
`with_masks=False` round-trip test.

### Also repaired

**A retained probe had stopped running.** `a5_transformations_chain` calls
`_Instrument.init_group(ws)`, which 04b generalised to `list[HidraWorkspace]`; it
has been broken since 04b landed, undetected because nothing runs the probes.
`probes/README.md` lists it as *retained* for the numeric composition — unlike
`a5_nxstress_roundtrip`, whose failure is documented and deliberate. Repointed; it
confirms Decisions row 26 still holds (8 of 8 transformations reachable).

**A test was asserting an impossible workspace.** `test_Fit_multiple_masks` built
`_diff_data_set` keys `mask1`/`mask2` without ever registering those masks, which the
new reference check correctly rejects. A real reduction using `mask1` had `mask1`
loaded; the test now does too.

### `OPEN DEFECT` — the mask-append path is unreachable and would break

`_Masks.init_group(ws, masks=...)` reads `names = masks["names"].nxvalue`, which
returns a bare `str` rather than a 1-element list when exactly one mask exists —
so `names.append(...)` would raise `AttributeError`. Unreachable today
(`_Instrument.init_group` never passes `masks=`), and left standing rather than
fixed blind, since the append semantics for masks are not otherwise specified.

---

## Batch 12 — `tests/unit/pyrs/utilities/NXstress/test_definitions.py`

Pairs with batch 1. 146 insertions: the identifier-policy tests (Decisions row
23) and `TestDiffractogramKeysAreNotMaskNames` (batch 5).

### No defects in the code under test

Every boundary probed behaves correctly. The identifier tests replace one that
pinned the *previous* behaviour — "replaces `:` with `_`, leaves other chars
unchanged" — and the file records why: that conversion was many-to-one, so
`HB2B:CS:X` and `HB2B_CS_X` both became `HB2B_CS_X` and one log silently
overwrote the other, `local_name` included.

The round-trip inputs are well chosen rather than decorative: `"a_3Ab"` (shaped
like a single-underscore escape), `"__"`, `"a___b"` are the adversarial cases for
an escape introducer.

### O12 — an assertion that could not fail

`test_allowed_identifier_output_matches_rule` asserted
`len(encoded) <= MAX_IDENTIFIER_LENGTH` over ~765 three-character names. Measured,
the longest encoding those produce is **14**, against a cap of **63** — the
assertion was unreachable, reading as length coverage while providing none.
Removed, and replaced by O13 and O14 below, which cover what it appeared to.

### O13 — the length boundary was untested

The only length test passed 32 colons, overshooting the cap several times over.
The edge itself was correct and unpinned:

```console
62 legal chars -> ok (len 62)
63 legal chars -> ok (len 63)
64 legal chars -> ValueError
```

Now `test_allowed_identifier_at_the_length_cap_is_accepted` (parametrised at
`MAX-1` and `MAX`) and `..._one_past_the_length_cap_raises`. **Confirmed able to
fail** by mutation — shifting the cap by `+1` and re-running breaks the `[64]`
case, which is what the assertion it replaced could not do.

Deliberately kept alongside `test_allowed_identifier_too_long_raises`: that one
passes a *short* name whose **encoding** overruns the cap, so the two pin
different paths. Said so in the docstring, since the pair otherwise reads as a
duplicate.

### O14 — round-trip coverage was far narrower than rule coverage

The rule test ran 765 generated names; round-trip ran 13 hand-picked ones.
Injectivity is the property the whole policy rests on, and the broad input set
was already built. The generator is now a `_every_byte_value_name` helper used by
both, so round-trip covers ~770 names:

```console
765 names, 0 fail to round-trip
```

`test_allowed_identifier_round_trip`'s dozen hand-picked inputs stay, because
they are *chosen* — escape-shaped strings and real HB2B log names — and would be
diluted by generation.

### Verification

530 unit (was 526) / 112 integration / 16 GUI; `ruff` clean.

---

## Batch 18 — `tests/unit/pyrs/projectfile/test_file_object.py`

76 lines, entirely new: two source-scan guarantees carrying 04's FIXME
invariant. The module docstring justifies the unusual form well -- the plan
series scheduled an audit of two "legacy log name" FIXMEs, and that work turned
out to have been done upstream already. A planning cycle was spent
re-discovering it; a test would have said so on the first commit after the fix.

`test_two_theta_key_is_canonical` walks the AST and excludes docstrings, with
the reason stated: the `RuntimeError` that *replaced* the legacy fallback names
the old key in its own documentation, which is a description of the fix, not a
recurrence of the defect. This is the idiom batch 14's `argsort` scan was
changed to follow.

### D39 — the stated invariant was broader than the check, and the file contradicts it

> *"Verify the module reads two-theta through HidraConstants, not a literal."*

It checks only for the **capitalized** `"2Theta"`. The canonical value is
lowercase -- `HidraConstants.TWO_THETA == "2theta"` -- and a literal of exactly
that string lives in executable code at `file_object.py:41`:

```python
class DiffractionUnit(Enum):
    TwoTheta = "2theta"
```

So a literal two-theta key does exist and the test passes. The key's nine real
uses all go through the constant, and the enum member is a *unit label* rather
than the HDF5 log key, so the enum is left alone and the **claim** is narrowed
to what is actually guarded: no capitalized legacy spelling in executable code.
Renamed `test_no_capitalized_legacy_two_theta_key_in_code` to match.

Same reasoning as D35's path count: an overstated invariant is what gets a guard
deleted by the next person who checks it and finds a counter-example.

### O24 — the FIXME scan could not see a two-line FIXME

It required `FIXME` **and** one of `legacy|2Theta|log name` on the *same* line.
The natural way to write the thing it guards against does not:

```python
# FIXME: this fallback is wrong
#   it accepts the legacy 2Theta key
```

`ast` is no help -- a FIXME is a comment, which the parser discards -- so a text
scan is right; it just has to be over the comment **block**. Measured against
both shapes:

```console
two-line FIXME   old scan: 0 found   block scan: 1 found
one-line FIXME   old scan: 1 found   block scan: 1 found
```

### O25 — the batch plan named a file the PR never touched

Appendix A lists `projectfile/__init__.py` in this batch;
`git log eb5457b1..cda22352 -- pyrs/projectfile/__init__.py` is empty. Second
instance after batch 8's "three `interface/*_model.py`" that were four.

Recorded in
[the review process's Appendix A](../PR-review-process/review-process.md) rather
than silently corrected, with the general lesson: a file list drawn from a diff
goes stale as soon as the review starts changing files, so each batch's
membership should be re-derived from the range when the batch opens -- which is
what the per-batch `git difftool` command does anyway, and is why both were
caught.

### Verification

534 unit / 112 integration / 16 GUI; `ruff` clean.

---

## Batch 20 — generator, consumer, binary

`tests/scripts/make_nxstress_sample_position_fixture.py` (119),
`tests/integration/test_nxstress_sample_position.py` (219), and the 2.7 MB
`tests/data/HB2B_1327_with_instrument.h5`. Opened in that order per §3.

### The provenance record still describes the artifact

The one property a committed binary plus its generator must have, and the one
that decays silently. Regenerated to a scratch path and compared:

```console
only in committed  : none
only in regenerated: none
value differences  : none
size               : 2724084 vs 2724084
```

Byte-identical, re-checked after the O29 edit below.

The generator documents what a reader needs: why the fixture must exist at all
(writing NXstress needs geometry and a parseable peak tag; non-NaN positions
need finite `vx`/`vy`/`vz`; those two sets of files were disjoint), what is real
(the coordinates under test, copied verbatim), what was supplied and from where,
and what was dropped and why.

The consumer pins several things worth pinning: that the written positions are
**not** the identically-named stage logs; that the z stage axis runs opposite to
the z sample axis (`correlation < -0.999`, so the open question is re-raised if
the frames ever coincide); and `assert escaped > 0`, guarding against the fixture
ceasing to exercise escaped PV-log names.

### D40 — the test ran only from the repository root

`FIXTURE = "tests/data/HB2B_1327_with_instrument.h5"` -- a bare relative path,
resolved against the process working directory. From anywhere else all seven
errored:

```console
7 errors in 0.26s      (before)
7 passed in 0.57s      (after)
```

Invisible under `pixi run test-integration`, which starts at the root; it bites
anyone invoking pytest on the file directly or through an IDE runner. Now via
`test_data_dir`, the session fixture built from `request.config.rootpath` that
the NXstress unit conftest already uses, plus an `assert ... is_file()` naming
the generator so a missing fixture says how to rebuild it.

### O28 — `import h5py` inside the fixture body

No side-effect reason, unlike `pyrs.utilities.config` in batch 19. Moved to the
module imports.

### O29 — the generator wrote `TARGET` before it knew the result was acceptable

The 8 MB check ran *after* `h5py.File(TARGET, "w")` had already truncated the
committed fixture, then printed to stderr and returned 1 -- leaving an oversized
file in `tests/data/` where someone who did not read stderr can commit it.

**My first fix was worse than the defect**, and the reviewer caught it:
*"Where did the file come from in the first place? We have no business deleting
files that we didn't create!"* Unlinking after the truncate turns a *modified*
tracked 2.7 MB fixture into a *deleted* one -- the script destroying a file it
did not create, having already destroyed its contents.

The actual fix is not to touch `TARGET` until the result is acceptable. The
build now goes to a `.staging` file beside it, the size is checked there, and
only then is it moved into place. A `try/finally` removes the staging file on
any failure, and that is the only file the script ever deletes -- one it made
itself.

Both paths measured:

```console
normal run : wrote .../HB2B_1327_with_instrument.h5 (2.72 MB)
             byte-identical to the committed fixture; `git status` clean

refusal    : refusing to install HB2B_1327_with_instrument.h5: 2.72 MB exceeds
               pre-commit's 1 MB ceiling.
               .../HB2B_1327_with_instrument.h5 is unchanged.
             exit=1; fixture byte-identical; no staging file left; `git status` clean
```

(The refusal was exercised by forcing the ceiling down, which also exposed that
the limit was two separate literals -- the comparison and the message -- and
could drift. It is now the single `SIZE_CEILING_MB`.)

### Also fixed — two pre-existing modules with the same cwd dependency **(outside this PR's scope)**

`test_texture_reduction.py` (`DATA_DIR = "tests/data/"`) and
`test_manual_reduction_ui.py` (ten inline `"tests/data/..."` literals) have the
same fault, predating this PR. Both now resolve from `__file__`.

They cannot use `test_data_dir`: the paths appear inside
`@pytest.mark.parametrize`, which is evaluated at **collection** time, before any
fixture exists. `Path(__file__).resolve().parents[2]` is the available
equivalent, and is what the generator script itself already uses. Collection
from another directory now succeeds for all three modules:

```console
16 tests collected in 0.08s
```

### Verification

534 unit / 112 integration / 16 GUI; `ruff` clean.

---

## Batch 23 — `plans/NXstress-prod/README.md`

First of the document batches, and the one everything else cites. **110 lines
from the PR**; 6 are this review's (Decisions rows 38-40).

Decisions rows 24-35, §5 ownership and test-inventory corrections, and
`## Follow-up 3`. Two entries there do unusual and useful work:

**F3.1** explains why the new rows start at 24 when an earlier consolidation
collapsed 23/24/25 into one -- the numbers were vacated, three specs still cite
the old ones, and all three still resolve to row 23. Not reconstructible later.

**F3.4** records a real conflict between two instruction files about where a
finding lives, resolves it, and explicitly declines to edit the shared
`CLAUDE.md` because it is not this series' to settle. It also records that two
findings were written into `docs/ground_truths.md` in full and **reverted**,
a second full copy being the drift mechanism the process rejects.

### D43 — one of F3.4's four citations did not resolve to what it claimed

F3.4's argument turns on four line-number citations. All four were opened:

| citation | claim | held? |
|---|---|---|
| `CLAUDE.md:12` | *"ALWAYS record key findings in `docs/ground_truths.md`"* | yes, exact |
| `CLAUDE.md:330-333` | probe results *"**also** belong"* there | yes, exact |
| `CLAUDE.md:357-363` | findings go in an append-only `## Follow-up N` | yes, exact |
| `process.md:109` | *"its single mention ... is an incidental example"* | **misleading** |

Line 109 is a row in the **anti-patterns table**, about `10-flip-defaults.md`
and `self.statusBar()`. It does mention the file -- *"`docs/ground_truths.md`
records that this repo moved PyQt5 to PyQt6"* -- so "incidental example" is
accurate and it genuinely is the only mention. But the sentence reads as though
109 were *about* routing, so a reader who checks it lands on a Qt anti-pattern
and concludes the argument is unsourced. The claim is true; the citation looked
wrong.

Now says what line 109 actually is. F3.4 is the entry a future session is
explicitly directed to read, so a citation that survives checking but not
reading is worth more there than elsewhere.

### Why the toolkit did not catch it

`check_citations` reports 423 resolved citations and 3 known problems, none
here. It verifies a citation **resolves** -- the path exists, the line range is
in bounds. It cannot verify the line says what the prose claims. That is exactly
the A1/A3 distinction the audit process draws, and this is a clean instance of a
citation passing the tool while misleading a reader.

### Verification

`check_links` 0 findings; `check_citations` 3, the known `open-questions/06`
residue.

---

## Batch 24 — `plans/NXstress-prod/open-questions/04-nxstress-internal-cleanup.md`

Two items: **O33**, a wording defect in invariant 7; and **Q5**, which the
reviewer declined to leave as it stood.

### O33 — invariant 7 asserted two things and could only fail on one

Invariant 7 bundled the identifier *alphabet* with the 63-character cap. Split
into 7 (alphabet) and 7b (the cap, stated at its edge), noting that the length
half "was an assertion that could not fail (longest encoding of those inputs is
14, against a cap of 63)". Same family as O12, O20, O30 and D42.

### Q5 — the sample `depends_on` chain

Q5 recorded that PyRS writes no `SAMPLE_DESCRIPTION/NXtransformations`, and
justified leaving it that way on the grounds that *"the sample side has no
equivalent [referent] that has been located"* — writing one would mean guessing a
goniometer convention.

The reviewer rejected both halves:

> we need to implement a working sample-transformations chain now; we can correct
> it if the answer to Q5 changes it. We cannot leave the depends-on pointing at
> the first entry only? That would be unacceptable.

A translations-only chain was implemented against that, with the rotations
excluded on the stated grounds that `rotate_project_q` "yields no axis vectors to
copy". The reviewer rejected that too:

> I'm not understanding how we can be presented with a 3-angle transformation, in
> an applicable form, that can't simply be mathematically decomposed? That doesn't
> make any sense!

**They were right, and the claim was withdrawn rather than defended.** The
decomposition is in `probes/a5_sample_rotation_axes.py` and is exact: of 2880
candidates, one survives, at `5.8e-14` over 200 random angle triples. Written up
as subspec 04's **Follow-up 9**, with the probe output pasted into Q5.

### The part worth keeping

A first pass at the decomposition *did* come back ambiguous — two survivors
differing by a handedness sign — and that was nearly recorded in Q5 as a residual
assumption the chain would have to adopt. It was an artefact of the search space:
the projection is `diag(1, -1, -1)` at zero stage angles, not the identity, and a
candidate set containing no constant factor cannot match a composition that has
one. **An underdetermined decomposition is as likely to mean the candidate space
is wrong as it is to mean the referent is ambiguous**, and nothing in the result
distinguishes them. Recorded as F9.2.

This is the same shape as the recurring finding in this review — an assertion
that could not fail, here inverted into a *negative* result that could not
succeed.

### Verification

`pixi run test-unit` 545 passed. Real-file round trip of
`HB2B_1327_with_instrument.h5` (150 scan points): seven links, all reachable,
chain-on-disk vs `rotate_project_q` max error `1.11e-16`. Three mutants of the
composition test all caught. ruff clean; mypy clean on `_sample.py`.
