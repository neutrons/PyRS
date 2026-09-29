# PR review — files with no subspec

Two batches of the walk cover files changed in the PR range that belong to **no
NXstress subspec**: they came from `61389b28` ("integration-test fixups"), which
landed between 04b and 04c as a fix for a defect the NXstress work exposed rather
than caused.

They have no `## Follow-up` section to record against, because they have no
specification document. This file is that record. Findings here are numbered in
the same `D<n>` sequence as the subspec comment documents, so a reference to a
defect number is unambiguous across the whole review.

| Batch | Files | Commit |
|---|---|---|
| 9 | `pyrs/core/reduction_manager.py` | `61389b28` |
| 22 | `tests/integration/test_pyrscore.py` | `61389b28` |

---

## Batch 9 — `pyrs/core/reduction_manager.py`

28 insertions, 16 deletions, all inside `load_hidra_project`. Two changes, and
the comment explaining them is better than most in this PR — it states the
defect, the mechanism, and the evidence.

**The mode no longer comes from the file's permissions.** It was
`READWRITE if os.access(path, os.W_OK) else READONLY` -- derived from what the
*file* allowed rather than what the *operation* needed. HDF5 marks the superblock
on a read-write open, so merely loading a writable project file modified it on
disk; committed fixtures showed up as modified in `git status` because of it.

**The handle is closed in a `finally`.** `close()` previously sat after the load,
so a load that raised left it open -- and under pytest a traceback keeps the
frame alive for the rest of the session, so the on-disk modification persisted
with it.

Both are already in [docs/ground_truths.md](../../docs/ground_truths.md), with a
regression test in `tests/integration/test_pyrscore.py` (batch 22).

### No defects in the change

Each of the comment's claims was checked rather than taken:

- *"this method only ever reads"* -- holds.
- *"every save path opens its own READWRITE handle by filename"* -- holds;
  verified at `peak_fitting_model.py:244`, `texture_fitting_model.py:217` and
  `powder_pattern.py:246,250`.
- **The same pattern elsewhere** -- swept for. `os.access(..., os.W_OK)` survives
  in four places (`pyrs_api.py`, `checkdatatypes.py`, `file_util.py`,
  `mantid_peakfit_calibration.py`), and every one is a permission *pre-check*
  before writing, not a mode derivation. The defect was fixed everywhere it
  existed.

### D30 — the comment named one save path twice

It listed *"`ReductionApp.save_diffraction_data`, `TextureFittingModel.save_fit_result`,
`powder_pattern`"*, but `ReductionApp` **is** the class in `powder_pattern.py` --
two names for one path, reading as broader evidence than it was. Corrected, and
`PeakFittingModel.save_fit_result` added, which is a genuine third.

### D31 — `HidraWorkspace._project_file`, a dead attribute **(outside this PR's scope)**

`HidraWorkspace.load_hidra_project` stored the handle this method closes:

```python
self._project_file_name = hidra_file.name
self._project_file = hidra_file
```

Three mentions in the whole repository -- one `None` in `__init__`, two
assignments -- and **no readers** in `pyrs/`, `tests/` or `scripts/`.
`hidra_project_file` answers from `_project_file_name`, set on the adjacent line.

**What it actually held, measured** -- because the first description of it was
wrong, and the correction matters:

```console
ws._project_file type      : HidraProjectFile
is it the same object?     : True
its ._project_h5 (open)    : File | h5py says open: True
after close(), attr type   : HidraProjectFile
its ._project_h5 now       : None
```

I had called it "a stale handle". It is not: `HidraProjectFile.close()` closes
the h5py `File` **and sets `_project_h5 = None`**, and since the workspace holds
the same object that clears through. What survived was a drained wrapper -- no
file descriptor, nothing to leak. Dead weight, not a resource leak.

**Reviewer:** *"what happens if we just delete the `_project_file` attribute? You
mention deleting the two assignments -- I guess this just leaves an uninitialized
attribute?"*

A fair correction of imprecise wording. It is *initialised*, at `workspaces.py:56`,
so deleting only the assignments would leave a declared attribute that is
permanently `None` -- the worst of the three options, since it reads as though
something used to set it and invites a future reader to start using it again. All
three mentions were deleted instead.

**This is outside the PR's scope** and is called out in the commit message
accordingly. `pyrs/core/workspaces.py` is already modified by this PR (the
`direction` restorable property, batch 2), but removing a `HidraWorkspace`
attribute is a change to a core data class that no subspec asked for. The one
caveat, stated rather than assumed: "nothing reads it" is a claim about *this*
repository, and the attribute is single-underscore private and undocumented, so
anything downstream poking at workspace internals would break.

### Verification

526 unit / 112 integration / 16 GUI; `ruff` and `mypy` clean; probe sweep 19 OK,
1 SKIPPED (retired), 0 FAIL.

---

## Batch 22 — `tests/integration/test_pyrscore.py`

68 lines from `61389b28`, the commit reviewed in batch 9. No subspec.

`TestLoadDoesNotModifyTheFile` states the mechanism well and gets two things
right: it covers **both** paths, noting that only one leaked; and it makes a
writable copy first, with the reason -- *"the bug cannot reproduce on a
read-only file, which is exactly why it went unnoticed on archive-mounted
data."*

### D41 — relative paths, the same fault as batch 20's D40

`Path("tests/data/Hidra_16-1_cor_log.h5")` at line 38 (new) and a second at
line 116 (pre-existing), both resolved against the process working directory in
a module that already receives `test_data_dir` elsewhere. The new test now uses
that fixture; `test_main`, being outside a test function's fixture scope for
this purpose, resolves from `__file__`. The filename is a single
`LEGACY_SCHEMA_FIXTURE` constant, so the `xfail` reason, `test_main` and this
class cannot drift about which file they mean.

### O32 — a bare `pytest.raises(RuntimeError)`

Now `match=r'non-empty "reduced diffraction data" entry'`. The comment states
why it matters: the closure assertion is only meaningful if the load reached
the point where a handle is open, and a `RuntimeError` raised earlier would
satisfy a bare `pytest.raises` while exercising none of it.

### D42 — the regression test did not regress

Adding that `match` prompted checking what the test caught. It caught nothing.
Reverting `reduction_manager` to the **full pre-fix state** -- READWRITE *and*
no `finally` -- left it green.

The mechanism, measured on h5py 3.16.0 / HDF5 1.14.6:

```console
r+ open, handle OPEN : changed
r+ after close       : unchanged
r  open, handle OPEN : unchanged
```

`docs/ground_truths.md` is right that a read-write open changes the bytes -- but
only **while the handle is open**; a clean close reverts the mark. By the time
the test takes its digest the `HidraProjectFile` is unreferenced and CPython has
closed it. The original symptom needed the handle held across the session, which
a digest taken after the call can never observe.

Same shape as O12 and O20 -- an assertion that cannot fail -- but worse: this is
the designated guard for a defect that reached committed files, and it is named
`test_..._leaves_the_file_unchanged`.

**Resolved by asserting what the fix changed**, through a recording
`HidraProjectFile` patched into `reduction_manager`: the **mode** each open uses,
and that every handle is **closed** even when the load raises. The digests stay
alongside, as the statement of intent -- they say what the code is for -- but
they no longer carry the guarantee alone.

Each half now fails independently:

| mutation | before | after |
|---|---|---|
| `READONLY` -> `READWRITE` | passed | **2 failed** |
| `try/finally` -> plain close | passed | **1 failed** |

**Incidental, and evidence in its own right:** with the `finally` removed, the
two tests run together *hung* rather than merely failing -- a leaked HDF5 handle
has consequences past the superblock mark, which is consistent with this repo's
"segfault at interpreter shutdown" ground truth. Run singly the mutant fails
cleanly in 0.12s.

### Verification

534 unit / 112 integration / 16 GUI; `ruff` clean.

---

## Batch 27 — `probes/README.md` + the probe files

Appendix A row 27 says "the twelve probe files". The PR touches **14**, of 21 in
the directory. Fourth row-vs-reality mismatch in the batch plan.

### The main check: every probe was run

20 of 21 exit 0. `a5_nxstress_roundtrip.py` exits 1, which is **correct** — it
is retired to record and sets `AUDIT_STATUS = "retired"`; it is the only file
carrying that marker, so the README's documented sweep idiom
(`grep -q '^AUDIT_STATUS'`) selects exactly it. **No probe is silently broken**,
which is what batch 3 found last time (`a5_transformations_chain.py` had been
dead since 04b).

One false alarm, recorded because the mechanism will recur: the first sweep
reported `a5_scan_point_family_resizable.py` timing out at 300 s. That was an
artefact of running two full sweeps concurrently — re-timed cleanly it finishes
in 4 s. **A probe sweep is not safe to parallelise with itself**, and a timeout
under contention is indistinguishable from a hang in the output.

### D49 — a probe in the PR was never added to the index

`a4_growable_string_fields.py`: 21 files on disk, 20 listed. Not orphaned —
`04c:615` cites it as a Follow-up 2 referent — it simply never got its row. The
04c plan called for three new rows here and two were written. Row added, with
the probe's real output: all three writer string forms grow correctly, bare
`<U` still fails outright, and **claim 5 is the finding** — a fixed-width `|S`
column silently truncates a longer appended value
(`b'a_considerably_longer_filename.h5'` → `b'a_consid'`, no exception).

### D50 — five stale `04c:` anchors in `probes/README.md`, self-inflicted

Batch 26's edits to 04c added ~8 lines in the Overview and ~28 in the Scope
section, shifting every anchor after them. **No tool reports this**:
`check_citations` verifies a citation *resolves*, and all five still landed
in-bounds on the wrong text. Eleven anchors re-resolved by content, plus one in
04b (`04c-nxstress-append.md:41` → `:49`).

### D51 — stale code citations inside four probe files

`a4_growable_string_fields.py` (`NXstress.py:459-460` → `1093-1094`, ×3),
`a5_nxstress_internals_today.py` (`_sample.py:127-137` → `191-201`),
`a5_peakcollection_ranges.py` (`_peaks.py:246-338` → `446-559` ×2,
`sample_logs.py:164-166` → `167-168`), `a5_subruns_nonmonotonic.py`
(`workspaces.py:517` → `574`).

Two wrinkles worth keeping:

- In `a4_growable_string_fields.py` the citation is a **dict key the probe
  prints**, so correcting it changes the probe's own output. Harmless here
  (confirmed that output is pasted into no document) — but this is how a
  pointer fix can silently invalidate pasted evidence somewhere else. The four
  edited probes were re-run and diffed against their pre-edit output: three
  byte-identical, and the fourth differing in exactly the one renamed key.
- In `a5_peakcollection_ranges.py` and `a5_subruns_nonmonotonic.py` the stale
  pointers sit **inside quotations** of README/04b text that batches 25–26 had
  since corrected, so the quotes had become unfaithful to their sources.
  Updating them restores fidelity *and* fixes the pointer.

### D52 — one stale document anchor

`a5_subruns_nonmonotonic.py` cited `04b:304` for the "reuse the merge logic"
bullet; that line is now `ws.set_sample_log(...)`. Corrected to `04b:336-339`.
The other document anchors in the probes (`04b:37-43`, `:43-45`, `:49-55`,
`README:729`, `README:335-342`) all still resolve — checked individually.

### Found by a reviewer correction, and fixed here rather than deferred

I had set `README:99`'s stale `NXstress.py:149` aside as "a batch-23 matter".
The reviewer's response — *"note that batch 23 is in the past!"* — is the
finding: **deferring to a completed batch is deferring to nowhere.** A defect
found while auditing document B does not become someone else's problem because
it lives in document A and A was walked earlier. Corrected to
`NXstress.py:156-164`, `write()`'s entry-targeting table, along with the
retired probe's quotation of that sentence so the quote stays faithful.

### D53 — fixed, and it was one bullet in eleven

The reported defect was that `README:114-123` still said append *"raise[s] on
any operation that would extend an existing `NXentry`"*. Auditing the section it
sits in showed the problem is the section, not the bullet: **§2.1 is a gap list
that was never revisited as gaps closed.** Of its eleven entries, **six have
since closed** (one only in part) and were still written as open, and **nine of
eleven carried stale pointers.**

| Entry | State | Pointer |
|---|---|---|
| Appending to an existing entry | **closed in part** by 04c | `_input_data.py:44,63`, `NXstress.py:151-152` → `NXstress.py:156-164`, `:278` |
| NXbeam intensity profile | open | `_instrument.py:127` → `:250` (+ `:348`) |
| Detector L2 not extractable | open | `_instrument.py:138-140` → `:261` |
| Detector rotation-order TODO | **closed** by 04 | `_instrument.py:165` → `:288` |
| Instrument name hardcoded | **closed** by 04 | `_instrument.py:70` → `:69-81` |
| `sx`/`sy`/`sz` hardcoded NaN | **closed** by 04 | `_peaks.py:235-239` → `:237-262` |
| Fit `fit`/`fit_errors` NaN | open | `_fit.py:425-429`, `:430-438` → `:527-530`, `:531-536` |
| Mask naming inconsistency | **closed** by 04 F8 | `_fit.py:546-549` → `_definitions.py:491`, `:509` |
| `STRESS_FIELD` shape unverified | open (blocked) | `_sample.py:171` → `:272` |
| Group-name validator workarounds | open | `_definitions.py:106,…`, `_fit.py:414` → `_definitions.py:270,273,280,286,291,297,299` |
| `allowed_identifier` coverage | **closed** by 04 | `_definitions.py:229` → `:574-640` |

Closed entries are **kept and marked `CLOSED`**, with what closed them and the
evidence, rather than deleted: a gap list that silently drops its resolved
entries cannot be checked afterwards against what was actually delivered. The
section now opens by saying it is a snapshot, so the next reader knows to date
it.

**One of these was made worse by my own batch 24.** The `STRESS_FIELD` bullet
cited `_sample.py:107`, which pointed at the NaN-fill line and never at the
`STRESS_FIELD` comment. Batch 24's bulk `_sample.py:107` → `:171` substitution
moved it faithfully to a line that is *also* the NaN-fill line. That is the F6.2
lesson a second time, now with a worked example of the bulk-sed variant: **a
mechanical remap cannot notice that the pointer it is moving was already wrong**,
and it launders the error into something that looks freshly checked. The claim
belongs at `_sample.py:272`, found by searching for the comment text the bullet
actually quotes.

### D54 — the rotation-order TODO, closed; and a different defect underneath it

The `# TODO: check order of rotations here!!!` comment at `_instrument.py:288`
is **removed**. The order it asked about is settled: the written sub-chain
composes to `Rx @ Ry @ Rz`, matching `generate_rotation_matrix`.

Closing it properly needed more than deleting a comment. The order was proven by
`probes/a5_transformations_chain.py`, **but a probe runs during an audit, not on
every commit**, and the two existing chain tests pin *reachability* and
*pairwise links* — neither of which a reordering breaks, because a reordered
chain is still fully connected and still reaches every link. So
`test_instrument.py::TestDetectorRotationOrder` now composes the written
rotation sub-chain and compares it against `generate_rotation_matrix` itself,
the function the reduction actually uses.

**Mutation testing caught a real flaw in that test's first draft.** It iterated
a hardcoded `("rotation_x", "rotation_y", "rotation_z")` and looked each up by
name, so it composed in *the test's* order no matter what the writer emitted —
and passed against a writer with the rotations reversed. It now reads the order
out of the file by traversing `depends_on`. Both mutants fail against the fixed
version: reversing the three links, and swapping the `x`/`z` axis vectors (the
latter caught *only* by the composition test, which is its purpose).

The tolerance is `1e-6`, not tighter, because `generate_rotation_matrix` builds
its factors in **float32** — measured max elementwise difference `6.5e-8`. A
tighter bound would assert the referent's precision rather than this writer's
order, so the test asserts `expected.dtype == np.float32` to say which it means.

### D55 — carried forward, not fixed: `two_theta_zero`'s axis

Found while closing D54, and **distinct from it** — an axis question, not an
order one, which is why it could not be fixed under that heading.

`_instrument.py` writes `two_theta_zero` about `ex`. The reduction applies
two-theta about **Y** (`reduce_hb2b_pyrs.py::rotate_detector_2theta` →
`_cal_rotation_matrix_y`), after folding `two_theta_0` into the arm angle as a
**scalar offset** (`reduce_hb2b_pyrs.py:139`) rather than as a transformation of
its own. So there are two questions: which axis, and whether it belongs in the
chain as an independent link at all.

Underneath sits a contradiction inside PyRS, not introduced by NXstress:
`instrument_geometry.py:166` documents `two_theta_0` as *"in rotation of
detector arm about X"*, disagreeing with the reduction's own code. **This writer
followed the comment.** The code is the authority, so the comment is the likelier
error — but that is a judgement for the spec that owns `DENEXDetectorShift`, not
for a PR review of the writer.

Carried into **spec 08** (resolve the contradiction, fix whichever is wrong) and
**spec 09** (emit the link on the settled axis), both now carrying it in their
`## Scope`.

### The gap list had no forward consumer

Raised by the reviewer: *"where is this gap list; is it something later
subspecs will even see?"*

**It is spec 04's work list and nothing else's.** The only reference to it
anywhere in the series is `04-nxstress-internal-cleanup.md:11` — *"Close the
NXstress-internal TODOs from Section 2.1 of the README"*. Specs 04, 04b and 04c
have now landed, so every remaining open entry was sitting in a list whose only
reader is finished.

Checked individually: four of the five pre-existing open entries *are* owned by
spec 08's `## Scope` (beam profile, L2/arm-shift, fit-spectrum reconstruction,
`STRESS_FIELD` as explicitly blocked) and one by spec 10
(`use_production_names`). So nothing was lost — but nothing was carried by
§2.1 either, and D55 had **no owner at all** until it was added.

§2.1 is now a closed-out record that says so in its first line, with a table
naming where each open entry went. The one entry deliberately left unscheduled
— extending an existing key, and overwriting an entry — is labelled as such
rather than left looking tracked.

**The general shape, worth keeping:** a list of open work is only as live as its
consumers. When the spec that reads it lands, the list stops being a schedule
and becomes a record, silently — and anything still open in it stops being
scheduled without anyone deciding that.

### Verification

Every probe run: 20 OK, 1 retired-by-design. The four edited probes re-run and
output-diffed. 8 of 8 corrected code pointers assert against current source.
`check_links` 0 across 25 documents; `check_citations` 3 of 504 (known
`open-questions/06` residue). 21 of 21 probe files now listed in the index. No
PyRS code changed; tiers not re-run.

---

## Batch 28 — `review/findings.md` + `audit.toml` + `prompts/implementation-prompt.txt`

### `audit.toml` — machine-checked facts verified, nothing wrong

`base_commit` resolves, `base_subject` matches the commit's subject exactly, the
base is an ancestor of `HEAD`, `claims_written_against` resolves, and `[scope]`
covers all 12 subspecs exactly once.

### Withdrawn: D56

I reported that `base_commit`'s rationale was false because `base_commit..HEAD`
now spans 34 commits rather than "exactly 04b's changes (one commit)". **The
reviewer rejected this, and was right:**

> The startup process for subspec 05 will *definitely* need to span over all of
> these PR-review commits from 04, 04a, 04b, and 04c; how could it simply
> *ignore* those? Significant code changes occurred during this review, and we
> reviewed the entirety of 04-series at once. Further, we do *NOT* change the
> audit process in the middle of running the process!!

The error was reading a *rationale written for one purpose* as a *specification
of the range*. The comment explains why the base was advanced in 2026-09-30; the
range legitimately grew afterwards because the review landed code in it, and
subspec 05's landing pass must see all of it. Two lessons, both general:
a widening range is the benign direction **and the intended one** here; and a
comment describing a past decision is not falsified by the world moving on.

### D57 — fixed, and the reviewer's follow-up made it a real toolkit gap

Reported as a wrong parenthetical: `[exclude]` said *"prompts/ holds no `*.md`
(only `append.prompt.txt`)"* when the directory holds **two** `.txt` files. The
reviewer's response went further — *"I could imagine placing an `*.md` into
'prompts' -- it needs to be excluded on a directory basis!"* — and that exposed
something the comment was hiding.

**`prompts/` could not be excluded at all.** `[exclude] paths` is a glob, and
`prompts/*.md` matches zero files today, which the loader's own zero-match rule
makes an error. So the directory was uncovered *by coincidence* — the walk is
`rglob("*.md")` and there were none — and the day a `*.md` landed there the
loader would fail with "matches no classification". The comment recorded the
coincidence as if it were a design.

`manifest.py` gains `[exclude] dirs`: whole directories, excluded whether or not
they hold any `*.md` today. **A glob cannot express "and anything added here
later".** Its staleness check is that the directory exists — the same guarantee
`paths` gets from zero-match, expressed for a directory rather than a pattern.
Required, not optional, consistent with the loader's "missing keys and unknown
keys are equally errors".

Both halves verified by measurement, not by reading: a `*.md` dropped into
`prompts/` leaves the document count at 25 (excluded, not counted, not an
error), and a `dirs` entry naming a missing directory fails at load with a
message that says why. Documented in `process.md` §7.2 beside `exclude.paths`.

### `open-questions/` — considered, deliberately not excluded

The reviewer also suggested *"fully excluding 'open-questions' (which is a
planning archive directory)"*. **Not done**, and the reasons are worth keeping:

1. `series.open_questions` already classifies those documents. Adding them to
   `[exclude]` would be a *second* classification, which the loader rejects —
   loudly and correctly. Making it possible means changing classification
   precedence in `manifest.py`.
2. That is a change to the audit machinery *while an audit is running*, which is
   the rule the reviewer stated one message earlier.
3. The three `check_citations` findings that remain all live in
   `open-questions/06`. Excluding the directory takes that tool to **0
   findings** — for a reason that has nothing to do with the findings being
   fixed. A tool that reports clean because it stopped looking is the exact
   failure shape this review has been finding all along.

Recorded in findings.md §4 so the next pass sees the reasoning rather than
re-deriving it.

### D58–D61 — `findings.md` corrections

- **D58:** §0's A7 column dated 04/04b/04c at 2026-09-29/30 and 10-01, all
  before this review. Those three gained Follow-ups 9, 11–12 and 3–6 since, and
  were re-covered on every axis — now `✓✓` on A1/A2/A3 with A7 at 2026-10-15.
- **D59:** §2's measurements were stale. Re-measured: `check_citations` 388 →
  **502** resolved (same three problems throughout); tiers 412/104/16 →
  **547/112/16**. The rise is §7.7's self-check property — this review's own
  Follow-ups are now scanned. **A count that moves while the problem set does
  not is growth, not regression**, and the table says so.
- **D60:** §5's invariants table was split into *three* tables by blank lines,
  so rows 10–11 and 13/12 rendered with no header; rows 12 and 13 were also
  transposed. One contiguous table, in order.
- **D61:** §3 still carried `_sample.py:171` for the `STRESS_FIELD` blocker —
  the pointer batch 27 corrected to `:272` in the README, surviving here because
  it lives in a different file. **A pointer fixed in one document is not fixed**;
  the same wrong value propagates to wherever else it was copied.

### D62 — `implementation-prompt.txt`, line references only

The reviewer's standing instruction: this file is adjusted before each subspec's
implementation begins and **should otherwise be left alone unless explicitly
asked**. Only the line references were touched, on explicit direction.

Its "known-stale citations, with current values" block had itself gone stale on
the right-hand side: `peakCollectionRanges` 419-529 → **446-559**, `_append_peak`
309-381 → **331-404**, `init_group` 289-306 → **305-329**, the sort-order
docstring 51-58 → **53-60**, the four raise sites 499/505/519/524 →
**528/534/548/553**, `_fit.py` 98 → **109**, the overwrite guard 163-164 →
**278**, `_input_data.py` 32-74/78-103 → **32-81/83-151**, and `04c:41` →
**04c:49**. Left-hand values untouched: those are the historical "was", which is
the content. All twelve targets re-asserted against current source.

### Verification

`check_links` 0 across 25 documents · `check_citations` 3 of 502 · `check_ownership`
10 · `check_drift` 10 of 14 — all four load the new manifest key and report
exactly as before. The two new `[exclude] dirs` behaviours tested directly. No
PyRS code changed in this batch; tiers not re-run.

---

## Batch 29 — `03-combine-runs-nxstress.md`, `07-manual-reduction-nxstress.md`, `10-flip-defaults.md`

+74/−0: one `## Follow-up` each, all sibling-doc notes written when 04b/04
changed something these downstream specs depend on.

### Everything the three Follow-ups claim is true

Worth stating, because the batches before this one were almost entirely
corrections:

- **03 F2.1** — the landed call really is `nx.write([self._hidra_ws], [[]])`
  (`combine_runs_model.py:41`); the named regression test exists; the body
  really does still spell it `[]`, so the Follow-up's premise holds.
- **07 F3.1** — all three cited sites exist and all three say `[]`.
- **10 Follow-up 2** — every concrete claim verifies. `NXstress.nxdl.xml` is
  present; its sha256 is `bee46c0723720bc440570cc8f74be37cea6145b15b7e6f4c3ec00c421144b905`,
  matching the recorded value exactly; commit `004da96e…` and the provenance
  block are at `IO_prototype.rst:26-32`; `.pre-commit-config.yaml:5` excludes
  it; `validItemName` is reproduced at `_definitions.py:385`. Its cross-spec
  claim that spec 04's F1.3 "stands unchanged" holds —
  `a4_nexusformat_validator.py` re-run, and `use_production_names: false` is
  still the shipped default.
- **03 F1.1** was the only citation the sweep flagged, and it is the documented
  exception: a finding that *diagnoses* a wrong pointer. Its corrected value
  `:21` is still correct.

### D63 — the bodies are wrong at copy-paste sites, and nothing said so

The Follow-ups correct the body rather than the body being corrected. That is
the series convention and is not reopened here. The consequence in these three
documents is sharper than usual, because the superseded text is not prose a
reader weighs — it is **a call signature they would copy**, and in one case it
sits inside a fenced code block annotated *"per 04b's signature"*, which makes
it look checked rather than stale.

Six live sites, now marked inline: `03:43`, `03:78` (a deliverables bullet, and
wrong on *both* counts — bare workspace and `[]`), `03:105`, `07:73`, `07:129`
(the code block) and 10's two artifact sites, where *"neither exists yet as of
this writing"* is now half false — the schema doc landed, the validator did not.

`03:50` is left unmarked deliberately: it says "**pre-04b** signature … a bare
workspace, not a list", so it labels itself. Quotations inside Follow-ups are
left alone, per the rule that a finding keeps what it diagnoses.

### (b) existed, in the one place the reader at risk never looks

`prompts/implementation-prompt.txt:90-98` already states the rule — *"the body
is deliberately left as the record of what was believed. Where the two
disagree, **the Follow-up wins.** So read your subspec's Follow-ups first."*

**But that file is the prompt for an agent starting an implementation pass.** A
human opening the spec directly, or an agent landing mid-document, never sees
it. The rule was real and the delivery was wrong. Each of the three documents
now carries it in a block quote above its Overview, in the prompt's own words so
there is one formulation rather than two, and crediting the prompt as the
source.

The header note also says the inline markers are **a convenience and not a
guarantee** — the Follow-ups remain the authority. Claiming the markers are
exhaustive would be an assertion nothing checks, which is the failure this
review has named repeatedly.

### Verification

`check_links` 0 across 25 documents; `check_citations` 3 of 502 (known
`open-questions/06` residue). Every claim in the three Follow-ups re-verified
against code, the vendored artifact's sha256 recomputed, and two probes re-run.
No PyRS code changed; tiers not re-run.

---

## Batch 30 — `PR_review/04_commit.txt`

**No changes. Five reported defects withdrawn.**

### What I got wrong

I audited this as the PR description for the PR under review, and raised five
defects against it: that it covers subspec 04 while the PR spans 04 + 04b + 04c
(zero occurrences of "04b" or "04c"); that `FILES CHANGED` lists 5 production
files and 9 test files where the PR touches 24 and 30; that every number in
`VERIFICATION RUN` is stale (`test-unit` 315 → 547, `check_citations` 358 → 502
citations, `check_ownership` 9 → 10); that its `ground_truths.md` paragraph
argues against something this PR now does in +249 lines; and that its
`POST-MERGE` list names a completed task (`04-*.md` is already in
`[scope] freshness_only`) while omitting the live one (`04c-*.md`).

Every one of those observations is factually true. **They are not defects**,
because the document is not what I took it to be. The reviewer:

> it was only used as a suggestion for the actual commit message (which was much
> more abbreviated). For the actual PR including 04, 04a, 04b, and 04c (, which
> we are not quite ready for yet), the commit-message used will be much more
> comprehensive; and all of that will be drafted later.

It is a **spent drafting artifact** — input to a commit message that was written
and abbreviated — not a live description anyone will read as an account of the
PR. Measuring a draft against the thing it was drafted toward produces findings
that are all true and all pointless.

### Why the misreading was available, which is the part worth keeping

`review-process.md` §6a describes this file as *"a 360-line PR description
written for the reviewer"* and justifies its late placement with *"Read after
the code, it is checkable against it."* That framing states a purpose
(description for the reviewer) and an invitation (check it against the code),
and both overstate what the file is. The batch plan's row 30 repeats it:
"narrative; deliberately late (§6a)".

**A document's status is not inferable from its content**, and this one reads
exactly like a live PR description — sectioned, specific, numerate. Nothing in
or around it says "draft, superseded, do not audit". The nearest analogue in
this series is `prompts/`, which batch 28 made explicit in `audit.toml` rather
than leaving it to be inferred; and `archive/`, whose exclusion comment states
*"a superseded draft is stale BY DEFINITION and reporting its staleness would
bury the live findings."* That is precisely what happened here.

No tooling change is needed: `plans/PR_review/` sits outside the series
directory, so `manifest.py`'s walk never reaches it — verified, 25 documents
scanned, none outside `plans/NXstress-prod/`. The gap is in the human-facing
process description, not the loader.

### Recorded for whoever drafts the real thing

Not findings against this file — inputs to the commit message that replaces it,
gathered while checking:

- The PR is **99 files, +21937/−857**: 24 production, 30 test, the rest plans
  and docs.
- Current tiers: `test-unit` **547** passed / 158 deselected · `test-integration`
  **112** passed / 28 skipped / 2 xfailed · `test-gui` **16** passed.
- Toolkit: `check_links` 0 · `check_citations` 3 of 502 · `check_ownership` 10 ·
  `check_drift` 10 of 14.
- The live post-merge `audit.toml` item is moving **`04c-*.md`** from
  `[scope] full` to `freshness_only`; `04-*.md` and `04b-*.md` are already there.
- `docs/ground_truths.md` gains **+249 lines** in this PR, which the superseded
  draft argues against. Whatever the real message says about ground truths
  should be written from the current rule, not from that paragraph.

---

## Batch 31 — `IO_prototype.rst` + `NXstress.nxdl.xml` + `.pre-commit-config.yaml`

The cleanest batch of the review: no defects, two comment-accuracy
observations, both fixed.

### The check nobody had run

The recorded sha256 proved only that the vendored file matched **itself**. It
was computed from the copy, so verifying the copy against it is circular — and
that is the entire evidentiary weight behind the word "verbatim".

An upstream checkout is reachable at `/home/ux0/workspaces/NeXuS-definitions`,
which the probes' search list already knows about, so the real check was
available all along:

- commit `004da96ef29bc6b7529e1f8e5d9415cd000bccb8` exists upstream with subject
  *"Update copytright dates to 2026"* — matching the recorded provenance exactly,
  the typo faithfully preserved;
- `git show 004da96e:applications/NXstress.nxdl.xml` is **byte-identical** to the
  vendored copy — `cmp` clean, both `bee46c07…b905`;
- upstream has made **no change** to that file since, nor to `nxdl.xsd` where
  `validItemName` lives.

So the snapshot is faithful *and* current. It also means
`a4_validitemname_rule.py` runs here in its live mode rather than its degraded
one — reading the real `nxdl.xsd` and confirming both the pattern and the
63-char cap against what the series records.

**Worth keeping:** a recorded hash is a tripwire for *later* corruption, not
evidence of *original* fidelity. The two look identical in a document and are
different claims, and only one of them had been checked.

### Claims verified against the schema, now that the referent is in-repo

`IO_prototype.rst`'s two resolved doubts are directly checkable against the
vendored definition, and both hold: `peaks/sx`, `sy`, `sz` are `minOccurs=1`
with `dims=['n_Peaks']`, exactly as the resolution states; and `vx`/`vy`/`vz`
are absent from the definition entirely, confirming "a PyRS extension — the
schema does not define them".

### An unremarked improvement in the diff

`.pre-commit-config.yaml`'s previous pattern was `"^pixi.lock$"` — an
**unescaped dot**, which as a regex matches any character. The new pattern
escapes both. This change quietly fixed a latent bug in a pre-existing entry,
which the commit message did not claim and nobody would have noticed.

### O36 — the exclusion comment was narrower than the exclusion

It said the file is excluded *"so whitespace/format hooks cannot alter it"*. But
`exclude:` is **top-level**: it removes the file from *every* hook. Measured
rather than reasoned — running the hooks against it reports `(no files to
check)` twelve times, including the read-only `check-xml` and `detect-secrets`.

Nothing is broken; the XML is well-formed, which I confirmed independently with
`ElementTree` since `check-xml` no longer does. But a reader was told the scope
was "whitespace/format" when it is "everything", and would have to run the hooks
to find out. The comment now says so, and says why that is acceptable: the
guarantee is upstream's, and the copy is verified by sha256 rather than by
re-validating it here.

### O37 — "NXDL version: `v2026.01`" contradicted the file it describes

The vendored file's own root carries `version="1.0"`, the NXDL
*schema-language* version; `v2026.01` is the definitions-repo *release tag*.
Both true, of different things — and a reader who opens the file to check the
provenance finds an apparent contradiction. Relabelled `definitions release`,
with the distinction stated inline.

### D69 — the developer documentation stopped at the prototype

Raised by the reviewer after the O36/O37 fixes:

> please re-read 'IO_prototype.rst', which is the developer documentation for
> the NXstress-I/O implementation. I think that we need to add a few
> sub-sections to this document describing the multi-workspace + append
> features (and their calling signatures) that we just added?

Correct, and it is the one gap a reader of this document would actually fall
into. `IO_prototype.rst` is organised by NXstress *group* — `PEAKS`, `FIT`,
`SAMPLE`, `INSTRUMENT` — because the prototype's question was "where does each
PyRS object go in the schema". 04b and 04c do not add groups; they change what
an entry *is* (N workspaces, not one) and what a write *does* (grow an entry,
not only create one). Nothing group-shaped changed, so nothing in a
group-shaped document did either, and the document silently went on describing
a one-workspace, write-once format.

Two sections added before `Possible extensions`, in the document's existing
voice and with its `Issues found:` convention:

- **Multiple workspaces in one `NXentry`** — `write(wss, peakss)` with the
  per-workspace `peakss` shape spelled out (including that one workspace with
  no fits is `[[]]`, not `[]` — the error 03 and 07 made in prose);
  `read` returning the same pair; `discriminator_fields` and the
  `merge_workspaces` fallback; and the scan-point / peak-index family split,
  which is the distinction that runs through the whole implementation.
- **Multiple `NXentry` groups in one file** — added after the reviewer noticed
  the first draft had missed it: *"That doesn't seem to include the fact that
  we can now write additional `NXentry`[s] (as an alternative append format)?"*
  Correct, and the omission mattered, because it is the axis the other two are
  **not**. A new entry is for a distinct reduction or sample condition;
  appending is for more of the same condition; and the compound peak index is
  for finer variation within one. Multi-direction stress data is the case most
  likely to be filed wrongly — it is *one* entry with `direction` as a
  discriminator, not one entry per direction.
- **Appending to an existing `NXentry`** — the mode-independent entry-targeting
  table, `entry_numbers()`, the three append outcomes with which of them
  invalidates the instance, and the three `Issues found:` that matter to a
  caller: every check runs before the first resize, a pre-04c file cannot be
  appended to, and extending an existing key is still unimplemented.

**One error caught by checking rather than recalling.** I first wrote the
compound key as `(discriminators, phase, h, k, l, mask, scan_point)`.
`IndexedPeaks.sort_key` returns `(*discriminators, phase, h, k, l, mask)` —
**no `scan_point`**. The distinction is the whole basis of the two row-order
guarantees: one key identifies one `PeakCollection`, and `scan_point` orders
rows *within* that key's run. Writing it the wrong way would have contradicted
the guarantee stated four paragraphs above it. `_peaks.py`'s own module
docstring is the trap — it describes the row *index* as
`(<phase>, h, k, l, <mask>, <scan point>)`, which is a different thing from the
grouping key and differs by exactly that element.

**The multi-entry section was written from measurement, not from reading.** A
script wrote an entry with `mode="w"`, added a second with
`mode="a", entry_number=2`, then listed and read both back. Three facts came
out of running it that would have been easy to state wrongly:

- on disk the first entry is named **`entry`**, not `entry_1`; later ones are
  `entry_2`, `entry_3`. Confirmed in the code afterwards —
  `numbers = {1} if GROUP_NAME.ENTRY in root else set()`, with `entry_N`
  suffixes added to it, which is also why `entry_numbers()` is correct on a
  file whose numbering has a gap;
- **omitting `entry_number` on an existing file appends; it does not add an
  entry.** The run proved it by failing the way an *append* fails — "the target
  entry carries no discriminator columns" — rather than quietly writing entry 2.
  This is the behavioural change most likely to surprise code written against
  the earlier implementation, and it now has its own `Issues found:` entry;
- `entry_number=9` against a two-entry file raises `ValueError` naming the next
  free number, as the targeting table says.

Everything else in the three sections was verified against the code as written:
the signatures, the targeting table (`NXstress.py:156-164`), that
`NotImplementedError` leaves the instance usable while the duplicate
`RuntimeError` sets `_invalid`, that `entry_numbers()` raises outside the
context manager, and that byte-neutrality really is asserted by digest —
`test_append.py:436-442`, not by `pytest.raises`.

### Verification

`pre-commit` re-run: the config still parses, the exclusion still holds (12
hooks skipped), and the edited files pass their own hooks. The vendored file's
sha256 is unchanged at `bee46c07…`, confirming nothing in this batch touched
it. `IO_prototype.rst` parses under `docutils` with no new warnings — the only
messages are the pre-existing Sphinx-only `:download:` role and an unreferenced
label, neither in the added range. No PyRS code changed; tiers not re-run.

---

## Batch 32 — `docs/ground_truths.md`

+249/−0 across five new sections. Four are my own writing from earlier batches,
so the risk here was re-reading them approvingly — the anti-pattern
`process.md` names first. Every claim was checked by running it.

### Everything substantive verifies, including the quoted error text

| Claim | Measured |
|---|---|
| contiguous dataset cannot be resized | `NeXusError: Shape incompatible with current NXfield` — the exact quoted message; `maxshape=(3,) chunks=None` |
| `growable(rank)` supplies the kwargs | `{'maxshape': (None,), 'chunks': (100,)}` |
| fixed-width `\|S` truncates **silently** | `b'a_considerably_longer_filename.h5'` → `b'a_consid'`, no exception |
| `<U` cannot be written to HDF5 | `TypeError: No conversion path for dtype: dtype('<U2')` |
| `SubRuns` rejects non-monotonic | `RuntimeError: subruns are not sorted in increasing order`; `__init__` routes through `set`; `append` bypasses it |
| `append_hidra_project` takes a file | `(self, hidra_file)` |
| vlen UTF-8 reads back object-of-bytes | `dtype=object kind='O' elem=bytes`, from **both** h5py and `nexusformat` |
| `READONLY` unconditional + `finally` | `reduction_manager.py:188`, `:200` |
| stage-convention numbers | probe: 2880 candidates, 2 matches, `5.818e-14`, `diag(1,-1,-1)` = 180° about `+x` |

Both cross-referenced test classes are in the files claimed —
`TestStringLogsAreTextInMemory` in `dataobjects/`, `TestStringLogsAreBytesOnDisk`
in `projectfile/`, near-identical filenames and the right way round — and
`TestLoadDoesNotModifyTheFile` really does cover the failing and succeeding
paths separately, as its entry says.

### D70 — one section had no date

Every other heading carries one, including the four other additions, and the
dates are what let the file be read as an accumulating record rather than a
pile. The stage-convention section had none. Added `(2026-10-14)`; all headings
are now dated, checked by pattern rather than by eye.

### O38 — four of the five had no inbound link

The file's own header says *"Link to this file from related documentation and
code comments when relevant"*, and `CLAUDE.md` says the same. Only the
string-dtype truth was linked, from `file_object.py` and `sample_logs.py`.

**A ground truth nobody is routed to is one that gets rediscovered** — which is
precisely what the file exists to prevent, so an unlinked entry is closer to a
private note than to a ground truth. Links added at the place each rule would
be needed, not at a convenient one:

- `_definitions.py:growable` — the function where resizability is decided, and
  the decision cannot be revisited later;
- `reduction_manager.py` — above the `READONLY` open, stating the general rule
  (choose the mode from what the operation *does*, never from what the
  filesystem *permits*) before the local justification;
- `_sample.py:TRANSFORMATION_CHAIN` — beside the recovered axes, carrying the
  generalisable half: an underdetermined decomposition may mean the candidate
  space is too small rather than the referent ambiguous;
- `sample_logs.py` — at the `raise` itself, noting that `__init__` routes
  through it, that `append` bypasses it, and what that forces on any
  workspace merge.

### Verification

ruff format and check clean on all four touched files; mypy clean. Tiers re-run
because production files changed: `test-unit`, `test-integration` and
`test-gui` all pass. Every heading in `docs/ground_truths.md` now matches the
date pattern, and all five new sections have at least one inbound link.

---

## Close-out — final audit pass, ready for subspec 05 (2026-10-16)

The 32-batch walk is complete through 04c. This records the handoff tasks and
the two contradictions found while doing them.

### `audit.toml`

`base_commit` advanced to **`eb5457b16dab`** ("Working subspec-implementation
prompt, now *applying* the audit process."), with `base_subject` updated in the
same edit — they are cross-checked at load, so a half-update makes every tool
exit 2. `04c-*.md` moved from `[scope] full` to `freshness_only`; the manifest
loads with 12 subspecs classified exactly once.

**The stated rule had to be rewritten, not just the value.** Both `audit.toml`
and the prompt said the range should span *"exactly the previous subspec's
changes"*. `eb5457b16dab` spans four subspecs and 43 commits, so the value the
reviewer chose and the rule as written contradicted each other.

This is the **D56 shape with the roles reversed**. In batch 28 I treated a
rationale as a specification and reported the value as wrong; the reviewer
corrected me. Here the value is deliberate — subspec 05's landing pass must see
every change the review made underneath it, and the 04 series was implemented
*and reviewed* as one unit — so leaving the old rule in place would have
recreated the same defect pointing the other way. The rule now reads: span
**every landed change the next subspec must see**, usually one subspec but a
whole series when one was reviewed as a unit; never a merge commit, because that
leaves an empty range; and too wide is the benign direction.

Measured and written into the prompt so the next session is not alarmed by it:
`landing_trigger.py` now reports **390 citations into 24 changed files across 17
documents** — an order of magnitude more than previous passes, with the heaviest
being `_peaks.py` (74), `_fit.py` (47), `_instrument.py` (45),
`file_object.py` (45), `_sample.py` (33). That is the range working.

### Decisions row 9 was contradicted by shipped code, and 05 is the spec it names

Row 9: *"`HidraWorkspace` gains a **settable** `direction` `@property`
(get/set)"*, "Spec: 05". Measured: `direction.fset is None`, assignment raises
`AttributeError`, and mypy reports the property read-only. What landed is a
read-only property marked `@restores("_direction")` — the better design, and one
that survived two independent reviews — but **nothing said row 9 was
superseded**, and row 9 is what a session starting 05 reads first.

Added **row 42**, which supersedes it and explains why; row 9 now opens with a
forward pointer rather than being left to mislead. Row 9's *other* half stands
and is still 05's work: flipping `nxstress.discriminator_fields` from `[]` to
`["direction"]`, which is only safe **because** `direction` is marked restorable
— `field_names()` refuses a read-only property that is not.

**The general shape:** a Decisions Log records what was decided, and shipped code
can silently stop matching it. Nothing in the toolkit checks that — `check_*`
verify links, citations, ownership and drift, none of which notices a decision
the code has outgrown. The only thing that caught this was reading row 9 while
preparing the spec that points at it.

### Prompt

`prompts/implementation-prompt.txt`'s header rewritten for 05: SUBSPEC, LANDED
THROUGH 04c, the new AUDIT BASE with its measured signal, and LAST AUDIT PASS
replaced by a briefing on what the review changed *underneath* 05 — the
read-only `direction`, the still-unflipped config default, `apply()`'s new
refusal, the list-shaped `read`/`write`, and append. 05's own documents were not
touched by the review; the code under them was, which is exactly the trap the
briefing exists to prevent.

05's dependencies (01, 04, 04b) have all landed, as has 04c. **05 is unblocked.**
