# PR review process — human review of a landed diff

**Status:** adapted for PyRS, 2026-10-06. Not yet exercised.

This document came from another project. Every path, tool name, command,
measurement and file reference below has been re-checked against PyRS as it stands
at `cda22352`; the places where the original was wrong for this repo are noted
where they matter. The questions that had to be settled before starting are marked
**AGREED** with what was decided.

## This is not the audit process

[`plans/audit-process/process.md`](../audit-process/process.md) and `CLAUDE.md`'s
*"Auditing a Plan or Subspec"* — specifically its *"What a complete audit consists
of"* — describe a **plan audit**: seven named axes, a coverage obligation, and an
evidentiary standard (*probe before asserting*). This document describes something
different and deliberately looser — **a human reading a diff**.

| | plan audit | this |
|---|---|---|
| subject | the subspec, before/independent of code | the diff, after it lands |
| driven by | the axis checklist; incomplete until all seven are covered | the reviewer's attention, wherever it goes |
| completeness | a defined property; a partial audit must be recorded as partial | not a property. The reviewer stops when satisfied |
| evidence | a claim is unverified until executed | the reviewer may comment on anything, including taste |
| output | findings, not code | comments, and the changes they produce |

Nothing here imposes an axis, requires a probe, or makes a file's review
"incomplete". The reviewer sets the depth, per file. The **audit** pass for this PR —
`landing_trigger.py` first, then `check_links.py`, `check_citations.py` and
`check_ownership.py` against the baseline recorded in
[`review/findings.md`](../NXstress-prod/review/findings.md) §2 (`check_drift.py`
alongside them, with the same recorded residue) — runs **after** review concludes
and is tracked separately.

---

## 1. This review

| | |
|---|---|
| **Base commit** | `eb5457b1` — *"Working subspec-implementation prompt, now \*applying\* the audit process."*, 2026-09-26 |
| **Head** | `cda22352` — *".. subspec 04c"*. **The range is pinned here, not at `HEAD`**: `ef85e9c0` added this document, which is process rather than work under review. `eb5457b1..HEAD` is 67 files; `eb5457b1..cda22352` is the 66 that belong to the PR |
| **Subspecs under review** | `04-nxstress-internal-cleanup.md`, `04b-multi-workspace-nxstress.md`, and `04c-nxstress-append.md`, together |
| **Scope** | **66 files, +10885 / −557**, across seven commits |
| **Reviewer's view** | per-batch `git difftool` commands in the §6a order — *not* one walk over the whole range |

The base is verified as the start of `04`'s work: `eb5457b1` is the commit
immediately before `7a3b57f6 subspec 04`, so the range contains all three subspecs,
their two interleaved audit passes, and the integration-test fixups that landed
between `04b` and `04c` — and nothing from `03` or earlier.

The seven commits:

```
cda22352  .. subspec 04c
9c676f8a  subspec 04c
61389b28  ... integration-test fixups: only open fixture files in READONLY mode
79095cec  .. defer PR review: subspec 04b: final audit pass, ready for 04c
7799a673  subspec 04b
d69e1834  .. defer PR review: subspec 04: final audit pass, ready for 04b
7a3b57f6  subspec 04
```

`61389b28` belongs to none of the three subspecs — see §4.

**Precondition.** `difftool.prompt` is unset in this repo, so git prompts before
every file. Run once, before starting:

```bash
git config difftool.prompt false
```

Without it, each command below asks for confirmation per file, 66 times.

**Why the commands below name only the base.** `git difftool eb5457b1 -- <paths>`
compares the base against the **working tree**, not against `cda22352`, so a file
changed during the review shows its new state when re-opened. That is the behaviour we
want mid-review. It is safe here because no batch names a file outside the 66 — this
document is in no batch.

---

## 2. The sequence, per batch

Files are reviewed in batches: a **batch** is one file, or a cluster that only makes
sense together (§6b). The overall walk follows the **code-first order agreed in
§6a**, not git's own diff order.

0. **I give the exact command for the batch**, so the reviewer opens precisely those
   files rather than hunting for them in a 66-file walk:

   ```bash
   git difftool eb5457b1 -- \
       pyrs/utilities/NXstress/_peaks.py \
       tests/unit/pyrs/utilities/NXstress/test_peaks.py \
       tests/unit/pyrs/utilities/NXstress/test_peaks_read.py
   ```

   This repo's `diff.tool` is `meld`; with `difftool.prompt=false` set per §1, the
   command opens the files directly with no per-file confirmation.

   **`difftool` walks a pathspec in git's sorted order, not the order it is given** —
   re-verified in this repo, not inherited: `_sample.py _definitions.py _peaks.py`
   and `_peaks.py _definitions.py _sample.py` both come back as `_definitions`,
   `_peaks`, `_sample`. So argument order is not a way to control reading order.
   **Splitting the invocation is.**

   One command per batch is therefore the *convenient* case, not the rule. The rule is:
   **however many commands it takes to present the files in the order they should be read,
   issued in that order** — and since a single file is always a legal pathspec, one command
   per file is the base case we fall back to whenever order matters more than round trips:

   ```bash
   git difftool eb5457b1 -- pyrs/utilities/NXstress/_definitions.py   # the leaf, first
   git difftool eb5457b1 -- pyrs/utilities/NXstress/_discriminator.py # then its consumer
   git difftool eb5457b1 -- pyrs/utilities/NXstress/_peaks.py         # then the caller
   ```

   My summary follows whatever order the commands establish, so the reviewer's tabs and my
   paragraphs always line up.

1. **Open the comments document** for the subspec this file belongs to:
   `plans/PR_review/<subspec>-comments.md` — `04-comments.md`, `04b-comments.md`,
   `04c-comments.md`. Created on first substantive file, not up front.

2. **Skip a file whose changes are only cosmetic**, and say so in one line rather than
   silently. If every file in a subspec is cosmetic, no comments document is created.

3. **Announce the file, then summarise it** — a few paragraphs on what changed and why,
   written for someone reading the diff alongside. **And state any defects or issues I
   find**, separately from the summary so they are not buried in it.

4. **Wait.** The reviewer types comments. I do not proceed to the next file, and do not
   edit anything, until they have.

5. **Discuss before implementing.** Any change the comments imply is proposed and agreed
   first — what I intend to change, where, and what it affects. No edits land from step 3
   or 4 alone.

6. **Record in the comments document**: the reviewer's comment, the agreed resolution,
   and the change actually made.

7. **Record in the subspec**, by appending to its `## Follow-up N` section: a summary of
   **the change**, not of the review conversation. The reviewer's words are included only
   where they are needed for the change to make sense.

   The three subspecs are already at Follow-up 7 (`04`), Follow-up 2 (`04b`) and
   Follow-up 3 (`04c`), so review changes open Follow-up 8, 3 and 4 respectively.
   Earlier Follow-ups are never edited.

8. **Report what changed, and wait again.** A batch is not finished when the edits
   land. I state what actually changed — each comment, the resolution, the edit — and
   **stop**. The next batch is not opened until the reviewer has accepted this one.

   **One message must never carry both the close of one batch and the opening of the
   next.** Doing so buries the thing the reviewer still has to approve underneath a
   wall of new material, and removes the only point at which they can say "that is not
   what I meant" before the walk has already moved on. Step 4's "wait" is not spent by
   the first round of comments: there is a wait before the work and a wait after it.

   *(Added after batch 1, where exactly this happened: the `D1`/`D2`/`D3` resolutions
   were reported in the same message that opened batch 2's summary.)*

9. **Stage on request, and draft the commit message. Reviewer-triggered, and specific
   to this process.** When the reviewer says *"please stage the changes"*: `git add`
   the changed files and **stop there — never `git commit`.** Then give the message,
   and nothing but the message.

   **Scope.** This step belongs to the PR review only. **Subspec implementation is
   unchanged**: there I neither stage nor commit nor draft a message, and leave the
   working tree for the reviewer. The difference is that a review change is small,
   already agreed, and already written up — so the message is a transcription rather
   than a judgement.

### 2.9.1 The format, exactly

```
PR review: subspec <list>: part <batch number>

- <what changed>
- <what changed>
```

Colons separate the three fields; the comma is used **only** inside the subspec list,
so the two punctuation marks never mean the same thing.

- **`<list>`** — the subspecs this batch actually changed, comma-separated:
  `04c`, or `04, 04c`, or `04, 04b, 04c`. Only the ones that changed, not the ones
  the files belong to.
- **`<batch number>`** — the Appendix A batch number, bare and not zero-padded. It
  identifies *which batch* the commit came from, so it is the same number whichever
  subspecs the commit covers, and a reader can find the batch it corresponds to.

**One commit may span subspecs, because this review does.** Whether to combine is
decided by the review's own scope, not per commit: `04`, `04b` and `04c` are being
reviewed together, so a batch whose changes land in more than one of them is a single
commit naming each. A review scoped to one subspec would naturally produce
single-subspec commits for the same reason.

```
PR review: subspec 04, 04c: part 6        a batch that changed both
PR review: subspec 04c: part 1            a batch that changed only 04c
```

### 2.9.2 Conciseness, as a budget rather than an adjective

Conciseness has been a recurring failure, so it is specified as something checkable
rather than aspired to:

- **One bullet per change**, not per finding. Three defects fixed by one edit is one
  bullet.
- **At most two sentences per bullet**, wrapped at 72 columns.
- **At most eight bullets.** More than that means the batch was too large to review in
  one pass — a `RE-REVIEW RECOMMENDED` signal (§5), not a longer message.
- **Count the bullets and lines before presenting.** Over budget means cut, not
  present-with-an-apology.

**The derivation rule, which is the part that actually works.** The bullets are the
comments document's `**Change:**` lines, compressed — **and nothing else**. That is
mechanical, and it closes the gap where a message grows: re-narrating the review. If a
sentence is not already in a `**Change:**` line, it does not belong in the commit.

### 2.9.3 What must not appear

Rationale beyond a short subordinate clause · decisions taken and alternatives
rejected · probe output, measurements, counts or timings · counter-examples ·
references to Follow-up sections, the comments documents or the Decisions Log · test
results · anything about the review conversation.

*Why* is already recorded in three places — the comments document, the subspec's
Follow-up, and the Decisions Log. The commit message is the one artifact that only has
to say *what*.

### 2.9.4 No attribution — mandatory

**No `Co-Authored-By`, no `Signed-off-by`, no "Generated with" line, no trailing
attribution of any kind.** Attribution is handled separately and never per commit.
This overrides any default, tooling convention or standing instruction that would
otherwise append one.

### 2.9.5 Worked example — batch 1

Batch 1 produced three changes, all in `04c`'s scope — `_definitions.py` also carries
`04`'s identifier work, but the `__` exchange there resolved to no change, so `04` is
not named:

```
PR review: subspec 04c: part 1

- `tail_append`'s error messages named the dataset `'unknown'` for a field
  not attached to a tree; added `_field_label` and used it in all three.
- Guarded `tail_append` against a scalar `values`, which previously raised
  `IndexError` naming neither the field nor the problem.
- Documented at `growable` that the unlimited trailing axes are a chunking
  artifact rather than a contract, and added that reason to `tail_append`'s
  trailing-axis message.
```

Three bullets, eleven lines, no attribution. What it leaves out is the point: the
h5py chunk constraint, the three measured `maxshape`/`chunks` cases, the spec-09
placeholder reason, the rejected pinning alternative, the `__` encoding exchange, and
the test counts — all of which are in `04c-comments.md` and `04c`'s Follow-up 4.

---

## 3. What "cosmetic" means

Step 2 is the only step that can lose something, so the test is stated rather than left
to judgement:

**Cosmetic** — a change that alters no behaviour, no claim's truth value, and no
obligation on anybody:

- a `path:LINE` citation re-pointed at the same unchanged content. PyRS has no
  symbolic-anchor convention — `check_citations.py` resolves literal `path:LINE`
  tokens — so re-pointing *is* the whole of the fix. The series rule is that a
  pointer is corrected in place while the belief it supports stays in its
  `## Follow-up N`
- a typo, a reflowed paragraph, a widened table column
- a comment reworded without changing what it asserts

**Not cosmetic**, and therefore summarised even when the diff looks small:

- any changed number, even in prose — a budget, a count, a measurement, a size
- a reworded invariant, policy, or guarantee, however slight
- a renamed symbol, moved module, or changed import
- a deleted sentence (what it said may have been load-bearing)
- a changed pasted probe output, or a changed `## Verification` measurement

When in doubt it is not cosmetic. The cost of over-summarising is the reviewer's time;
the cost of under-summarising is a silent skip.

### One file meld cannot show

`tests/data/HB2B_1327_with_instrument.h5` is a **binary HDF5 fixture**, added in this
diff. It has no readable diff, so it is neither summarised nor skipped as cosmetic.
It is reviewed through the script that generates it —
`tests/scripts/make_nxstress_sample_position_fixture.py` — and through its consumer,
`tests/integration/test_nxstress_sample_position.py`, which are batched with it. What
the reviewer needs to judge is the *provenance*: what the fixture contains, which run
it derives from, and why it had to be committed rather than generated at test time.

---

## 4. Which comments document a file belongs to — AGREED

One document per subspec: `plans/PR_review/04-comments.md`, `04b-comments.md`,
`04c-comments.md`. A file is attributed to the subspec whose work introduced the
change under review — not to the subspec that first created the file.

Attribution was worked out per file from `git log <base>..HEAD -- <path>`:

| Work | Files | Document |
|---|---|---|
| `04` (`7a3b57f6`) | `.pre-commit-config.yaml`, `tests/unit/pyrs/projectfile/{__init__,test_file_object}.py`, the NXstress identifier/mask/transformation changes, `docs/.../NXstress.nxdl.xml`, `docs/.../IO_prototype.rst` | `04-comments.md` |
| `04b` (`7799a673`) | `pyrs/utilities/config.py`, `tests/unit/pyrs/utilities/test_config.py`, the three `pyrs/interface/*_model.py`, `_discriminator.py` + its test, `test_multi_workspace.py`, `tests/integration/test_nxstress_viewer_roundtrip.py` | `04b-comments.md` |
| `04c` (`9c676f8a`, `cda22352`) | `_definitions.py`'s `growable`/`tail_append`/`appendable`, the append paths across the NXstress package, `test_append.py` | `04c-comments.md` |

**Work belonging to no subspec**, because it landed between `04b` and `04c` as a
repo-wide fix (`61389b28`, "integration-test fixups: only open fixture files in
READONLY mode"): `pyrs/core/reduction_manager.py` and
`tests/integration/test_pyrscore.py`. **Decided: these go in `04c-comments.md`**, as
the document for the commit range's tail. They are flagged there as belonging to no
subspec, so a reader does not mistake them for append work, and any Follow-up they
generate goes to `docs/ground_truths.md` or a new follow-on spec rather than to
`04c`'s Follow-up 4.

**Files touched by more than one subspec** — `docs/ground_truths.md`,
`pyrs/resources/application.yml`, `tests/util/peak_collection_helpers.py`, and every
`plans/NXstress-prod/*.md` — are discussed once, in the document for whichever
subspec's change is under discussion, splitting across documents if the reviewer's
comments span more than one. The sibling-doc edits to `03`, `07` and `10` came in
during the `04` and `04b` commits and follow the same rule.

---

## 5. Two standing signals, because there is probably no second pass

We are unlikely to run another review pass over this diff. So two things must be raised
explicitly rather than left for a later reader to notice, and both are recorded in the
comments document so they survive the conversation:

### `OPEN DEFECT`

Something I flagged that was not resolved — either the reviewer did not comment on it, or
we agreed to leave it. Recorded whether or not it was acted on, with what would resolve
it. Raised again at the end of each subspec's walk (§6c).

### `RE-REVIEW RECOMMENDED`

A judgement that a set of files has become complicated enough that one pass is not
enough — **raised by either of us**, and about the *work*, not about anyone's attention.
I should raise it when:

- **A change made late in the walk invalidates something summarised early.** The common
  case: a symbol renamed or a contract changed at file 50, after the doc asserting it was
  reviewed at file 14. Whoever reviewed it early reviewed a different thing.
- **A fix agreed during review outgrew its estimate.** A side fix that outgrows its host
  makes the host unreviewable; inside a review, the host is the review. (Stated here
  rather than quoted — PyRS's `CLAUDE.md` carries no such rule, though the sub-agent
  round recorded in `04c`'s Follow-up 3 is what the risk looks like in practice.)
- **Coupled changes landed in batches reviewed at different times**, so no single step
  ever saw the whole of the behaviour.
- **I implemented something non-trivial during the review and the only verification was a
  targeted test.** Reviewing a diff I just wrote, in the same pass, is the weakest
  configuration this process has.
- **I am not confident in my own summary.** Saying so is cheaper than having it believed.

The flag names *which files*, and *why*, so a second pass can be scoped rather than
repeated wholesale. It is not a failure of the review; it is the review working.

---

## 6. Questions settled before starting

All six are decided; none remain open.

**6a. Code before docs — AGREED.** Git's order puts **every** plan document at files
5–29 and **every** source file at 31–44, so following it would review all 25 subspec
documents 15-odd files before any of the code they describe. That is the wrong way
round: a claim reads as true until the code contradicts it, and `04c`'s Follow-ups 2
and 3 exist precisely because several claims were false. **Decided: walk the code
first.** The order is:

```
1. pyrs/utilities/NXstress/*        the library this PR is about          (8 files)
2. pyrs/{core,interface,resources,utilities}/*  callers and configuration (6 files)
3. tests/**                                                              (22 files)
4. plans/**                 subspecs, probes, findings, manifest, 04_commit.txt
                                                                         (26 files)
5. docs/**  and  .pre-commit-config.yaml                                  (4 files)
```

66 files, all accounted for. Within each stage, batches follow §6b. Because this is
not git's order, **every batch needs an explicit command** — which §2 step 0 requires
regardless.

Note `plans/PR_review/04_commit.txt`, a 360-line PR description written for the
reviewer. It is deliberately in stage 4 rather than read first: as narrative *about*
the change it would prime the review with my account before the code is seen, which
is the failure §6a exists to avoid. Read after the code, it is checkable against it.

**6b. Batching coupled files — AGREED.** 66 files at one round trip each is 66 round
trips, and some are inseparable. The clusters in this diff:

- *stage 1*: `_definitions.py` → `_discriminator.py` → `_peaks.py` (leaf to caller);
  `_fit.py` + `_instrument.py` + `_sample.py` + `_input_data.py` (the four group
  modules); `NXstress.py` alone, last, since it orchestrates all of them
- *stage 2*: `pyrs/utilities/config.py` + `pyrs/resources/application.yml` (the config
  mechanism and the keys it reads); the three `pyrs/interface/*_model.py` (one change,
  three viewers); `pyrs/core/reduction_manager.py` alone (belongs to no subspec)
- *stage 3*: each NXstress unit test with nothing else, since its module was already
  read in stage 1; `test_peaks.py` + `test_peaks_read.py` together; the fixture triple
  `tests/data/HB2B_1327_with_instrument.h5` + `tests/scripts/make_nxstress_sample_position_fixture.py`
  + `tests/integration/test_nxstress_sample_position.py` (§3)
- *stage 4*: `probes/README.md` + the twelve probe files; the three subspecs with their
  `open-questions/` counterparts

Batching is agreed, **and step 0 is what makes it work**: batching alone would leave the
reviewer's `difftool` lagging behind my summaries, hunting for which files I meant. I give
the exact `git difftool … -- <paths>` command instead. Every file in the batch is named in
the command and in the summary, so nothing hides inside a group, and the reviewer can ask
to split one out.

**Batching is a round-trip optimisation and it yields to reading order**, which is the
property that actually affects review quality. Because `difftool` sorts within one
invocation, a batch whose files want reading in a non-alphabetical order is presented as
several commands — down to one per file, the base case. Grouping six files into one command
that then opens them in the wrong order would spend the reviewer's attention to save my
round trips, which is the wrong trade.

**6c. A defect I flag that the reviewer does not comment on — AGREED.** Options: (i) I ask
explicitly before moving on, (ii) it is recorded as **open** in the comments document and
we return to it at the end, (iii) it is dropped. **Decided: (ii) plus (i)** — recorded as
open, and raised again at the end of the subspec rather than at every file, so it neither
evaporates nor interrupts the walk.

**6d. When tests run.** The original document assumed a ~7.5-minute unit suite that
could not run per file. **That is not this repo**: measured at `cda22352`,
`pixi run test-unit` is **17 s** (412 passed), `pixi run test-integration` **2 m 27 s**
(104 passed, 28 skipped, 2 xfailed) and `pixi run test-gui` **2 m 24 s** (16 passed).
Nothing costs money. Revised proposal: **`pixi run test-unit` after every implemented
change**, since 17 s is cheaper than the risk of batching failures; `test-integration`
and `test-gui` at the end of each subspec's walk, and whenever a change touches a file
either tier exercises. Note `pixi run test` does **not** set `QT_QPA_PLATFORM=offscreen`
— export it, or the GUI tier hangs to the 300 s timeout (`docs/ground_truths.md`).

**6e. Do my summaries get persisted? — AGREED.** Step 6 records the reviewer's comment, the
resolution, and the change. **Decided: not** the executive summaries — they are long,
and the diff is the better record of what changed. The exception is a **defect I flagged**,
which is recorded whether or not it was acted on, since that is the finding rather than
narration of the diff.

**6f. What if a comment is out of scope for this PR? — AGREED.** Some will be — the diff touches
`03` through `10`, and the series runs `01`–`10`. **Decided:** record it in the comments
document and, if it needs doing, name the subspec that should carry it, exactly as a
flagged invariant does. Not silently absorbed, and not silently dropped. A finding that
is about the *code or the environment* rather than about a document goes to
`docs/ground_truths.md` as a short durable fact pointing at the Follow-up that holds the
evidence — `README.md`'s Follow-up 3 F3.4, not a copy of the finding.

---

## Appendix A. The batch plan

32 batches over 66 files, in the §6a stage order. Derived from the real intra-package
import graph, not from directory names: `_definitions` imports nothing in-package;
`_discriminator`, `_sample`, `_instrument` and `_input_data` import only it; `_fit`
imports `IndexedPeaks` from `_peaks` at module level while `_peaks` imports `_fit`
only inside a function, so `_peaks` is the earlier of that pair; `NXstress.py` imports
all seven.

Where a batch would open files in the wrong order (difftool sorts within one
invocation), it is split — marked **→** below.

| # | Stage | Files | Note |
|---|---|---|---|
| 1 | 1 | `_definitions.py` | leaf; `04` identifier policy + `04c` growth helpers |
| 2 | 1 | `_discriminator.py` | new in `04b` |
| 3 | 1 | `_peaks.py` | **→** before `_fit`, which imports from it |
| 4 | 1 | `_fit.py` | |
| 5 | 1 | `_input_data.py` `_instrument.py` `_sample.py` | mutually independent |
| 6 | 1 | `NXstress.py` | orchestrator, last |
| 7 | 2 | `resources/application.yml` `utilities/config.py` | keys, then their reader |
| 8 | 2 | the three `interface/*_model.py` | one change, three viewers |
| 9 | 2 | `core/reduction_manager.py` | **no subspec** (`61389b28`) |
| 10 | 3 | `NXstress/conftest.py` | fixtures everything else uses |
| 11 | 3 | `tests/util/peak_collection_helpers.py` | the other shared helper |
| 12 | 3 | `test_definitions.py` | pairs with 1 |
| 13 | 3 | `test_discriminator.py` | pairs with 2 |
| 14 | 3 | `test_peaks.py` `test_peaks_read.py` | pairs with 3 |
| 15 | 3 | `test_fit.py` | pairs with 4 |
| 16 | 3 | `test_input_data.py` `test_instrument.py` `test_sample.py` | pairs with 5 |
| 17 | 3 | `test_NXstress.py` `test_append.py` `test_multi_workspace.py` `test_workspace_read.py` | pairs with 6 |
| 18 | 3 | `projectfile/__init__.py` `projectfile/test_file_object.py` | `04`'s FIXME invariant |
| 19 | 3 | `utilities/test_config.py` | pairs with 7 |
| 20 | 3 | `scripts/make_nxstress_sample_position_fixture.py` **→** `integration/test_nxstress_sample_position.py` **→** `data/HB2B_1327_with_instrument.h5` | generator, consumer, then the binary (§3) |
| 21 | 3 | `integration/test_nxstress_viewer_roundtrip.py` | |
| 22 | 3 | `integration/test_pyrscore.py` | **no subspec** (`61389b28`) |
| 23 | 4 | `NXstress-prod/README.md` | Decisions Log; everything references it |
| 24 | 4 | `04-*.md` + `open-questions/04-*.md` | |
| 25 | 4 | `04b-*.md` + `open-questions/04b-*.md` | |
| 26 | 4 | `04c-*.md` | no open-questions change |
| 27 | 4 | `probes/README.md` + the twelve probe files | |
| 28 | 4 | `review/findings.md` `audit.toml` `prompts/implementation-prompt.txt` | |
| 29 | 4 | `03-*.md` `07-*.md` `10-*.md` | sibling-doc notes |
| 30 | 4 | `PR_review/04_commit.txt` | narrative; deliberately late (§6a) |
| 31 | 5 | `IO_prototype.rst` `NXstress.nxdl.xml` `.pre-commit-config.yaml` | vendored schema, provenance, its hook exclusion |
| 32 | 5 | `docs/ground_truths.md` | |

File counts per stage: 8 / 6 / 22 / 26 / 4 = 66.

### Corrections found while walking it

The plan was derived from the diff before the walk began, and two rows have
turned out not to match it. Recorded here rather than silently edited, because a
batch plan that is quietly corrected cannot be checked afterwards against what
was actually reviewed.

| Row | As planned | As found |
|---|---|---|
| 8 | "the three `interface/*_model.py`" | **Four.** `strainstressviewer/model.py` joined during batch 2, after this plan was written. All four were walked. |
| 18 | `projectfile/__init__.py` + `projectfile/test_file_object.py` | **`__init__.py` was never changed by the PR** -- `git log eb5457b1..cda22352 -- pyrs/projectfile/__init__.py` is empty. Only the test file was walked. |

Neither changes the batch *count* or the order. The lesson for the next series is
that a file list drawn from a diff goes stale as soon as the review starts
changing files, so the walk should re-derive each batch's membership from the
range at the moment it opens the batch -- which is what `git difftool <base> --
<paths>` does anyway, and is why both were caught.
