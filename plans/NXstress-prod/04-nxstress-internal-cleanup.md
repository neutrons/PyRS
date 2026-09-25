# 04 — NXstress Internal Cleanup (Phase 2 TODOs)

**Plan:** [NXstress GUI Hookup](README.md)
**Phase:** 2
**Depends on:** [02](02-peak-and-texture-nxstress.md), [03](03-combine-runs-nxstress.md)

---

## Overview

Close the NXstress-internal TODOs from Section 2.1 of the README that require
no PyRS data-model changes. After this spec, NXstress files produced by the
GUI viewers will contain correct sample positions, consistent mask naming, a
parameterized instrument name, and verified scan-orientation data. The `NaN`
placeholders for sx/sy/sz are replaced with real values.

---

## Scope

**In scope:**
- Restore `sx`/`sy`/`sz` from `SampleLogs` in `_peaks.py:235-239`
- Reconcile default vs. named mask storage convention (`_fit.py:546-549`)
- Complete disallowed-character coverage in `_definitions.py::allowed_identifier` (L229)
- Parameterize instrument name in `_instrument.py:70`
- Detector rotation-order cross-check and fix in `_instrument.py:165`
- Audit `HidraProjectFile` legacy log-name patches (`file_object.py:404`, `:494`)
  and confirm / update the key names NXstress emits

**Out of scope:**
- Flipping `nxstress.use_production_names` default to `true` — deferred until
  the upstream `nexusformat` validator bug is resolved; when it is, simply
  update `pyrs/resources/application.yml`.
- Any PyRS data-model changes.
- Any GUI viewer changes.

---

## PyRS Changes

- `pyrs/projectfile/file_object.py` — audit lines 404 and 494 (legacy
  log-name FIXMEs). Determine the canonical log names the reduction pipeline
  emits. No code changes required if the audit confirms NXstress already uses
  the correct names; otherwise patch the relevant `_sample.py` or
  `_peaks.py` key lookups.

---

## NXstress Changes

### `pyrs/utilities/NXstress/_peaks.py` — restore sx/sy/sz (L235-239)

Uncomment and fix the `logs['sx']`/`logs['sy']`/`logs['sz']` block.
Reconcile the log-key names against what the reduction pipeline actually
stores in `SampleLogs`. If the keys differ from `'sx'`/`'sy'`/`'sz'`,
use the correct names (or fall back to `NaN` gracefully if the keys are
absent, with a logged warning rather than a hard failure).

### `pyrs/utilities/NXstress/_fit.py` — mask naming convention (L546-549)

Define a single convention for storing default vs. named masks:
- The default mask always uses the key `DEFAULT_TAG`
  (`HidraConstants.DEFAULT_MASK`) in the diffractogram dict.
- Named masks use the mask name directly.
Apply this convention consistently in both the writer (`init_group`) and the
reader (`diffractogramFromNexus`, `masksFromNexus`). Add a test that writes
a file with both default and named masks and reads it back.

### `pyrs/utilities/NXstress/_definitions.py` — allowed_identifier (L229)

Extend `allowed_identifier` to cover at minimum `$`, whitespace, and any
other characters disallowed by the NXstress/HDF5 group-name rules. Add unit
tests for all newly-covered cases.

### `pyrs/utilities/NXstress/_instrument.py` — instrument name (L70)

Replace the hardcoded `"HB2B"` with a value drawn from the `HidraWorkspace`
(e.g., from a sample-log entry or a new workspace attribute). If no instrument
name is available, fall back to `"HB2B"` with a logged warning. This makes
the writer usable at other beamlines without a code change.

### `pyrs/utilities/NXstress/_instrument.py` — rotation order (L165)

Cross-check the rotation sequence used in `NXtransformations` against the
`DENEXDetectorGeometry` conventions in `pyrs/core/instrument_geometry.py`.
Fix the order if it is wrong. Add a regression test that writes a geometry,
reads it back, and asserts the rotation components are numerically equal
(within floating-point tolerance).

---

## Tests

- Add / extend tests in `tests/unit/pyrs/utilities/NXstress/test_instrument.py`
  for rotation-order correctness and instrument-name parameterization.
- Add tests in `tests/unit/pyrs/utilities/NXstress/test_peaks.py` for
  sx/sy/sz round-trip.
- Add tests in `tests/unit/pyrs/utilities/NXstress/test_fit.py` for the
  unified mask-naming convention.
- Add tests in `tests/unit/pyrs/utilities/NXstress/test_definitions.py` for
  the extended `allowed_identifier`.

---

## Delivered Feature

> **For end users and downstream NXstress consumers:**
> NXstress files produced by PyRS now contain more complete and correct data:
>
> - **Sample positions** (sx, sy, sz) are populated from the measurement logs
>   rather than being left as `NaN`.
> - **Mask naming** is consistent — the same mask name means the same thing
>   in every part of the file, whether the default mask or a named user mask.
> - **Instrument name** is taken from the workspace rather than being hardcoded,
>   making PyRS NXstress files usable at beamlines other than HB2B.
> - **Detector orientation** has been verified against the reduction pipeline's
>   rotation convention.
>
> These improvements make PyRS NXstress files more reliably readable by
> external NXstress-aware software.

---

## Verification

- `pytest tests/unit/pyrs/utilities/NXstress/` — all tests pass including new
  ones added in this spec.
- Write a `.nxs` file from a real HB2B dataset; inspect sx/sy/sz fields
  with `h5dump` or the `nexusformat` Python API — confirm they are non-NaN.
- Write a `.nxs` file and open it in the `nexusformat` validator
  (with `nxstress.use_production_names = true` temporarily, to exercise
  the production-name path even before it becomes the default).

---

## Follow-up 1 — 2026-09-25 (first seven-axis pass)

**F1.1** (A3/A7) — Scope: "Audit `HidraProjectFile` legacy log-name patches
(`file_object.py:404`, `:494`)"; `## PyRS Changes`: "audit lines 404 and 494
(legacy log-name FIXMEs)".
- Referent: `pyrs/projectfile/file_object.py`.
- Verdict: **the work is already done, and the citation never matched.** Two
  defects, one inside the other. (a) The cited lines were wrong when written: at
  `492350bd^` the two legacy FIXMEs were at L423 (`# FIXME - This is a patch for
  'legacy' data`, falling back to the capitalized `"2Theta"` key) and L513
  (`# FIXME - Remove when Hidra-16_Log.h5 is fixed…`), never 404/494. (b) More
  importantly, **`492350bd` removed both** — and `492350bd` is this series'
  declared `claims_written_against` commit. The file now contains **zero**
  FIXMEs: the silent legacy-key fallback was replaced by an explicit
  `RuntimeError` telling the user to re-reduce
  (`file_object.py:428-445`), and the stray-mask removal is gone.
- Action: **this spec's only `## PyRS Changes` item is struck.** It is replaced
  by a one-line confirmation step: verify `grep -n FIXME
  pyrs/projectfile/file_object.py` returns nothing and that NXstress emits the
  canonical `HidraConstants.TWO_THETA` name, then close the item. No code change.
  The same correction applies to `README.md:188-190` and `README.md:444-445` —
  see the README's own Follow-up 1.

**F1.2** (A1) — `## Tests`: "Add tests in
`tests/unit/pyrs/utilities/NXstress/test_sample.py` for sx/sy/sz round-trip."
- Referent: this document's own `## Scope` and `## NXstress Changes`.
- Verdict: self-contradictory. Both locate sx/sy/sz in `_peaks.py`
  (`_peaks.py:235-239` is the commented-out block; `:240-242` the NaN fill —
  both confirmed). `_sample.py` does not write these fields.
- Action: the test belongs in
  `tests/unit/pyrs/utilities/NXstress/test_peaks.py`. **Corrected in place**
  in the Tests section above — a test-module path is a pointer, not a belief,
  so it follows the same in-place rule as a `path:LINE` citation.
  Tier: unit, unmarked — synthetic, no widget, no real file.

**F1.3** (A1/A2) — `## Verification`: "Write a `.nxs` file and open it in the
`nexusformat` validator".
- Referent: the installed `nexusformat` 1.0.8, probed by
  [`probes/a4_nexusformat_validator.py`](probes/a4_nexusformat_validator.py).
- Verdict: **under-specified, not wrong.** The NeXus-org NXstress validator is
  real, but it is a *separate repository* — the installed package has no
  `validate` module, no `nxvalidate` script and no `valid*` name in its public
  API (probe output pasted below). Spec 09's Verification and spec 10's Overview both
  carry that hedge; this one does not, so an implementer following it finds
  nothing and has no idea where to look. This spec's *other* two tools are fine:
  `h5dump` and the `nexusformat` Python API are both present in the environment.

  ```console
  $ pixi run python plans/NXstress-prod/probes/a4_nexusformat_validator.py
  installed nexusformat version: 1.0.8

    CLAIM   1.0.8 has no `validate` module
    RESULT  importlib.util.find_spec('nexusformat.validate') -> None

    CLAIM   no validator submodule under nexusformat
    RESULT  importable validator submodules: NONE

    CLAIM   1.0.8 has no `nxvalidate` script
    RESULT  shutil.which('nxvalidate') -> None

    CLAIM   no validator capability at all
    RESULT  names containing 'valid' in nexusformat.nexus: NONE

    CLAIM   spec 04 Verification says to use `h5dump` or the nexusformat Python API
    RESULT  shutil.which('h5dump') -> '…/.pixi/envs/default/bin/h5dump'
  ```
- Action (for the implementing PR; the body above is left as the record):
  replace that Verification bullet with — "Write a `.nxs` file and open it in
  the NeXus-org NXstress validator. **That validator is not part of the
  installed `nexusformat` package and must be obtained from its own
  repository** — see spec 10's Overview, which tracks adding it to this repo.
  Until then this step is blocked, exactly as in specs 09 and 10."

**F1.4** (A1/A2) — Two different referents are given for the same rotation-order
cross-check.
- Referent: `## NXstress Changes` says cross-check against "`DENEXDetectorGeometry`
  conventions in `pyrs/core/instrument_geometry.py`"; `## Delivered Feature` says
  verified "against the **reduction pipeline's** rotation convention";
  `README.md:132-133` says "the reduction code's convention".
- Verdict: these are **not interchangeable**, and this series says so elsewhere:
  `08-fit-spectrum-prereqs.md:150-158` records that the real reduction pipeline
  "applies the arm shift directly to the pixel matrix instead
  (`reduce_hb2b_pyrs.py`), bypassing this class entirely". Cross-checking against
  `DENEXDetectorGeometry` therefore does not establish agreement with the
  reduction pipeline, which is what the Delivered Feature promises a user.
- Action (for the implementing PR): `DENEXDetectorGeometry` is the correct
  referent for *this* spec's change — it is what `_instrument.py` writes from.
  Replace the `## Delivered Feature` bullet's "verified against the reduction
  pipeline's rotation convention" with "verified against the detector-geometry
  convention PyRS records (`DENEXDetectorGeometry`)". Whether
  `reduce_hb2b_pyrs.py` agrees with `DENEXDetectorGeometry` is a **separate,
  open question**, flagged here rather than silently assumed.

**Checked and accurate — no action.** `_peaks.py:235-239`, `_fit.py:546-549`,
`_definitions.py:229`, `_instrument.py:70`, `_instrument.py:165` all land exactly
on their claimed targets.

**Invariant flagged, not written** (an audit flags; the implementing PR writes):
a test that `pyrs/projectfile/file_object.py` contains no `FIXME` referencing a
legacy log name would have caught F1.1 the moment it was fixed upstream. Owner:
this PR. Tier: unit.

---

## Follow-up 2 — 2026-09-26 (closing the A5 gap left open by Follow-up 1)

Follow-up 1 covered this spec's citations (A3) but never executed the code they
point at, leaving A5 partial. Every claim in this spec is A5 — the counterparty
is `pyrs/utilities/NXstress/`, which is in this repo and has landed. Probed now:
[`probes/a5_nxstress_internals_today.py`](probes/a5_nxstress_internals_today.py).

**F2.1** (A5) — **All four premises confirmed**, so the work this spec schedules
is still needed:

```console
  CLAIM   `allowed_identifier` does NOT yet cover `$`, whitespace, or other
          disallowed characters (claim 1)
  RESULT  passed through unchanged: ['has space', 'dollar$sign', 'tab\there',
          'slash/bad', 'newline\nbad', 'ünïcode']
  RESULT  source: return s.replace(":", "_")

  CLAIM   default vs named masks are addressed differently in the writer (claim 2)
  RESULT  `_Fit.init_group` normalises by dropping `None` and injecting the
          default tag: ['mask_keys.discard(None)', 'mask_keys.add(DEFAULT_TAG)']

  CLAIM   instrument name is hardcoded to "HB2B" (claim 3)
  RESULT  ['inst = cls._init("HB2B", "HB2B")']
```

**F2.2** (A5) — **`/` is the character this spec's Scope does not mention, and
it is the dangerous one.** Scope says "cover at minimum `$`, whitespace, and any
other characters disallowed by the NXstress/HDF5 group-name rules". Probing what
each character actually *does* inverts that priority:

```console
  CLAIM   …and what each actually does to the file (claim 1, sharpened)
  RESULT  'dollar$sign': accepted, stored at '/dollar$sign'
          'has space': accepted, stored at '/has space'
          'slash/bad': accepted, stored at '/slash/bad' -- SILENTLY NESTED, not one group
          top-level groups afterwards: ['dollar$sign', 'has space', 'slash']
```

`$` and a space are **legal HDF5 link names** — it is the NeXus convention, not
HDF5, that disallows them, and writing one produces a schema-nonconformant but
structurally intact file. A `/` does something worse: it does **not raise**, it
silently creates a *nested* group, so one PV-log name becomes two groups and the
file's structure is wrong in a way nothing reports.
- Action (for the implementing PR): add `/` to this spec's Scope list explicitly,
  and order the work by consequence — `/` (silent structural corruption) before
  `$` and whitespace (schema conformance). Promote the round-trip assertion to a
  test in this PR; tier: unit.

**F2.3** (A5) — The rotation-order cross-check this spec requires, and which
Follow-up 1 F1.4 left open on the question of *which* convention is the referent,
now has a concrete subject. The chain `_instrument.py` writes, each
transformation depending on the previous:

```console
  ("translation_x", tx, ex, "m", "translation")
  ("translation_y", ty, ey, "m", "translation")
  ("translation_z", tz, ez, "m", "translation")
  ("distance", distance, ez, "m", "translation")
  ("rotation_x", rotx, ex, "deg", "rotation")
  ("rotation_y", roty, ey, "deg", "rotation")
  ("rotation_z", rotz, ez, "deg", "rotation")
  ("two_theta_zero", tth0, ex, "deg", "rotation")
```

Note that `translation_z` and `distance` are both written along `ez`, which is
the double-count spec 09 addresses — visible here as an ordering fact rather
than only as a sentence. **This does not resolve F1.4**: it records what PyRS
writes, so that the comparison against `DENEXDetectorGeometry` (and, separately,
against `reduce_hb2b_pyrs.py`) can be made against something concrete.

---

## Follow-up 3 — 2026-09-26 (identifier policy: the question Follow-up 2 F2.2 opened)

**F3.1** (A5) — **`allowed_identifier` is a many-to-one conversion with no
collision check, and the collision silently destroys a log.** Not claimed
anywhere in this series; found by probing
[`probes/a5_nxstress_internals_today.py`](probes/a5_nxstress_internals_today.py).

```console
  CLAIM   the converter is MANY-TO-ONE and nothing checks for collisions
  RESULT  distinct PV logs ['HB2B:CS:X', 'HB2B_CS_X'] both convert to 'HB2B_CS_X';
          groups actually written: ['HB2B_CS_X']; surviving local_name attribute:
          'HB2B_CS_X' -- the first log is gone, with no record that it existed.
```

`_sample.py:127-137` iterates **every** sample log, converts the key, and writes
into a flat `NXcollection`. A second key converging on the same identifier
overwrites the first — **including the `local_name` attribute that exists
specifically to preserve the original PV name**. So the one mechanism that could
have recorded the loss is itself overwritten.

This exists **today**, with only the `:` → `_` replacement. It is not introduced
by this spec; this spec's widening of the conversion makes it **more likely**.

**F3.2** — Why "just disallow them" needs one distinction before it is acted on.

The only production caller (`_sample.py:129`) converts **instrument PV-log
names**. That input is not ours: it comes from the control system, and the set
of names is open-ended. So "disallow" splits into two very different policies:

| Policy | Effect on a name like `HB2B CS:X` | Failure mode |
|---|---|---|
| **Reject the input** | the whole `.nxs` write raises | one unanticipated PV name loses an entire reduction's output |
| **Restrict the output alphabet** (allowlist + total conversion) | becomes `HB2B_CS_X` | never fails — but collides silently with a real `HB2B_CS_X`, per F3.1 |

Neither is right on its own, and the current code is the worst of both: a
*partial* conversion (only `:`) that is also many-to-one and unchecked.

- **Action (proposed decision, for the implementing PR — see Decisions item 23):**
  define an explicit allowlist `[A-Za-z0-9_.]`, convert every character outside
  it to `_`, and **raise on collision, never on the character**. That keeps a
  save robust against an odd PV name — which is not PyRS's to control — while
  making the one genuinely unrecoverable case loud instead of silent. Two
  exceptions get rejected outright rather than converted, because each indicates
  a bug rather than an awkward name: an **empty** identifier, and one containing
  **`/`** (a path separator, not a character — see F2.2; raw h5py silently nests
  on it).

**Invariants flagged, not written** (owner: this PR; tier: unit):
1. Every character outside the allowlist maps into it — asserted by iterating the
   allowlist rather than restating it, in the idiom of
   [`tests/unit/pyrs/utilities/NXstress/test_definitions.py`](../../tests/unit/pyrs/utilities/NXstress/test_definitions.py).
2. Two distinct log keys converging on one identifier **raise**, and no log is
   silently dropped — the regression test for F3.1.
3. `/` and the empty string are rejected, not converted.

Note that `tests/unit/pyrs/utilities/NXstress/test_definitions.py:100-105` currently
pins the *present* behaviour ("replaces `:` with `_` and leaves other chars
unchanged"). That test will need updating in the same PR; it is not a
counter-example to this finding but a record of the behaviour being replaced.

---

## Follow-up 4 — 2026-09-26 (identifier policy resolved; supersedes F3.2's proposal)

F3.2 proposed "allowlist `[A-Za-z0-9_.]`, convert outside it to `_`, raise on
collision". That proposal is **withdrawn**. It had two defects, and the second
is the one this audit exists to catch.

**F4.1** — The allowlist wording conflated two different sets: the **output
alphabet** (what may legally appear in an identifier) and the **pass-through
set** (what needs no escaping). An escape marker is necessarily in the first and
not the second. Stating one allowlist for both is incoherent.

**F4.2** (A4) — More seriously, F3.2 rested on `.` being legal in a NeXus
identifier, and **that is not verifiable in this repo**. The only support is an
unsourced comment in `_definitions.py:225`. Probed:

```console
  CLAIM   an authoritative NeXus identifier rule is available to verify against
  RESULT  FALSE -- files in nexusformat defining `validItemName`: NONE.
          The absence IS the finding; the schema doc is still not in the repo.
```

This is the *same* missing artifact spec 10 already tracks (`NXstress.xml`/
`.html`, plus the NeXus-org validator in its own repository). A design decision
must not rest on it.

**F4.3** — **Adopted rule: allow only what is legal in a Python identifier**
(`str.isidentifier()`). It needs no external specification, and it is strictly
*narrower* than any plausible NeXus rule, so it stays valid whatever the schema
doc eventually says.

The rule settles the escape marker by itself:

```console
  CLAIM   `$` is legal in a Python identifier
  RESULT  FALSE -- punctuation legal in a Python identifier: ['_']
          -- so `_` must be the escape marker.
```

`$` is **not** permitted, contrary to a reasonable expectation. `_` is the only
punctuation Python identifiers allow, so it is forced to be the marker, and an
incoming `_` must double.

**The encoding.** Every `_` in the output is a marker; a literal `_` from the
input always appears as `__`, so a lone `_` can only introduce an escape. The
character *after* the marker selects the form, unambiguously, because neither
`_` nor `u` is a hex digit and the hex is uppercase:

| After `_` | Form | Consumes | Meaning |
|---|---|---|---|
| `_` | `__` | 2 | a literal `_` |
| hex digit | `_XX` | 3 | byte `0xXX` — `_3A` is `:` |
| `u` | `_uXXXX` | 6 | codepoint beyond U+00FF |

Position matters: a digit is legal *inside* an identifier but not leading, and
`2theta` is a real log name in `tests/data`.

**Verified against real data** —
[`probes/a5_identifier_policy.py`](probes/a5_identifier_policy.py):

```console
  RESULT  185 distinct names sampled from tests/data; 116 already valid
          identifiers, 69 need encoding; non-ASCII: 0; leading-digit: ['2theta',
          '2thetaSetpoint']

    'HB2B:Mot:sz_real'       -> 'HB2B_3AMot_3Asz__real'   id=True round-trip=True
    '_DEFAULT_'              -> '__DEFAULT__'             id=True round-trip=True
    'Scan Index'             -> 'Scan_20Index'            id=True round-trip=True
    '2theta'                 -> '_32theta'                id=True round-trip=True
    'a$b'                    -> 'a_24b'                   id=True round-trip=True

  CLAIM   every real name encodes to a valid identifier AND round-trips
  RESULT  True over all 185 sampled names

  CLAIM   the adopted encoding is injective
  RESULT  YES over A/_/:/./space/9 up to length 4 (1554 inputs)
```

Inputs that *look* like escapes round-trip correctly, because the doubling
protects them: `a_3Ab` → `a__3Ab` → `a_3Ab`.

**F4.4** — **The cost, stated plainly.** `_` is the only punctuation a Python
identifier allows, so it must be the marker, so every incoming `_` doubles —
37 of the 185 real names (47 occurrences), including `average_value` →
`average__value`.

> **Superseded by Follow-up 5.** This paragraph went on to call that cost "not
> avoidable under this rule". That was wrong, and the claim was not probed
> before it was made — an always-escape introducer was assumed to be the only
> option. Making the *introducer* `__` rather than `_` avoids it entirely. The
> paragraph is left standing as the record of what was believed.

Because the encoding is **injective and reversible**, the collision check F3.1
required is no longer needed to prevent data loss — collisions cannot occur. A
decode round-trip assertion replaces it.

**Invariants flagged, not written** (owner: this PR; tier: unit) — superseding
F3.2's list:
1. `encode(name).isidentifier()` for every name, including leading-digit and
   non-ASCII cases.
2. `decode(encode(name)) == name` — the reversibility requirement, over both
   synthetic adversarial inputs (`a_3Ab`, `__`, `_3A`) and the real log names in
   `tests/data`.
3. `encode` is injective: no two distinct inputs share an output.
4. `tests/unit/pyrs/utilities/NXstress/test_definitions.py:100-105` pins the
   behaviour being replaced and must be rewritten in the same PR.

---

## Follow-up 5 — 2026-09-26 (the escape introducer; supersedes F4.4's cost claim)

**F5.1** — F4.4 stated that doubling every incoming `_` was "not avoidable under
this rule". **That was wrong, and it was asserted without being probed** — an
always-escape introducer was assumed to be the only option. It is not.

**Making the escape introducer `__` rather than `_`** means a lone underscore is
never the start of an escape, so it passes through untouched:

```console
  legibility -- the point of the `__` introducer:
    'my_log_value'           -> 'my_log_value'                 rt=True  <- unchanged
    'average_value'          -> 'average_value'                rt=True  <- unchanged
    '_DEFAULT_'              -> '_DEFAULT_'                    rt=True  <- unchanged
    'a_3Ab'                  -> 'a_3Ab'                        rt=True  <- unchanged
    'HB2B:Mot:sz_real'       -> 'HB2B__3AMot__3Asz_real'       rt=True
    'Scan Index'             -> 'Scan__20Index'                rt=True
    '2theta'                 -> '__32theta'                    rt=True

  CLAIM   how many REAL names need an escaped underscore
  RESULT  0 of 185 [] -- versus 37 under a single-underscore introducer, which is
          every name containing '_'.
```

It also removes the two-character hex lookahead an earlier draft needed: with a
`__` introducer, whether `DE` happens to be valid hex no longer matters, which is
why `_DEFAULT_` survives verbatim.

**F5.2** — **What escaping can and cannot buy.** A literal *double* underscore is
still rendered non-verbatim (`a__b` → `a__5F_b`). That is not a defect and not a
restriction on input: it round-trips exactly, as does `my__log__value` →
`my__5F_log__5F_value` → `my__log__value`. Nothing is unrepresentable.

The general fact worth writing down, because it bounds every alternative anyone
proposes later: **an injective map from an open input alphabet into the 63
characters a Python identifier permits cannot leave every input verbatim.**
Something must be escaped; the only design freedom is *which* inputs pay. The
`__` introducer spends that cost on a pattern that occurs **zero times** in the
185 real log names, where the `_` introducer spent it on 37 of them.

Measured over the real namespace: **116 of 185 names encode verbatim**, 69 need
encoding (almost all for `:`), and 0 need an escaped underscore.

**F5.3** — Final encoding, superseding F4.3's table:

| Form | Means |
|---|---|
| `__XX` | the character with byte value `0xXX` — `__3A` is `:` |
| `__uXXXX` | a codepoint above U+00FF |
| `__5F` | a literal `_` that would otherwise be ambiguous |
| a lone `_` | itself |

Encoding needs **one** character of lookahead — is the next emitted token also
going to begin with `_`? Decoding needs none.

Verified over all 185 real log names (encode → valid identifier, and exact
round-trip) and injective over 1364 synthetic inputs:
[`probes/a5_identifier_policy.py`](probes/a5_identifier_policy.py).

**F5.4** — A counting correction. F4.4 and earlier notes said "47 of 185 names
contain an underscore". That conflated occurrences with names: it is **37 names,
47 occurrences**. The figure came from a character-frequency scan that
incremented per occurrence. Corrected wherever repeated.

---

## Follow-up 6 — 2026-09-26 (bookkeeping: the identifier decision is now one row)

**F6.1** — The identifier policy was settled over three exchanges and was
recorded as three superseding Decisions Log rows (23 → 24 → 25). Those have been
**consolidated into a single row 23**, which carries the final decision plus the
three rejected alternatives and why each was rejected.

Follow-ups 3, 4 and 5 are append-only and still cite "item 23", "item 24" and
"item 25". **All three now refer to the same consolidated row 23**, which says so
explicitly. Nothing else in the series referenced 24 or 25.

The number 23 was kept rather than assigned afresh for exactly that reason:
Follow-up 3's pointer to "Decisions item 23" remains correct, and the log stays
contiguous at 1-23.

**What was deliberately *not* collapsed.** The reasoning trail lives in Follow-ups
3-5 and is untouched, because the rejected alternatives are the useful part —
particularly that `.` as an escape marker is unverifiable in this repo, and that
F4.4's "not avoidable" claim was asserted without probing and then falsified.
A consolidated decision row records *what was decided*; the Follow-ups record
*what was believed along the way*, which is what makes a later reader able to
tell a settled question from a lucky guess.
