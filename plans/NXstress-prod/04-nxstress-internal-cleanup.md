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

`_sample.py:191-201` iterates **every** sample log, converts the key, and writes
into a flat `NXcollection`. A second key converging on the same identifier
overwrites the first — **including the `local_name` attribute that exists
specifically to preserve the original PV name**. So the one mechanism that could
have recorded the loss is itself overwritten.

This exists **today**, with only the `:` → `_` replacement. It is not introduced
by this spec; this spec's widening of the conversion makes it **more likely**.

**F3.2** — Why "just disallow them" needs one distinction before it is acted on.

The only production caller (`_sample.py:193`) converts **instrument PV-log
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

---

## Follow-up 7 — 2026-09-29 (implementation pass; the schema turned out to exist)

Implementing this spec falsified the premise Follow-up 4 rests on, and turned one
of its TODOs into a defect larger than the TODO described. Per `process.md` §5.6
this is budgeted cost, not audit failure. Follow-ups 1-6 are untouched.

**F7.1** (A4) — **The NXstress schema is available, and F4.2's "unverifiable" no
longer holds.** F4.2 probed the installed `nexusformat` package, found no
`validItemName`, and concluded a design decision must not rest on `.` being legal.
The rule is published — not in the package, and not in the application definition
either, but in `nxdl.xsd` of the NeXus *definitions repository*:

```console
  CLAIM   `nxdl.xsd` defines a `validItemName` simpleType with a pattern and a length cap
  RESULT  pattern='[a-zA-Z0-9_]([a-zA-Z0-9_.]*[a-zA-Z0-9_])?' maxLength=63

  CLAIM   `.` is legal in the interior of an identifier (F4.2 called this unverifiable)
  RESULT  True -- so the unsourced comment in `_definitions.py` was correct.
          Leading/trailing `.`: False/False

  CLAIM   a LEADING DIGIT is legal (Decision 23's Python rule escapes it)
  RESULT  True -- `2theta` and `2thetaSetpoint` are real log names, and need no
          encoding at all under this rule.
```

The probe reading it live is
[`probes/a4_validitemname_rule.py`](probes/a4_validitemname_rule.py).

Two corrections follow. First, the round-one search was in the wrong place, not
absent: the failure mode was concluding "unavailable" from one package's contents.
Second, **`_definitions.py`'s unsourced comment asserting `.` is interior-legal was
right all along** — it was disbelieved because it was unsourced, which is the
correct instinct but produced the wrong answer here.

**F7.2** (A4) — **The 63-character cap is a constraint nothing in this series had
accounted for**, and encoding *lengthens* names:

```console
  CLAIM   worst-case encoded length against the cap
  RESULT  96 chars if EVERY character of the longest name escaped, vs cap 63 --
          CAN EXCEED, so the encoder must check
  CLAIM   any real name exceeds the 63-character cap once encoded
  RESULT  NONE -- longest encoded form: 36 chars
```

Reachable in principle, not reached by current data. `allowed_identifier` raises
on it rather than emitting a name NeXus will reject.

**F7.3** — **Adopted rule, superseding Decision 23** (recorded as Decisions row 25).
Same `__` introducer and the same escape forms; the alphabet widens from
`str.isidentifier()` to `validItemName`. Measured against the same 185 real names:

```console
  RESULT  185 distinct names sampled from tests/data; 118 already valid identifiers,
          67 need encoding; non-ASCII: 0; leading-digit: ['2theta', '2thetaSetpoint']
    '2theta'                 -> '2theta'                       rt=True  <- unchanged
    'a.b'                    -> 'a.b'                          rt=True  <- unchanged
    'HB2B:Mot:IS:Y:Center.RBV' -> 'HB2B__3AMot__3AIS__3AY__3ACenter.RBV' rt=True
  CLAIM   every real name encodes to a valid identifier AND round-trips
  RESULT  True over all 185 sampled names
  CLAIM   the adopted encoding is injective
  RESULT  YES over 3905 inputs
```

118 verbatim rather than 116; `2theta` and dotted names stop being mangled.
Decision 23's stated virtue — "strictly narrower than any plausible NeXus rule, so
it stays valid whatever the schema says" — held: its output was always conformant,
merely not minimal.

**F7.4** — **Deviation from the approved plan, recorded because it was deliberate.**
The plan said `allowed_identifier` should raise on `/`, carrying forward F3.2's
reasoning that a `/` indicates a bug and h5py silently nests on it. Implementation
showed that rationale does not survive the encoding: F2.2's nesting result is about
an **unencoded** `/`, and under a total encoding `a/b` becomes `a__2Fb`, which
round-trips and creates exactly one group. Raising would discard an entire
reduction's output over one unanticipated PV name — the failure mode F3.2 itself
rejected when it chose "restrict the output alphabet" over "reject the input".
`/` is therefore encoded like any other illegal character. Empty input and the
63-character overflow still raise: neither is representable.

**F7.5** (A5) — **`_instrument.py`'s transformation chain was not traversable, and
the defect is larger than the TODO it sits next to.** The TODO at the rotation
block reads `check order of rotations here!!!`. The order is *correct*. The chain
is not connected.

`NXtransformations`' own documentation states the entry point "will be outside of
this class and point to a field in here", and that for `T₁` depending on `T₂`
depending on `T₃`, `T_f = T₃T₂T₁`. `_instrument.py` set the detector's
`depends_on` to `translation_x` — the **first** link, whose own `depends_on` is
`"."` — so traversal ended immediately:

```console
  CLAIM   how many transformations `_instrument.py` writes
  RESULT  8: ['translation_x', 'translation_y', 'translation_z', 'distance',
              'rotation_x', 'rotation_y', 'rotation_z', 'two_theta_zero']

  CLAIM   how many are REACHABLE by following depends_on from the detector
  RESULT  1 of 8 -- entry point './transformations/translation_x',
          traversal ['translation_x']

  CLAIM   the composed transform, as the file currently stands
  RESULT  translation=[0.01, 0.0, 0.0], rotation_is_identity=True
```

Every rotation and the two-theta zero were written to the file and excluded from
the geometry. Pointing the entry point at the last link fixes it:

```console
  RESULT  8 of 8 -- entry point './transformations/two_theta_zero', traversal
          ['two_theta_zero', 'rotation_z', 'rotation_y', 'rotation_x', 'distance',
           'translation_z', 'translation_y', 'translation_x']
  RESULT  translation=[0.01, 0.02, 2.03], rotation_is_identity=False
```

Probe: [`probes/a5_transformations_chain.py`](probes/a5_transformations_chain.py).

**Why no one noticed, and why the spec's own test would not have caught it.**
`instrumentFromNexus` reads each transformation *by name* and never follows the
chain, so a PyRS round trip is insensitive to it. This spec's `## Tests` section
asks for exactly that round trip — "writes a geometry, reads it back, and asserts
the rotation components are numerically equal". It would have passed throughout.
The same shape as `a4_basename_extensions.py`'s finding against spec 07, where
07's own Verification case was one the two branches agree on. The replacement
test walks the chain and asserts every written transformation is reached.

**F7.6** — **F1.4 now has a concrete referent, and Chris's Q2 answer has a bad
citation.** F1.4 left open whether the cross-check referent is
`DENEXDetectorGeometry` or "the reduction pipeline". `open-questions/04` Q2's
answer names `file_object.py:generate_rotation_matrix` — **that symbol is not in
`file_object.py`**; it is `reduce_hb2b_pyrs.py:257`, and the `@`-fix described has
landed there with regression tests in `tests/unit/pyrs/core/test_reduce_hb2b_pyrs.py`.
So the reduction pipeline's convention *is* readable, and the probe compares
against it directly:

```console
  CLAIM   does the NeXus rotation sub-chain match reduce_hb2b_pyrs's Rx @ Ry @ Rz
  RESULT  True
  CLAIM   and with the rotation sub-chain traversed the other way
  RESULT  False
```

The written order already agrees with the reduction pipeline. F1.4's question is
answered in the affirmative for the rotations; the `translation_z`/`distance`
double-count along `ez` that F2.3 noted remains spec 09's.

**F7.7** (A5) — **The instrument name cannot come from the workspace, because PyRS
does not record one.** This spec says to draw it "from the `HidraWorkspace` (e.g.,
from a sample-log entry or a new workspace attribute)". Neither exists: the Hidra
project format's `instrument` group holds only `calibration`, `efficiency
calibration`, `geometry setup` and `monochromator setting`, and no sample log
carries a beamline name. A new workspace attribute is a PyRS data-model change,
which this spec's Scope puts **out of scope**.

Resolved through configuration instead — `nxstress.instrument_name` and
`nxstress.instrument_short_name` in `pyrs/resources/application.yml`, read by
`_Instrument._instrument_names()` with a logged fallback to `"HB2B"`. This meets
the stated goal ("usable at other beamlines without a code change") using the
config mechanism spec 01 already shipped, and touches no data model.

**F7.8** — **`peaks/sx,sy,sz` carry `vx,vy,vz`, and the schema settles it.** The
fields are required (`minOccurs="1"`, dimension `n_Peaks`) and documented as *"the
sample position in the sample reference frame"*. In PyRS that quantity is
`PointList.(vx, vy, vz)`; the identically-spelled `sx`/`sy`/`sz` logs are stage
positions the stress/strain workflow does not use (`open-questions/04` Q3).
`docs/developer/source/design/nexus/IO_prototype.rst` had already recorded the
doubt — *"included from the logs, but mostly just because the logs had the same
variable names -- **this is probably incorrect**!"* — and it was right.

The commented-out block's `# This doesn't make sense!` is now explicable: it read
`logs['sx']`, which spans **every** workspace scan point, into a slice needing only
the scan points of one `PeakCollection`. `SampleLogs.get_pointlist(subruns)` takes
the subset directly. Each field records its real source in `local_name`, because
the field name alone is misleading (Decisions row 24).

**F7.9** — A tier correction. `review/findings.md` §5 assigns invariants 7-9 to the
unit tier, and the real-namespace sweep they describe reads `tests/data`, which
`CLAUDE.md`'s tier rules make integration. Resolved by splitting rather than
re-tiering: the unit tests exhaust the rule over generated and adversarial inputs
(every byte value at every position), and the real-namespace measurement stays in
the probe, where it already was.

**Invariants written by this PR** (the flagged set, plus two the above added):

| # | Invariant | Where |
|---|---|---|
| 6 | no legacy log-name `FIXME` in `file_object.py`; no capitalized `"2Theta"` key in code | `tests/unit/pyrs/projectfile/test_file_object.py` |
| 7 | every character at every position encodes into the NeXus alphabet | `test_definitions.py` |
| 7b | the 63-character cap is enforced at its edge, and one past it raises | `test_definitions.py` — split out from 7 by the PR review: the length half of 7 was an assertion that could not fail (longest encoding of those inputs is 14, against a cap of 63) |
| 8 | `decode(encode(name)) == name` over escape-shaped and real-shaped inputs | `test_definitions.py` |
| 9 | `allowed_identifier` is injective | `test_definitions.py` |
| 10 | every written transformation is reachable from the detector's `depends_on` | `test_instrument.py` |
| 11 | default and named masks round-trip; `DEFAULT_TAG` never leaks into the workspace | `test_workspace_read.py` |

Invariant 6's second test walks the AST rather than scanning text: the
`RuntimeError` that *replaced* the legacy fallback names `"2Theta"` in its own
docstring, and a text scan cannot distinguish documenting a fix from repeating a
defect. The first version of that test failed for exactly that reason.

**Still open, recorded rather than assumed.** The schema expresses `peaks/sx,sy,sz`
in the frame defined by `SAMPLE_DESCRIPTION`'s `NXtransformations` group, which
PyRS does not write. The values are correct; the frame they are expressed in is
undefined for an external consumer. Raised as `open-questions/04` Q5 — settling it
needs goniometer conventions this spec has no referent for.

**F7.10** — **A `## Verification` step is not runnable against this repository, and
that is a finding rather than a step to skip.** Step 2 reads: "Write a `.nxs` file
from a real HB2B dataset; inspect sx/sy/sz fields with `h5dump` or the
`nexusformat` Python API — confirm they are non-NaN."

**No file in `tests/data` can satisfy it.** Writing an NXstress file requires
instrument geometry and a parseable peak tag; producing non-NaN sample positions
additionally requires finite `vx`/`vy`/`vz`. Surveying every `tests/data/*.h5`:

```console
  with geometry + parseable peak tag:   HB2B_938_peak.h5  -- vx/vy/vz NON-FINITE
  with finite vx/vy/vz:                 3393/3394/3395_PWHT-*.h5, HB2B_1320,
                                        _1327, _1328, _1331, _1332
                                        -- ALL lack instrument geometry, and all
                                           use the unparseable tag 'peak0'
  with BOTH:                            NONE
```

The two halves are disjoint. This is the same class as spec 09's `STRESS_FIELD`
block (`probes/README.md`, "Not probed, and why"): externally blocked by the
absence of a fixture, not by anything in the design.

**What was verified instead**, separately, so the claim is not left unsupported:

*(a) The non-NaN path, against real logs* — `_Peaks._sample_positions` on
`HB2B_1327.h5`'s real sample logs:

```console
  tests/data/HB2B_1327.h5: 150 workspace subruns; PeakCollection covers 150
    sx[:5] = [-30. -30. -30. -30. -30.]
    sy[:5] = [3. 6. 3. 6. 3.]
    sz[:5] = [-10. -10.   0.   0.  10.]
    all finite: True      any NaN: False
    matches vx: True
    equals the sx LOG (should be False -- different quantity): False
    sx log[:5] = [-8.000972 -8.000972 -8.000972 -8.000972 -8.000972]
```

> **Superseded by F7.11.** This paragraph went on to say the `sx` **log** is
> *constant* at -8.000972, making it obviously a stage position. **That was wrong,
> and wrong in a specific way worth recording: it generalised from the first five
> of 150 values**, which happen to be equal because the scan's outer loop is over
> `x`. Over the full array `sx` varies by 60 mm. The conclusion it supported —
> that the two are different quantities — survives, but the evidence given for it
> did not. Left standing as the record of what was believed.

Writing the log, as this spec's body instructs, would have recorded the wrong
quantity, and no test that only checked "not NaN" would have noticed.

*(b) The written file, end to end* — `HB2B_938_peak.h5` → `.nxs`, read back with
`h5py`:

```console
  peaks/sx: local_name='vx' units=mm values=[nan]      <- fallback path, logged
  entry point: ./transformations/two_theta_zero
  reached  : ['two_theta_zero', 'rotation_z', 'rotation_y', 'rotation_x',
              'distance', 'translation_z', 'translation_y', 'translation_x']
  ALL REACHABLE: True
  'HB2B__3ACS__3AStrainDirection'  <- local_name 'HB2B:CS:StrainDirection'
  (58 of 120 names needed escaping)
```

That file exercises the geometry and identifier work on real data, and shows the
NaN fallback behaving as designed (with its warning) on a dataset whose
coordinates are genuinely non-finite.

- Action (for a future PR, not this one): either add a `tests/data` fixture
  carrying geometry *and* finite sample coordinates *and* a parseable peak tag, or
  restate this Verification step as the two halves above. Recorded rather than
  quietly marked done. Note this also bears on spec 04b/05, whose round trips will
  want the same fixture.

**F7.11** — **The fixture now exists, and the relationship between the two log
families is sharper than F7.10 claimed.** `tests/data/HB2B_1327_with_instrument.h5`
is built by `tests/scripts/make_nxstress_sample_position_fixture.py` from
`HB2B_1327.h5`, whose sample coordinates are real and finite. Two peripheral
things were supplied, neither of them the quantity under test: the
`instrument/geometry setup/detector` group, which **exists in the source but is
empty** (its three datasets are simply absent — which is why
`get_instrument_setup()` returned `None`), filled from the real values in
`HB2B_938_peak.h5`; and parseable peak tags, `peak0`/`peak1` → `peak000`/`peak111`,
since `_parse_peak_tag` needs a digit run of length divisible by three. Nothing
physical is lost by that rename: the source records `d reference` as 1.0 for one
peak and `NaN` for the other, so those tags never identified a reflection. The
8.4 MB `mask` group is dropped, keeping the fixture at 2.72 MB, under
pre-commit's 8 MB ceiling; NXstress generates a default mask when a workspace
carries none.

Measured across all 150 subruns, the stage and sample logs are **not** unrelated —
they describe the same motion in different frames:

```console
  sx: min= -8.0010 max= 51.9994 ptp= 60.0004 uniq= 18
  vx: min=-30.0000 max= 30.0000 ptp= 60.0000 uniq= 18
  sy: min=  3.5596 max= 11.5293 ptp=  7.9697 uniq= 58
  vy: min=  3.0000 max=  6.0000 ptp=  3.0000 uniq=  3
  sz: min= 53.0245 max= 73.0079 ptp= 19.9834 uniq=  3
  vz: min=-10.0000 max= 10.0000 ptp= 20.0000 uniq=  3

  sx vs vx: allclose=False maxdiff=22.0006 corr= 1.0000
  sy vs vy: allclose=False maxdiff= 5.5293 corr= 0.6378
  sz vs vz: allclose=False maxdiff=83.0079 corr=-1.0000
```

`sx` tracks `vx` with correlation **+1** and a 22 mm offset; `sz` tracks `vz` with
correlation **-1** — the stage z axis runs *opposite* to the sample z axis; `sy` is
an encoder readback (58 distinct values) against a commanded 3-point grid. So
writing the stage logs into `peaks/sx,sy,sz` would have stored an offset,
partly sign-flipped coordinate in a field the schema defines as the sample
position — a defect no "is it NaN" check could detect, and one that would have
silently inverted a strain gradient along z for any external consumer.

**This also sharpens `open-questions/04` Q5 from a tidiness concern to a concrete
one.** The offset-and-inversion between the two frames is exactly what
`SAMPLE_DESCRIPTION`'s `NXtransformations` is for, and PyRS writes no such group —
so the frame our (correct) values are expressed in is stated nowhere in the file.
The anti-correlation is pinned by `test_stage_and_sample_axes_differ_in_frame`, so
if the frames ever coincide the question is re-raised rather than lapsing quietly.

Verification step 2 is therefore **runnable**, as
`tests/integration/test_nxstress_sample_position.py` (7 tests, tier `integration`):
non-NaN, equal to `vx`/`vy`/`vz` under the writer's own flattening, *not* equal to
the stage logs, `local_name` provenance, full chain reachability on a real written
file, and `decode(encode(name)) == name` over the fixture's real PV-log names.

---

## Follow-up 8 — 2026-10-08 (PR review: detector masks and diffractogram keys are two namespaces, not one)

Raised in the human PR review of 04 + 04b + 04c (batch 5,
[plans/PR_review/04-comments.md](../PR_review/04-comments.md)), brought forward
into batch 4 because the D8 all-NaN-mask discussion ran into the same code.

Four defects (D10–D13) with **one root cause**, which this document does not
name anywhere: `masks/names` was being asked to hold two different kinds of
string.

| | what it is | has an array | produced by |
|---|---|---|---|
| **detector mask name** | `_DEFAULT_` and the entries of `HidraWorkspace._mask_dict` | yes | `set_detector_mask` |
| **diffractogram key** | a key of `_diff_data_set` | **no** | `reduction_manager.py:613-615`, as `"{mask_id}_eta_{eta_cent}"` with either part dropped when absent |

So the key space is `{None, "mask_a", "eta_-5.0", "mask_a_eta_-5.0"}`, and only
the second of those four is also a mask name. `_Masks.mask_keys` returned the
union of both sets.

### F8.1 — `nxstress_mask_names` conflated them; there are now two functions

`_definitions.py` gains `nxstress_diffractogram_keys(ws)` alongside
`nxstress_mask_names(ws)`. They differ in two ways, both load-bearing:

- `nxstress_mask_names` reads `_mask_dict` only; `nxstress_diffractogram_keys`
  reads `_diff_data_set` only.
- `nxstress_mask_names` injects `DEFAULT_TAG`, because the default detector mask
  is **always** present. `nxstress_diffractogram_keys` does not, because a
  reduction keyed `eta_0.0` is a reduction *that used the default mask* — the
  bare `DEFAULT_TAG` key means "the default mask with no eta ROI", which such a
  reduction never produced.

The second point is the correction the reviewer supplied twice, and it is the
whole of the fix: there is no reduction "with no mask", so `None` is not an
absence to be filled in — it is the name of a specific reduction.

Also new: `diffractogram_detector_mask(key, known_masks)`, which resolves a key
to the detector mask it references by matching against the names actually
present (longest first, so `mask_a` wins over a hypothetical `mask`), rather
than by splitting on `_eta_`. Parsing would have been wrong in both directions —
a mask legitimately named `rotated_eta_bar` parses as an eta ROI, and `__`-escaped
identifiers (Follow-up 5) do not survive a naive split at all.

### F8.2 — `masks/names` listed names with no array (D11); the `NXlink` is gone

Measured on a texture-shaped workspace before the fix:

```console
  names          : ['_DEFAULT_', 'eta_0.0', 'eta_-5.0', 'eta_5.0']
  detector/      : ['_DEFAULT_']
      eta_0.0    -> NOWHERE
```

[_instrument.py](../../pyrs/utilities/NXstress/_instrument.py)'s `_Masks` had an
`NXlink`-to-default branch intended to cover this. It never serialised. **Making
it work would have been a worse defect than leaving it broken**: `masksFromNexus`
would then hand back the default array under each composite key, and
`set_masks_from_dict` would install `_mask_dict` entries the source workspace
never had — D8 (all-NaN masks appearing on read) in a different property.
Branch deleted; `mask_keys` returns the mask namespace only.

### F8.3 — a texture workspace could not be written at all (D13)

```console
  eta-only reductions      -> ValueError: Reduced data required for mask '_DEFAULT_'
  eta-keyed PeakCollection -> ValueError: Mask 'eta_0.0' required by `PeakCollection`
```

Both from F8.1's conflation: the first from forcing `DEFAULT_TAG` into the key
space, the second from `validateWorkspaceAndPeaksData` checking
`PeakCollection.mask` — a diffractogram key — against the *mask* namespace. That
function's **second** check, against `_diff_data_set`, was correct all along, so
the fix was to delete the first rather than add a third.

This means §Tests' round-trip coverage never exercised a texture workspace. It
does now.

### F8.4 — the solid-angle heuristic misclassified in both directions (D10)

`_Masks` guessed between detector and solid-angle masks from dtype and length
parity:

```console
  detector mask, float, 16 px -> solid_angle=True    MISCLASSIFIED
  solid-angle, 3 floats (odd) -> solid_angle=False   MISCLASSIFIED
```

Nothing in PyRS ever loads a solid-angle mask into a `HidraWorkspace`
(`HidraProjectFile.read_mask_solid_angle` has exactly one caller, a test), so the
branch has never run on real data. Per the reviewer's direction it now
`raise NotImplementedError` rather than classify. The `solid_angle` group stays
as the schema placeholder and `masksFromNexus` still reads it, so implementing
the writer later needs no reader change. The dead `detector_mask=False` branch of
`_generate_default_mask` — which returned a plausible-looking `[-180.0, 180.0]` —
became a raise for the same reason.

### F8.5 — a workspace with no masks wrote a file PyRS could not read (D12)

```console
  masks SET (default exists):        on disk (16,)  -> read back OK
  masks ABSENT (default generated):  on disk (4, 4) -> READ FAILED
                                     RuntimeError: Mask array with shape (4, 4) is not acceptable
```

`_generate_default_mask` returned `np.ones(detector_size)`, which is 2-D, while
`set_detector_mask` refuses a 2-D array whose second axis is not 1. All 42
existing round-trip tests pass `with_masks=True`, so the generated path was never
round-tripped. Now returns 1-D `np.ones(nrows * ncols, dtype=np.int64)`, with the
missing `with_masks=False` round trip added.

### F8.6 — invariants added

Per `CLAUDE.md`'s "invariants belong in tests, not prose", the namespace split is
pinned rather than described:

- `test_definitions.py::TestDiffractogramKeysAreNotMaskNames` — over a texture
  workspace, `nxstress_diffractogram_keys` and `nxstress_mask_names` are
  disjoint except where a key genuinely *is* a mask name, and `DEFAULT_TAG` is in
  the second and not the first.
- `test_instrument.py::TestMaskGroupHoldsOnlyRealMasks` — every name in
  `masks/names` resolves to an array under `detector/` or `solid_angle/`. This is
  the one that would have caught D11, and it iterates the written group rather
  than restating a list, so a future mask kind is covered automatically.

### F8.7 — an unrelated repair, and an open defect

`probes/a5_transformations_chain.py` has been broken since 04b generalised
`_Instrument.init_group` to take `list[HidraWorkspace]`; nothing runs the probes,
so it went unnoticed. Repointed — it again confirms README Decisions row 26 (8 of
8 transformations reachable). This is distinct from `a5_nxstress_roundtrip`, whose
failure is documented and deliberate.

**Open:** `_Masks.init_group(ws, masks=...)` reads `names = masks["names"].nxvalue`,
which returns a bare `str` rather than a 1-element list when the group holds
exactly one mask, so the following `names.append(...)` would raise
`AttributeError`. Unreachable today (`_Instrument.init_group` never passes
`masks=`) and left standing rather than fixed blind, since mask-append semantics
are not specified anywhere. Recorded here so 04c's append work does not inherit it
silently.

---

## Follow-up 9 — 2026-10-14 (PR review: the sample frame is now defined, and Q5's premise was wrong)

Raised in the human PR review of 04 + 04b + 04c (batch 24,
[plans/PR_review/04-comments.md](../PR_review/04-comments.md)).

This document's `## Verification` and Follow-up 7 F7.5 both treat the *detector*
`NXtransformations` chain as the only one at issue. The sample side has no chain
at all, which [open-questions/04 Q5](open-questions/04-nxstress-internal-cleanup.md)
records — together with the reason it was left that way: *"the sample side has no
equivalent [referent] that has been located."*

**That reason was false, and the stakeholder said so.** Asked to accept leaving
`depends_on` unwritten, the reviewer's response was: *"I'm not understanding how
we can be presented with a 3-angle transformation, in an applicable form, that
can't simply be mathematically decomposed? That doesn't make any sense!"*

### F9.1 — the referent existed; it just had no matrix in it

`PoleFigureCalculator.rotate_project_q` (`pyrs/core/polefigurecalculator.py:186`)
applies `omega`, `chi` and `phi`. It was passed over during the implementation
pass because it implements the closed-form pole-figure projection of Bob He,
*Two-Dimensional X-ray Diffraction*, ch. 8.3 — built from sines and cosines, with
no matrix anywhere in it and therefore no axis to read off.

A composition of rotations is still a composition of rotations however it is
spelled. Driving that function through its **public signature** — aiming the
probe vector its `theta`/`eta` arguments fix along each basis vector in turn —
reads off the matrix it implements without touching its internals. Of 2880
candidate compositions over the six signed axes, exactly one survives.

Probe: [`probes/a5_sample_rotation_axes.py`](probes/a5_sample_rotation_axes.py).
Full output is pasted in Q5; the result is

```
sample -> lab  ==  M0 @ R_omega(0,-1,0) @ R_chi(1,0,0) @ R_phi(0,0,-1)
worst elementwise error over 200 random triples: 5.818e-14
```

### F9.2 — the finding inside the finding: a first decomposition said "ambiguous", wrongly

An earlier attempt enumerated stage rotations only and reported two survivors
differing in a handedness sign, which was written up as an irreducible ambiguity
for Q5 to settle. It is not. At `omega = chi = phi = 0` the projection does not
reduce to the identity but to `diag(1, -1, -1)` — a constant 180-degree rotation
about `+x`, written as the chain's `frame_offset` link. **A search over stage
rotations alone cannot match a composition carrying a constant factor**, so it
returns partial matches that disagree, and the disagreement looks exactly like a
real ambiguity. With the offset in the search the answer is exact and unique.

Recorded because the failure mode generalises: a decomposition that comes back
*underdetermined* is as likely to mean the candidate space is wrong as it is to
mean the referent is ambiguous, and the two are indistinguishable from the result
alone.

### F9.3 — what `_sample.py` now writes

`SAMPLE_DESCRIPTION/transformations`, as a per-scan-point chain, with the group's
`depends_on` naming the **last** link — the F7.5 defect asserted against on the
sample side before it could occur.

| # | link | type | vector | source |
|---|---|---|---|---|
| 1–3 | `translation_x/y/z` | translation | `(1,0,0)`, `(0,1,0)`, `(0,0,1)` | `vx`, `vy`, `vz` |
| 4 | `frame_offset` | rotation | `(1, 0, 0)` | constant `180` |
| 5 | `omega` | rotation | `(0, -1, 0)` | log `omega` |
| 6 | `chi` | rotation | `(1, 0, 0)` | log `chi` |
| 7 | `phi` | rotation | `(0, 0, -1)` | log `phi` — the entry point |

The rotation links are written all together or not at all, gated on every input
carrying all three angle logs; the synthetic unit fixtures carry none and get a
translations-only chain, while every real HB2B file in `tests/data` gets all
seven. The raw angle logs stay untouched under `logs/`, so this group is purely
additive. `frame_offset` is gated with the angles although it is constant:
written alone it would assert the stage was at zero.

### F9.4 — verification

Writing `tests/data/HB2B_1327_with_instrument.h5` (150 scan points) through
`NXstress.write` and reading it back with h5py: all seven links reachable by
following `depends_on`, all 150 rows long, and the chain composed from the
**on-disk** vectors and values reproduces `rotate_project_q` to a maximum
elementwise error of `1.11e-16`.

`test_sample.py::TestSampleTransformationChain` pins reachability, that the entry
point is the last link and not the first, traversal order against write order,
the axes, the all-or-nothing rotation gate, and lockstep growth on append.
`test_the_rotations_compose_in_write_order` composes the written chain and
compares it against `rotate_project_q` itself — so an edit to an axis *or to the
order* fails against the referent rather than against a restatement of it. Three
mutants confirm it can fail: swapping `omega` and `chi` in write order (axes
unchanged, rotation different), dropping `frame_offset`, and setting its value to
zero are each caught.

### F9.5 — what stays open

Nothing about the convention. Two things are worth a stakeholder's eye, both
recorded at the end of Q5: whether `frame_offset` is physical or an artefact of
the ch. 8.3 output convention (the chain is correct either way; only the name
would change), and whether `mrot` — present in every real HB2B file, absent from
`rotate_project_q` — moves the sample, in which case the chain is incomplete.
