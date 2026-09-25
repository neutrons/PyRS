# Open Questions — 03 NXstress I/O for CombineRuns Viewer

**Spec:** [03-combine-runs-nxstress.md](../03-combine-runs-nxstress.md)
**Blocking:** No

## Q1 — RESOLVED: read-back completeness for the CombineRuns case

Same plan-level caveat as spec 02 (README.md:353-360): `NXstress.read()`
doesn't reconstruct raw counts unless `input_data` was written. This spec is
explicitly export-only (loading a `.nxs` back into CombineRunsViewer is
out of scope), but the round-trip *test* in this spec does read the file
back via `NXstress.read()` to assert equality.

**Response from Chris:** What prevents us from testing that `NXstress.read()`
will reconstruct raw counts when `input_data` are present?

**Resolved:** nothing prevents that as a general NXstress capability — it's
a legitimate test, just not one this spec's own round-trip test should own.
`CombineRunsModel.combine_project_files` calls
`self._hidra_ws.load_hidra_project(_project, load_raw_counts=False, ...)`
(`combine_runs_model.py:21`) — raw counts are never loaded by this viewer's
own merge step, so the workspace this spec exports never has `input_data`
to write in the first place. Testing "does `NXstress.read()` reconstruct
raw counts when `input_data` is present" would require a workspace this
spec doesn't actually produce; that test belongs at the general NXstress
level (already exercised by other specs' fixtures, not CombineRuns-specific
ones). This spec's round-trip test legitimately narrows to sub-run counts
and sample-log arrays — not an arbitrary scoping choice, but a direct
consequence of `load_raw_counts=False`.
