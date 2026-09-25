# Open Questions — 10 Flip Defaults: NXstress Becomes Primary

**Spec:** [10-flip-defaults.md](../10-flip-defaults.md)
**Blocking:** Partially — Q1 gates *entering* this spec at all and remains
open. Q2 and Q3 are resolved (see below).

---

## Q1 — Are the trigger criteria for entering this spec actually met, and who signs off?

The spec lists three trigger criteria itself but flags them as provisional:
> **Trigger criteria** for entering this spec (to be confirmed at the end of
> spec 02 / start of spec 03):
> - All NXstress round-trip tests pass.
> - At least one real-data smoke test has been run by a scientist and signed off.
> - No open correctness bugs against the NXstress writer or reader.

None of the three currently has a concrete owner, a defined "scientist"
signing off, or a tracked bug list to check against "no open correctness
bugs." This mirrors the plan-level README's own note that Phase 6
acceptance criteria are "to be defined at the end of Phase 1"
(README.md:288, :667) — as of this writing that definition still hasn't
happened.

**Why it matters:** without a concrete checklist and named sign-off owner,
"entering Phase 6" is a judgment call that could happen prematurely (before
NXstress is genuinely trustworthy as the *only* save format offered) or be
indefinitely delayed (no one is unambiguously responsible for declaring the
criteria met).

**Next step:** convert the three bullet points into a literal checklist with
an owner and a link to wherever "open correctness bugs" would be tracked
(e.g., a GitHub issue label); do this at the latest by the end of Phase 1,
per the plan's own stated schedule.

**Chris:** Instrument scientist will confirm that the new format provides the same functionality as the previous data. One route is to save data both data in parallel, HIDRA Project files and NXStress files.

**Note (config-schema design, not a resolution of this question):** the
"save both in parallel" route is now directly available at any point in
the rollout, not just as a Phase-6 transition device — setting both
`nxstress.enable: true` and `legacy_io.enable: true` makes every wired
viewer's two save actions independently available side by side (see specs
02/03/05/07), and `reduce_hidra_workflow` writes both files automatically
on every reduction in that state. The trigger-criteria/sign-off question
above is unaffected and remains open.


---

## Q2 — RESOLVED: Reject, or silently route, a manually-typed `.h5` save path?

The spec's own Verification section originally posed this as an open
either/or rather than a decided behavior:
> Smoke test: attempt to save as `.h5` (by typing the extension manually in
> the save dialog); confirm the viewer either rejects it with a clear
> message or routes it through the NXstress writer regardless.

**Why it matters:** these produce very different user experiences — silently
routing a `foo.h5` filename through the NXstress writer would produce a file
with a `.h5` extension but NXstress-format contents (misleading and
potentially breaking any external tooling that dispatches on extension);
rejecting outright is safer but needs an explicit error message and matching
GUI validation logic that isn't described anywhere else in the plan.

**Chris:** NXstress should automatically apply the correct extenssion and silently reoute to the format if needed.

**Resolved, per Chris's answer above** — but the mechanism that ended up
implementing it is stronger than "reject vs. route": each save action
(`Save`, `Save as NXstress…`) now **imposes** its own configured extension
(`legacy_io.extension`/`nxstress.extension`) directly, regardless of what a
user types into the `QFileDialog` (see specs 02/03/05/07's "extension is
imposed, never user-chosen" sections, and the config-schema design). There
is no scenario where a user's typed `.h5` reaches the NXstress writer, or
vice versa — the *action clicked* determines the format and its extension
outright, so the smoke test's either/or never actually arises: a user
typing `foo.h5` into the `Save as NXstress…` dialog gets `foo.nxs` written,
silently corrected, exactly as Chris described, but enforced at the action
level rather than via a validate-then-reject/route branch.

---

## Q3 — RESOLVED: deprecation-hint mechanism and validator availability, verified against real code

A research pass checking this spec's actual GUI/tooling assumptions found
two real corrections:

- **Only `PeakFittingViewer` has a status bar today.**
  `TextureFittingViewer`, `CombineRunsViewer`, and `StrainStressViewer` are
  all code-built `QMainWindow`s with no status bar at all — only modal
  `QMessageBox`-style dialogs. **Decided:** build a status bar for all
  three (`self.statusBar()`, cheap — lazily created), rather than falling
  back to each viewer's existing modal-message pattern, for consistent
  non-blocking UX across all four viewers. `StrainStressViewer`'s
  one-time flag additionally needs to live at the window level, since its
  three direction slots share one load handler that could otherwise fire
  the hint three times per session.
- **The `nexusformat` NXstress validator referenced in this spec's (and
  specs 04's and 09's) Verification sections does not exist as an
  installed capability, and no schema doc exists in the repo either.**
  (Corrected from an earlier draft's "02" — spec 02 has no validator
  reference anywhere; spec 04's Verification section is the one that
  does.)
  Confirmed directly: the installed `nexusformat` 1.0.8 package has no
  `validate` module or `nxvalidate` script; no `NXstress.html`/`.xml`/
  `.nxdl` file exists anywhere in the repo. This traces back to spec 04b's
  open questions, which already noted both would be "added separately" —
  that hasn't happened yet. **Not resolvable in this session** — see the
  reminder added to spec 10's Overview to add both: the validator (which
  does exist, but only as a separate `nexusformat`-org repository, not
  part of the installed package) and the schema doc (to
  `docs/developer/source/design/nexus/`, linked from that directory's
  existing `IO_prototype.rst`).
