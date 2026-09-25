# Open Questions — 02 NXstress I/O for PeakFitting & Texture Viewers

**Spec:** [02-peak-and-texture-nxstress.md](../02-peak-and-texture-nxstress.md)
**Blocking:** No

---

## Q1 — Does `NXstress.read()`'s reconstructed subset satisfy these two viewers' load paths?

The plan-level README notes (§2.4, "Read-back completeness", README.md:353-360):
> `NXstress.read` reconstructs wavelengths, sample logs, masks, reduced
> diffraction data, and peak collections. It does NOT reconstruct raw counts
> unless the optional `input_data` group was written. Verify each viewer's
> load path is satisfied by this subset during Phase 1 wiring.

This spec's own scope does not include an explicit check of that assumption
for PeakFittingViewer / TextureFittingViewer specifically — it's asserted
implicitly by "the natural first integration point" framing (README.md:93-95)
but never verified against what `PeakFittingModel.load_hidra_project` /
`TextureFittingModel.load_hidra_project_file` actually read from a
`HidraProjectFile` today (e.g., do either ever touch raw detector counts?).

**Why it matters:** if either viewer's existing load path reads a field that
`NXstress.read()` doesn't reconstruct by default (raw counts being the
flagged example), the round-trip tests in this spec would need to write
files with the optional `input_data` group included, or the viewer would
silently lose data on an NXstress round-trip.

**Next step:** during round-trip test implementation, diff the fields read
by the `.h5` load path against what `NXstress.read()` returns for the same
workspace; confirm no gap for these two viewers specifically.


**Response from Chris:** The PeakFittingViewer / TextureFittingViewer UIs load different sets of diffraction data. PeakFittingViewer **only** loads the default "mask". TextureFittingViewer can load either the "defualt" or "eta_*" masks, and the workflow is for the Viewer to load "eta_*" if present with a fallback to the "default" if "eta_*" are not present.
