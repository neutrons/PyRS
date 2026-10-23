"""A4: the Qt behaviours spec 10 and Decisions item 16a assume.

``docs/ground_truths.md`` records that this repo moved from PyQt5 to PyQt6 via
``qtpy``. A plan asserting a PyQt5 API reads perfectly and fails at import, so
every Qt claim here needs executing rather than recalling.

Claims under test
-----------------
1. ``README.md:723`` (Decisions item 11) and ``README.md:240-244`` --
   "``setEnabled``, never ``setVisible`` -- the action stays visible, grayed out
   when disabled".
2. ``05-strain-stress-viewer.md:229-230`` --
   "with ``nxstress.enable: false``, assert **Save as NXstress...** is disabled
   but still visible."
3. ``10-flip-defaults.md:40-43`` --
   "The deprecation hint needs a status bar to attach to, so this spec adds one
   (``self.statusBar()``, which creates it lazily -- cheap) to the three viewers
   that lack one."
4. ``README.md:728`` (Decisions item 16a) and ``10-flip-defaults.md:35-40`` --
   "Only ``PeakFittingViewer`` has a status bar today -- ``TextureFittingViewer``,
   ``CombineRunsViewer``, and ``StrainStressViewer`` have none, only modal
   dialogs."
5. ``10-flip-defaults.md:86-93`` -- the hint uses "``PeakFittingViewer``'s
   existing ``showMessage`` pattern".
6. ``05-strain-stress-viewer.md:195-196`` -- a file dialog filter written
   ``"NXstress (*.nxs)"``.

Claim 4 is the one worth care. ``PeakFittingViewer``'s status bar is **not**
created by ``self.statusBar()``: it is declared in
``pyrs/interface/designer/peakfitwindow.ui`` and reached as
``self.ui.statusbar``. Whether ``self.statusBar()`` returns that same object or
silently creates a second one decides whether spec 10's mechanism is even
consistent with the viewer it is modelled on.

Run under offscreen Qt::

    QT_QPA_PLATFORM=offscreen pixi run python \\
        plans/NXstress-prod/probes/a4_qtpy_qaction_statusbar.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO = Path(__file__).resolve().parents[3]


def report(claim: str, result: str) -> None:
    print(f"\n  CLAIM   {claim}")
    print(f"  RESULT  {result}")


def main() -> int:
    import qtpy
    from qtpy.QtWidgets import QApplication, QMainWindow

    print("=" * 78)
    print("A4 PROBE: qtpy / PyQt6 QAction and QStatusBar behaviour")
    print("=" * 78)
    print(f"\nqtpy {qtpy.__version__}, API {qtpy.API_NAME}, Qt {qtpy.QT_VERSION}")

    try:
        import PyQt5  # noqa: F401

        pyqt5 = "PRESENT"
    except ImportError:
        pyqt5 = "ABSENT"
    report("this repo is PyQt6 via qtpy, not PyQt5", f"import PyQt5 -> {pyqt5}")

    app = QApplication.instance() or QApplication([])

    # --- Claims 1 and 2: setEnabled leaves a QAction visible ---
    win = QMainWindow()
    menu = win.menuBar().addMenu("File")
    action = menu.addAction("Save as NXstress…")
    action.setEnabled(False)
    report(
        "setEnabled(False) leaves the action VISIBLE, merely grayed out (claims 1, 2)",
        f"isEnabled()={action.isEnabled()} isVisible()={action.isVisible()}; for contrast, after setVisible(False): ",
    )
    action.setVisible(False)
    print(f"          isEnabled()={action.isEnabled()} isVisible()={action.isVisible()}  <- setVisible contrast")
    action.setVisible(True)

    # --- Claim 3: does self.statusBar() create lazily? ---
    fresh = QMainWindow()
    had_before = fresh.findChild(type(fresh.statusBar())) is not None if False else None
    existing = [c for c in fresh.children() if c.__class__.__name__ == "QStatusBar"]
    bar = fresh.statusBar()
    after = [c for c in fresh.children() if c.__class__.__name__ == "QStatusBar"]
    report(
        "self.statusBar() creates a status bar lazily on a bare QMainWindow (claim 3)",
        f"QStatusBar children before call: {len(existing)}; after call: {len(after)}; returned {type(bar).__name__}",
    )
    _ = had_before

    # --- Claim 3/4: does it return a .ui-declared bar, or make a second one? ---
    second = fresh.statusBar()
    report(
        "calling it twice returns the SAME object (idempotent, so 'cheap' holds)",
        f"first is second: {bar is second}",
    )

    # --- Claim 5: showMessage exists on the returned object ---
    bar.showMessage("deprecation hint")
    report(
        "the PeakFittingViewer showMessage pattern works on it (claim 5)",
        f"currentMessage() -> {bar.currentMessage()!r}",
    )

    # --- Claim 4, the part that decides whether spec 10's mechanism is safe ---
    # PeakFittingViewer's bar comes from a .ui file and is reached as
    # `self.ui.statusbar`. If `self.statusBar()` on such a window created a
    # SECOND bar, spec 10's instruction would be actively wrong for the one
    # viewer it cites as precedent.
    from qtpy.QtWidgets import QStatusBar

    ui_like = QMainWindow()
    declared = QStatusBar(ui_like)
    ui_like.setStatusBar(declared)
    returned = ui_like.statusBar()
    bars = [c for c in ui_like.children() if c.__class__.__name__ == "QStatusBar"]
    report(
        "on a window whose status bar came from a .ui file, does self.statusBar() "
        "return THAT bar or create a second one? (claim 3 vs claim 4)",
        f"statusBar() is the declared bar: {returned is declared}; QStatusBar children: {len(bars)}",
    )

    # --- Claim 4: which viewers actually have a status bar today? ---
    ui_dir = REPO / "pyrs" / "interface" / "designer"
    with_bar = sorted(p.name for p in ui_dir.glob("*.ui") if "QStatusBar" in p.read_text(errors="replace"))
    grep = subprocess.run(
        ["grep", "-rl", "statusBar()", str(REPO / "pyrs" / "interface")],
        capture_output=True,
        text=True,
        check=False,
    )
    report(
        "Only PeakFittingViewer has a status bar today (claim 4)",
        f".ui files declaring a QStatusBar: {with_bar}; "
        f"source files calling self.statusBar(): {grep.stdout.split() or 'NONE'}",
    )

    # --- Claim 6: file-dialog filter syntax ---
    from qtpy.QtWidgets import QFileDialog

    dialog = QFileDialog()
    dialog.setNameFilters(["NXstress (*.nxs)", "HidraProjectFile (*.h5)"])
    report(
        "the filter string form specs 02/03/05 use is accepted verbatim (claim 6)",
        f"nameFilters() -> {dialog.nameFilters()}",
    )

    _ = app
    print("\n" + "-" * 78)
    print("VERDICT -- see probes/README.md")
    print("-" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
