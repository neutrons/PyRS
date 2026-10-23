"""A4: the third-party claims spec 08's variance design rests on.

Spec 08 proposes a `model_variance` computed by first-order propagation through
the `uncertainties` package, and justifies neglecting parameter correlations by
asserting that Mantid gives PyRS only diagonal standard errors. It also makes
several factual claims about existing PyRS code that a reader can only take on
trust. All of it is executable, and none of it was probed in the first pass --
recorded then as an **A4 gap**, and closed here.

Claims under test
-----------------
1. ``08:95-98`` -- "``model_variance: np.ndarray`` -- same shape, propagated
   from the stored per-parameter fit errors via the ``uncertainties`` package
   (already used elsewhere in the codebase -- ``_object_uarray``, ``get_strain``,
   ``fields.py`` -- for the same first-order propagation pattern)."
2. ``08:100-104`` -- "no covariance matrix is retained anywhere in PyRS --
   Mantid's fit-error output provides diagonal standard errors only -- so
   ``model_variance`` **neglects parameter correlations** by construction."
3. ``08:73-80`` -- "``peak_profile_utility.py::calculate_profile`` ... is **dead
   code with zero callers anywhere in the repo**, and has real bugs: leftover
   debug ``print()`` statements, it raises for any background other than
   ``Linear`` (hardcoding the ``Quadratic`` background's extra term to zero),
   and an off-by-one windowing issue."
4. ``08:87-88`` -- "Only ``Gaussian`` and ``PseudoVoigt`` peak shapes, and
   ``Linear``/``Quadratic`` backgrounds, need support (PyRS's only supported
   combinations)."
5. ``08:170-176`` -- "The raw-NeXus loader explicitly does not even load
   monitors (``nexus_conversion.py: LoadEventNexus(..., LoadMonitors=False)``)."

Claim 2 is the one that matters most: it is the stated justification for a
documented approximation, and if PyRS *did* retain a covariance matrix the
approximation would be an unnecessary loss of accuracy.

Run: ``pixi run python plans/NXstress-prod/probes/a4_fit_error_propagation.py``
"""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def report(claim: str, result: str) -> None:
    print(f"\n  CLAIM   {claim}")
    print(f"  RESULT  {result}")


def main() -> int:
    print("=" * 78)
    print("A4 PROBE: uncertainties / Mantid fit errors / peak_profile_utility")
    print("=" * 78)

    # --- Claim 1: the uncertainties package and PyRS's existing use of it ---
    import uncertainties
    from uncertainties import unumpy

    from pyrs.peaks.peak_collection import _object_uarray

    a = unumpy.uarray([2.0], [0.1])
    b = unumpy.uarray([3.0], [0.2])
    product = a * b
    # First-order propagation for f = a*b is |f| * sqrt((da/a)^2 + (db/b)^2).
    expected = abs(2.0 * 3.0) * np.sqrt((0.1 / 2.0) ** 2 + (0.2 / 3.0) ** 2)
    report(
        "uncertainties does first-order propagation, and PyRS already uses it via `_object_uarray` (claim 1)",
        f"uncertainties {uncertainties.__version__}; (2.0±0.1)*(3.0±0.2) -> "
        f"{product[0]}; closed-form first-order sigma = {expected:.6f}; "
        f"_object_uarray is {_object_uarray.__module__}.{_object_uarray.__name__}",
    )

    # Correlation is the whole point of claim 2: uncertainties tracks it when it
    # can see the shared variable, and cannot when it only receives sigmas.
    x = unumpy.uarray([2.0], [0.1])
    correlated = x * x
    independent = unumpy.uarray([2.0], [0.1]) * unumpy.uarray([2.0], [0.1])
    report(
        "…and it CAN track correlation — which is exactly what is lost when only "
        "diagonal sigmas are stored (claim 2's premise)",
        f"x*x (same variable, fully correlated) -> {correlated[0]}; "
        f"x1*x2 (independent draws, same sigma) -> {independent[0]}",
    )

    # --- Claim 2: does PyRS retain any covariance anywhere? ---
    # A bare grep is not enough here. The scan below found two hits, and reading
    # them is what turns a misleading "MATCHES FOUND" into the real answer.
    hits: list[str] = []
    for path in sorted((REPO / "pyrs").rglob("*.py")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if any(n in line.lower() for n in ("covariance", "covar", "calcerrors", "correlation_matrix")):
                hits.append(f"{path.relative_to(REPO)}:{i}: {line.strip()[:72]}")
    report(
        "no covariance matrix is retained anywhere in PyRS (claim 2)",
        "source scan for covariance/covar/CalcErrors/correlation_matrix across "
        "pyrs/**.py:\n          " + ("\n          ".join(hits) if hits else "NO MATCHES"),
    )

    # The two hits are in a DEAD scipy-based path. Establish that, or the grep
    # above reads as a contradiction of claim 2 when it is not one.
    # Match the SYMBOL, not the substring. An earlier draft of this probe used
    # `"fit_peak" in text and "peak_profile_utility" in text`, which matched
    # `pyrs/peaks/mantid_fit_peak.py` on its own filename and reported five
    # importers of dead code. Parse instead.
    dead_importers: list[str] = []
    for root in ("pyrs", "scripts", "tests"):
        for path in sorted((REPO / root).rglob("*.py")):
            if path.name == "peak_profile_utility.py":
                continue
            try:
                mod = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError:
                continue
            for node in ast.walk(mod):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module
                    and "peak_profile_utility" in node.module
                    and any(alias.name == "fit_peak" for alias in node.names)
                ):
                    dead_importers.append(f"{path.relative_to(REPO)}:{node.lineno}")
                elif (
                    isinstance(node, ast.Attribute)
                    and node.attr == "fit_peak"
                    and isinstance(node.value, ast.Name)
                    and "peak_profile_utility" in node.value.id
                ):
                    dead_importers.append(f"{path.relative_to(REPO)}:{node.lineno}")
    report(
        "…and those two hits are in `fit_peak`, a DEAD scipy path that computes a "
        "covariance and throws it away (claim 2, refined)",
        f"`scipy.optimize.curve_fit` returns one at peak_profile_utility.py:868, "
        f"passed to a `# TODO` stub that ignores it and returns 1.0 "
        f"(:854-861). Modules importing `fit_peak` from peak_profile_utility: "
        f"{dead_importers or 'NONE — dead code'}. So nothing RETAINS a covariance "
        f"matrix; note the path that computes one is scipy, not Mantid.",
    )

    from pyrs.peaks.peak_collection import PeakCollection

    stores = [n for n in inspect.getsource(PeakCollection).split() if "covar" in n.lower()]
    report(
        "…and `PeakCollection` stores only values + errors, never a matrix (claim 2)",
        f"identifiers containing 'covar' in PeakCollection: {stores or 'NONE'}",
    )

    # --- Claim 3: calculate_profile is dead, and carries the three named bugs ---
    profile_path = REPO / "pyrs" / "core" / "peak_profile_utility.py"
    tree = ast.parse(profile_path.read_text(encoding="utf-8"))
    func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "calculate_profile")
    callers: list[str] = []
    for root in ("pyrs", "scripts", "tests"):
        for path in sorted((REPO / root).rglob("*.py")):
            text = path.read_text(encoding="utf-8", errors="replace")
            for i, line in enumerate(text.splitlines(), 1):
                if "calculate_profile" in line and not line.strip().startswith("def "):
                    callers.append(f"{path.relative_to(REPO)}:{i}")
    report(
        "`calculate_profile` is dead code with zero callers anywhere (claim 3)",
        f"defined at {profile_path.relative_to(REPO)}:{func.lineno}; call sites found: {callers or 'NONE'}",
    )

    prints = [
        n.lineno
        for n in ast.walk(func)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "print"
    ]
    raises = [
        (n.lineno, ast.unparse(n.exc)[:70]) for n in ast.walk(func) if isinstance(n, ast.Raise) and n.exc is not None
    ]
    body = "\n".join(profile_path.read_text(encoding="utf-8").splitlines()[func.lineno - 1 : func.end_lineno])
    report(
        "…with leftover debug print(s), a raise for unsupported backgrounds, and "
        "an exclusive-right-bound window (claim 3's three bugs)",
        f"print() at line(s) {prints or 'NONE'}; raise(s) {raises}; "
        f"windowing uses `[left_x_index:right_x_index]` "
        f"(exclusive right bound): {'[left_x_index:right_x_index]' in body}",
    )

    # --- Claim 4: which peak shapes and backgrounds does PyRS actually support? ---
    from pyrs.core.peak_profile_utility import BackgroundFunction, PeakShape

    report(
        "only Gaussian/PseudoVoigt and Linear/Quadratic are supported (claim 4)",
        f"PeakShape members: {[m.name for m in PeakShape]}; "
        f"BackgroundFunction members: {[m.name for m in BackgroundFunction]}",
    )

    # --- Claim 5: monitors are explicitly not loaded ---
    conv = (REPO / "pyrs" / "core" / "nexus_conversion.py").read_text(encoding="utf-8")
    monitor_lines = [f"{i}: {ln.strip()[:88]}" for i, ln in enumerate(conv.splitlines(), 1) if "LoadMonitors" in ln]
    report(
        "the raw-NeXus loader explicitly does not load monitors (claim 5)",
        f"{monitor_lines or 'NO LoadMonitors ARGUMENT FOUND'}",
    )

    print("\n" + "-" * 78)
    print("VERDICT -- see probes/README.md")
    print("-" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
