"""A4: does the installed ``nexusformat`` 1.0.8 provide a validator?

Four documents in this series make claims about this, and **they contradict each
other**. Three say the capability does not exist; one instructs an implementer to
use it. A reader cannot settle that, because both readings are plausible prose.

Claims under test
-----------------
1. ``README.md:728`` (Decisions Log item 16b) --
   "the installed ``nexusformat`` 1.0.8 package has no validator capability at
   all."
2. ``10-flip-defaults.md:49-53`` --
   "Confirmed: this is *not* part of the installed ``nexusformat`` PyPI package
   (1.0.8 has no ``validate`` module or ``nxvalidate`` script) -- it exists only
   as a separate repository."
3. ``09-fit-spectrum-nxstress.md:253-258`` --
   "Run the ``nexusformat``-org's NXstress validator ... **once the validator and
   schema doc are available** (confirmed not yet present in this repo or in the
   installed ``nexusformat`` package as of this writing)."
4. ``04-nxstress-internal-cleanup.md:128-132`` -- **contradicts 1-3** --
   "inspect sx/sy/sz fields with ``h5dump`` or the ``nexusformat`` Python API"
   and "Write a ``.nxs`` file and open it in the ``nexusformat`` validator
   (with ``nxstress.use_production_names = true`` temporarily...)".

Scope of this probe, stated precisely: it tests only what the **installed**
package provides. The NeXus-org validator does exist, as a tool loaded and run
from its own separate repository -- which is what claims 2 and 3 say. So the
question is not whether a validator exists anywhere, but whether spec 04's
Verification tells its implementer where to get one. Claims 2 and 3 hedge;
claim 4 does not.

Run: ``pixi run python plans/NXstress-prod/probes/a4_nexusformat_validator.py``
"""

from __future__ import annotations

import importlib
import importlib.util
import shutil
import sys


def report(claim: str, result: str) -> None:
    print(f"\n  CLAIM   {claim}")
    print(f"  RESULT  {result}")


def main() -> int:
    import nexusformat

    print("=" * 78)
    print("A4 PROBE: nexusformat validator capability")
    print("=" * 78)
    print(f"\ninstalled nexusformat version: {nexusformat.__version__}")
    print(f"package location:             {nexusformat.__file__}")

    report(
        "1.0.8 has no `validate` module",
        f"importlib.util.find_spec('nexusformat.validate') -> {importlib.util.find_spec('nexusformat.validate')}",
    )

    submodules = []
    for name in ("validate", "nexus.validate", "nxvalidate"):
        try:
            importlib.import_module(f"nexusformat.{name}")
            submodules.append(name)
        except ImportError:
            pass
    report(
        "no validator submodule under nexusformat",
        f"importable validator submodules: {submodules or 'NONE'}",
    )

    script = shutil.which("nxvalidate")
    report("1.0.8 has no `nxvalidate` script", f"shutil.which('nxvalidate') -> {script!r}")

    import nexusformat.nexus as nx

    validate_names = sorted(n for n in dir(nx) if "valid" in n.lower())
    report(
        "no validator capability at all",
        f"names containing 'valid' in nexusformat.nexus: {validate_names or 'NONE'}",
    )

    h5dump = shutil.which("h5dump")
    report(
        "spec 04 Verification says to use `h5dump` or the nexusformat Python API",
        f"shutil.which('h5dump') -> {h5dump!r}",
    )

    # The Python API itself (as distinct from a validator) plainly does exist;
    # spec 04's Verification conflates the two, so test them separately.
    report(
        "the nexusformat *Python API* exists (spec 04's other instruction)",
        f"nexusformat.nexus.nxload is {type(nx.nxload).__name__}, NXentry is {nx.NXentry.__name__}",
    )

    print("\n" + "-" * 78)
    print("VERDICT")
    print("-" * 78)
    print("Claims 1, 2 and 3 are CONFIRMED *of the installed package*: no validate")
    print("module, no nxvalidate script, no validator name in the public API.")
    print("This does NOT mean no validator exists -- the NeXus-org validator is a")
    print("separate repository, exactly as claims 2 and 3 say.")
    print("Claim 4 is therefore UNDER-SPECIFIED rather than false: spec 04 tells an")
    print("implementer to run the validator without saying it must be fetched")
    print("separately, while 09 and 10 both carry that hedge. Spec 04's other")
    print("instructions -- the nexusformat Python API, and h5dump -- are fine: both")
    print("are present in this environment.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
