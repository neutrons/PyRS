"""A4: the ``neutrons_standard.Config`` contract README section 2.3 rests on.

Section 2.3 makes at least eight independently falsifiable claims about a
third-party package, and spec 01 has already shipped code that depends on all of
them. Each requires *executing* something; none is knowable by reading.

The import-ordering claim in particular cannot be tested in-process at all --
once ``neutrons_standard.config`` is imported, its module-level
``package_name = Spec.client_package_name`` is bound for the life of the
interpreter. So each ordering variant runs in its own subprocess.

Claims under test
-----------------
1. ``README.md:267-270`` -- "``neutrons_standard.init("pyrs")``, which must run
   before ``neutrons_standard.config`` is ever imported".
2. ``README.md:270-273`` -- "a stray direct import would **race** ``init()`` and
   silently corrupt ``package_name`` for the whole process."
3. ``README.md:223-225`` -- "A YAML config file at ``pyrs/resources/application.yml``
   -- this exact filename and location (a genuine ``pyrs.resources`` subpackage)
   is a hard requirement of ``neutrons_standard``, not a PyRS convention."
4. ``README.md:276-280`` -- "overridden by setting the ``env`` OS environment
   variable to the name or path of a ``.yml`` file, whose contents are
   deep-merged on top of the shipped default".
5. ``README.md:280-282`` -- "``neutrons_standard`` also auto-loads a
   ``~/.pyrs/pyrs-user.yml`` override when one exists and no explicit ``env`` is
   set."
6. ``README.md:249-252`` -- "``neutrons_standard.Config`` provides no schema
   validation of its own".
7. ``README.md:274-275`` -- "Callsites use dot-string key access directly
   (``Config["nxstress.enable"]``)".
8. ``README.md:220-222`` -- "no ``pyyaml`` dependency needed;
   ``neutrons_standard.Config`` handles its own YAML I/O internally."

Two behaviours no plan document mentions are probed as well, because an audit
that only checks the claims that were made cannot find the ones that were not:

9.  ``_Config.__init__`` calls ``persistBackup()``, which **writes to the user's
    home directory** on every load.
10. ``_find_root_dir()`` redirects the resources root to ``<repo>/tests/`` when
    ``isTestEnv()`` -- that is, whenever the ``env`` variable contains "test"
    *and* ``conftest`` is in ``sys.modules``. Spec 01 ships a test framework.

Run: ``pixi run python plans/NXstress-prod/probes/a4_neutrons_standard_config.py``
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def report(claim: str, result: str) -> None:
    print(f"\n  CLAIM   {claim}")
    print(f"  RESULT  {result}")


def run(code: str, env: dict[str, str] | None = None) -> str:
    """Run a snippet in a fresh interpreter; return its last non-empty line."""
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        capture_output=True,
        text=True,
        env=full_env,
        check=False,
    )
    out = (proc.stdout + proc.stderr).strip().splitlines()
    return out[-1] if out else "(no output)"


def main() -> int:
    print("=" * 78)
    print("A4 PROBE: neutrons_standard.Config contract (README section 2.3)")
    print("=" * 78)
    import neutrons_standard

    print(f"\nneutrons_standard {neutrons_standard.__version__}")

    # --- Claims 1 and 2: import ordering, and what "race" actually means ---
    correct = run(
        "import neutrons_standard; neutrons_standard.init('pyrs');"
        "from neutrons_standard.config import Config;"
        "print('OK package_name=', __import__('neutrons_standard.config', fromlist=['x']).package_name)"
    )
    report("init() before importing config works (claim 1)", correct)

    wrong = run(
        "import neutrons_standard.config as c; import neutrons_standard;"
        "neutrons_standard.init('pyrs'); print('package_name=', repr(c.package_name))"
    )
    report(
        "importing config BEFORE init() -- is package_name silently corrupted, or does it fail loudly? (claims 1, 2)",
        wrong,
    )

    # --- Claim 3: is the resources location a hard requirement? ---
    location = run(
        "import neutrons_standard; neutrons_standard.init('pyrs');"
        "from neutrons_standard.config import Resource, _find_root_dir;"
        "print('root=', _find_root_dir(), '| application.yml path=', Resource.getPath('application.yml'),"
        "'| exists=', Resource.exists('application.yml'))"
    )
    report("pyrs/resources/application.yml is a hard requirement (claim 3)", location)

    missing = run(
        "import neutrons_standard; neutrons_standard.init('nonexistent_pkg_xyz');"
        "from neutrons_standard.config import Config; print('loaded')"
    )
    report("what happens when the client package has no resources (claim 3)", missing)

    # --- Claim 4: env= deep merge ---
    tmp = Path(tempfile.mkdtemp(prefix="a4_ns_"))
    override = tmp / "override.yml"
    override.write_text("nxstress:\n  enable: false\n")
    merged = run(
        "import neutrons_standard; neutrons_standard.init('pyrs');"
        "from neutrons_standard.config import Config;"
        "print('nxstress.enable=', Config['nxstress.enable'],"
        "'| nxstress.extension=', Config['nxstress.extension'],"
        "'| legacy_io.enable=', Config['legacy_io.enable'])",
        env={"env": str(override)},
    )
    report(
        "env=<file> DEEP-merges over the shipped default -- siblings survive "
        "(claim 4). Override sets only nxstress.enable=false.",
        merged,
    )

    # --- Claim 5: ~/.pyrs/pyrs-user.yml auto-load ---
    user_yml = Path.home() / ".pyrs" / "pyrs-user.yml"
    auto = run(
        "import neutrons_standard; neutrons_standard.init('pyrs');"
        "from neutrons_standard.config import Config, _Config;"
        "print('userHome=', _Config._userHome(), '| expects=', _Config._userHome()/'pyrs-user.yml',"
        "'| shouldSwap=', Config.shouldSwapToUserYml())"
    )
    report(
        f"auto-loads ~/.pyrs/pyrs-user.yml when present and no env set (claim 5). "
        f"That file currently exists: {user_yml.exists()}",
        auto,
    )

    # --- Claim 6: no schema validation of its own ---
    junk = tmp / "junk.yml"
    junk.write_text("nxstress:\n  enable: 'not-a-boolean'\n  nonsense_key: 17\n")
    novalidate = run(
        "import neutrons_standard; neutrons_standard.init('pyrs');"
        "from neutrons_standard.config import Config;"
        "print('enable=', repr(Config['nxstress.enable']), '| nonsense_key=', repr(Config['nxstress.nonsense_key']))",
        env={"env": str(junk)},
    )
    report(
        "Config provides NO schema validation of its own -- a wrong type and an "
        "unknown key both load silently (claim 6)",
        novalidate,
    )

    # --- Claim 7: dot-string key access ---
    dotted = run(
        "import neutrons_standard; neutrons_standard.init('pyrs');"
        "from neutrons_standard.config import Config;"
        "print('dot-string ->', Config['nxstress.enable'], '| missing key ->', end=' ');"
        "\ntry:\n    print(Config['nxstress.no_such_key'])\nexcept Exception as e:\n    print(type(e).__name__, e)"
    )
    report("dot-string key access, and what a missing key does (claim 7)", dotted)

    # --- Claim 8: no pyyaml dependency needed ---
    # NOTE: this must run in a subprocess. An earlier draft of this probe did
    # `import neutrons_standard.config` here, in a main process that had never
    # called init(), and crashed with `ModuleNotFoundError: No module named
    # 'None'` -- accidentally demonstrating claim 2's failure mode. That crash is
    # now reproduced deliberately below rather than by accident.
    yaml_use = run(
        "import neutrons_standard; neutrons_standard.init('pyrs');"
        "import neutrons_standard.config as nsc, sys;"
        "print('config module uses yaml=', nsc.yaml.__name__, 'and ruamel=', nsc.rYAML.__name__,"
        "'| YAML packages loaded:', sorted({m.split('.')[0] for m in sys.modules"
        " if m.split('.')[0] in {'yaml','ruamel'}}))"
    )
    report("no pyyaml dependency needed -- Config handles YAML internally (claim 8)", yaml_use)

    pyrs_deps = run(
        "import tomllib, pathlib;"
        "d = tomllib.loads(pathlib.Path('pyproject.toml').read_text());"
        "deps = str(d.get('tool', {}).get('pixi', {}).get('dependencies', {}));"
        "print('pyyaml in pyrs pixi dependencies:', 'yaml' in deps.lower())"
    )
    report("...and PyRS itself therefore declares no pyyaml dependency (claim 8)", pyrs_deps)

    # --- Claim 9 (undocumented): a write to the user's home on every load ---
    backup = Path.home() / ".pyrs" / "application.yml.bak"
    existed = backup.exists()
    mtime_before = backup.stat().st_mtime if existed else None
    run("import neutrons_standard; neutrons_standard.init('pyrs'); from neutrons_standard.config import Config")
    mtime_after = backup.stat().st_mtime if backup.exists() else None
    report(
        "UNDOCUMENTED: _Config.__init__ -> persistBackup() writes to the user's HOME on every load",
        f"{backup} existed_before={existed}; mtime changed by a bare Config import: {mtime_before != mtime_after}",
    )

    # --- Claim 10 (undocumented): the test-environment root redirect ---
    testenv = run(
        "import sys, types; sys.modules['conftest'] = types.ModuleType('conftest');"
        "import neutrons_standard; neutrons_standard.init('pyrs');"
        "from neutrons_standard.config import _find_root_dir, isTestEnv;"
        "print('isTestEnv=', isTestEnv(), '| resources root=', _find_root_dir())",
        env={"env": "integration_test.yml"},
    )
    report(
        "UNDOCUMENTED: with env containing 'test' AND conftest imported, the "
        "resources root moves to <repo>/tests/ -- spec 01 ships a test framework",
        testenv,
    )

    print("\n" + "-" * 78)
    print("VERDICT -- see probes/README.md")
    print("-" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
