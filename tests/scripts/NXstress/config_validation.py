"""Configuration is validated against a schema, and a bad value names its key.

Shows that:

1. each malformed value is refused with a `RuntimeError` naming the key, where
   it used to be accepted: an extension with its dot, the YAML string `"false"`
   for a boolean, a bare string for the discriminator list, a repeated
   discriminator;
2. several bad values are reported together, one line per key;
3. a key PyRS does not define is refused, whether misspelled, in the wrong
   section, or left over from a removed setting, while the sections
   `neutrons_standard` adds itself are accepted;
4. disabling both output formats is refused with `ValueError`;
5. a discriminator naming a read-only `HidraWorkspace` property is refused when
   NXstress reads the configuration, since its value could not be restored;
6. in production, an override file named by the `env` environment variable is
   validated when PyRS starts, so a bad file, including one with a misspelled
   key, stops it with the same message;
7. `nxstress.instrument_name` is what is written to the file.

Run from the repository root:

    pixi run python tests/scripts/NXstress/config_validation.py [output-dir]
"""

import os
import subprocess
import sys

import h5py

from _fixtures import REPO, check, config_override, nxstress_config, output_dir, run, section
from _fixtures import synthetic_peaks, synthetic_workspace

from pyrs.utilities.NXstress.NXstress import NXstress

BAD_VALUES = [
    ("nxstress.extension", {"nxstress": {"extension": ".nxs"}}),
    ("nxstress.enable", {"nxstress": {"enable": "false"}}),
    ("nxstress.discriminator_fields", {"nxstress": {"discriminator_fields": "direction"}}),
    ("nxstress.discriminator_fields", {"nxstress": {"discriminator_fields": ["direction", "direction"]}}),
    ("legacy_io.extension", {"legacy_io": {"extension": ""}}),
]


def refusal(exception: type[BaseException], overrides: dict) -> str | None:
    """The message applying `overrides` raised `exception` with, or None if it was accepted."""
    try:
        with config_override(overrides):
            pass
    except exception as error:
        return str(error)
    return None


UNKNOWN_KEYS = [
    ("nxstress.discriminator_field", {"nxstress": {"discriminator_field": ["direction"]}}),
    ("discriminator_fields", {"discriminator_fields": ["direction"]}),
    ("nxstress.merge_workspaces", {"nxstress": {"merge_workspaces": True}}),
]


def write_one(path) -> None:
    ws = synthetic_workspace([1, 2, 3])
    with NXstress(path, "w") as nx:
        nx.write([ws], [[synthetic_peaks(ws)]])


def main() -> None:
    out = output_dir()

    section("each bad value is refused, naming its key")
    for key, overrides in BAD_VALUES:
        message = refusal(RuntimeError, overrides)
        detail = next((line.strip() for line in (message or "").splitlines() if line.strip().startswith(key)), None)
        print(f"  {overrides} -> {detail}")
        check(f"refused, naming {key}", detail is not None)

    section("several bad values are reported together")
    message = refusal(RuntimeError, {"nxstress": {"extension": ".nxs", "use_production_names": "yes"}})
    print("  " + (message or "").replace("\n", "\n  "))
    check("both keys are named", "nxstress.extension" in message and "nxstress.use_production_names" in message)

    section("a key PyRS does not define is refused")
    for key, overrides in UNKNOWN_KEYS:
        message = refusal(RuntimeError, overrides)
        detail = next((line.strip() for line in (message or "").splitlines() if line.strip().startswith(key)), None)
        print(f"  {overrides} -> {detail}")
        check(f"refused, naming {key}", detail is not None)
    message = refusal(RuntimeError, {"environment": "pyrs-user", "application": {"version": "1.2.3"}})
    print(f"  neutrons_standard's own sections -> {'accepted' if message is None else message}")
    check("neutrons_standard's own sections are accepted", message is None)

    section("at least one output format must stay enabled")
    message = refusal(ValueError, {"nxstress": {"enable": False}, "legacy_io": {"enable": False}})
    print(f"  both disabled -> ValueError: {message}")
    check("refused", message is not None)

    section("a discriminator that could not be restored")
    with nxstress_config(discriminator_fields=["name"]):
        try:
            write_one(out / "unrestorable.nxs")
            message = None
        except ValueError as error:
            message = str(error).splitlines()[0]
    print(f"  discriminator_fields: [name] -> ValueError: {message}")
    check("refused before anything is written", message is not None)

    section("in production: a bad override file stops PyRS at start-up")
    bad = out / "bad_override.yml"
    bad.write_text("nxstress:\n  extension: .nxs\n  enable: 'false'\n  discriminator_field: [direction]\n")
    result = subprocess.run(
        [sys.executable, "-c", "import pyrs.utilities.config"],
        cwd=REPO,
        env={**os.environ, "env": str(bad)},
        capture_output=True,
        text=True,
    )
    lines = [line for line in result.stderr.splitlines() if line.strip()]
    starts = [n for n, line in enumerate(lines) if line.startswith("RuntimeError: Invalid PyRS configuration")]
    report = lines[starts[0] :] if starts else lines[-3:]
    print(f"  env={bad.name}: exit status {result.returncode}")
    print("  " + "\n  ".join(report))
    check("PyRS does not start", result.returncode != 0)
    check("every key is named", all(any(key in line for line in report)
          for key in ("nxstress.extension", "nxstress.enable", "nxstress.discriminator_field")))

    section("the instrument name comes from configuration")
    path = out / "instrument_name.nxs"
    with nxstress_config(instrument_name="HB2B-test", instrument_short_name="T"):
        write_one(path)
    with h5py.File(path, "r") as f:
        name = f["entry/instrument/name"]
        value = name[()].decode() if isinstance(name[()], bytes) else str(name[()])
        short = name.attrs["short_name"]
        short = short.decode() if isinstance(short, bytes) else str(short)
    print(f"  instrument/name = {value!r}, short_name = {short!r}")
    check("the configured name is written", (value, short) == ("HB2B-test", "T"))


if __name__ == "__main__":
    run(main)
