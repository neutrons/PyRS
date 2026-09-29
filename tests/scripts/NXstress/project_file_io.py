"""Project-file I/O changes that NXstress relies on.

Shows that:

1. a string sample log is `str` in memory, whatever it was built from, and is
   saved to a Hidra project file as variable-length UTF-8;
2. loading a project file opens it READONLY, closes it, and leaves it
   byte-for-byte unchanged, even when the file is writable;
3. `restore()` sets a restorable property such as `HidraWorkspace.direction`,
   the value survives an NXstress round trip, and a property that is not
   restorable is refused;
4. the strain/stress viewer's model refuses a project file whose `direction` log
   disagrees with the direction it was selected as. The file, a real HB2B run
   with a `direction` log of "22" added, is left in the output directory for the
   GUI steps in `plans/PRs/phase-2.md`.

Run from the repository root:

    pixi run python tests/scripts/NXstress/project_file_io.py [output-dir]
"""

import os
import shutil

import h5py
import numpy as np

from _fixtures import TEST_DATA, check, nxstress_config, output_dir, run, section, sha256
from _fixtures import synthetic_peaks, synthetic_workspace

from pyrs.core import pyrscore, reduction_manager
from pyrs.interface.strainstressviewer.model import Model
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode
from pyrs.utilities.NXstress.NXstress import NXstress
from pyrs.utilities.restorable_property import restore

LOGS = "raw data/logs"


def save_project(ws, path) -> None:
    project = HidraProjectFile(str(path), HidraProjectFileMode.OVERWRITE)
    ws.save_experimental_data(project, sub_runs=ws.get_sub_runs(), ignore_raw_counts=True)
    ws.save_reduced_diffraction_data(project)
    project.save(False)


def string_logs(out) -> None:
    section("string sample logs")
    ws = synthetic_workspace([1, 2, 3])
    points = ws.get_sub_runs().raw_copy()
    ws.set_sample_log("comment", points, np.array([b"as bytes", b"from a file", b"reader"]))
    ws.set_sample_log("operator", points, np.array(["as", "numpy", "str"]))
    for name in ("comment", "operator"):
        value = ws.get_sample_log_values(name)[0]
        print(f"  in memory: {name}[0] = {value!r} ({type(value).__name__})")
        check(f"{name} is str in memory", isinstance(value, str))

    path = out / "string_logs.h5"
    save_project(ws, path)
    with h5py.File(path, "r") as f:
        for name in ("comment", "operator"):
            info = h5py.check_string_dtype(f[LOGS][name].dtype)
            print(f"  on disk: {name} has encoding {info and info.encoding}, length {info and info.length}")
            check(
                f"{name} is variable-length UTF-8 on disk",
                info is not None and info.encoding == "utf-8" and info.length is None,
            )


def readonly_load(out) -> None:
    section("loading a project file does not modify it")
    path = out / "loadable.h5"
    save_project(synthetic_workspace([1, 2, 3]), path)
    check("the copy is writable", os.access(path, os.W_OK))
    before = sha256(path)

    # The digest alone cannot show the defect: a read-write open changes the
    # bytes only while the handle is open. So record the mode and the closure.
    opens, handles = [], []

    class Recording(HidraProjectFile):
        def __init__(self, name, mode=HidraProjectFileMode.READONLY):
            opens.append(mode)
            super().__init__(name, mode=mode)
            handles.append(self)

    original = reduction_manager.HidraProjectFile
    reduction_manager.HidraProjectFile = Recording
    try:
        pyrscore.PyRsCore().load_hidra_project(str(path), "demo", False, True)
    finally:
        reduction_manager.HidraProjectFile = original

    print(f"  opened with {[mode.name for mode in opens]}; closed: {[h._project_h5 is None for h in handles]}")
    check("opened READONLY", opens == [HidraProjectFileMode.READONLY])
    check("closed after loading", all(h._project_h5 is None for h in handles))
    check("the file is byte-for-byte unchanged", sha256(path) == before)


def restorable(out) -> None:
    section("restore() and restorable properties")
    ws = synthetic_workspace([1, 2, 3])
    restore(ws, "direction", "22")
    print(f"  after restore(ws, 'direction', '22'): ws.direction = {ws.direction!r}")
    check("restore sets the property", ws.direction == "22")

    path = out / "restored_direction.nxs"
    with nxstress_config(discriminator_fields=["direction"]):
        with NXstress(path, "w") as nx:
            nx.write([ws], [[synthetic_peaks(ws)]])
        with NXstress(path, "r") as nx:
            (back,), _ = nx.read()
    print(f"  after an NXstress round trip: direction = {back.direction!r}")
    check("the direction survives the round trip", back.direction == "22")

    try:
        restore(ws, "name", "other")
        message = None
    except TypeError as error:
        message = str(error).splitlines()[0]
    print(f"  restore(ws, 'name', ...) -> TypeError: {message}")
    check("a property that is not restorable is refused", message is not None)


def direction_mismatch(out) -> None:
    section("the strain/stress viewer refuses a disagreeing direction")
    path = out / "HB2B_1320_direction_22.h5"
    shutil.copy2(TEST_DATA / "HB2B_1320.h5", path)
    with h5py.File(path, "a") as f:
        n = f[LOGS]["sub-runs"].shape[0]
        f[LOGS].create_dataset("direction", data=np.array(["22"] * n, dtype=h5py.string_dtype("utf-8")))

    model = Model()
    failures = []
    model.failureMsg.connect(lambda *args: failures.append(args))

    ws, _ = model.load_hidra_project_file(str(path), "11")
    reported = " ".join(str(part) for part in failures[0]) if failures else ""
    print(f"  selected as 11 -> {reported.splitlines()[0] if reported else 'loaded'}")
    check("refused and reported", ws is None and "'22'" in reported)

    ws, peaks = model.load_hidra_project_file(str(path), "22")
    print(f"  selected as 22 -> loaded, {len(ws.get_sub_runs())} sub-runs, peaks {list(peaks)}")
    check("loads under the direction it records", ws is not None and ws.direction == "22")
    print(f"  Left for the GUI steps: {path}")


def main() -> None:
    out = output_dir()
    string_logs(out)
    readonly_load(out)
    restorable(out)
    direction_mismatch(out)


if __name__ == "__main__":
    run(main)
