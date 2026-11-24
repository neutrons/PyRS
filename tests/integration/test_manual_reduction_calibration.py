"""
A Manual Reduction batch gives each run the shared calibration of its own monochromator (#1096), unless the user
supplies a calibration file, which then applies to every run.

The runs are minimal synthetic NeXus files and the reduction itself is recorded, not run.
"""

import h5py
import numpy as np
import pytest

from pyrs.interface.manual_reduction import manual_reduction_model as model_module
from pyrs.interface.manual_reduction.manual_reduction_model import ManualReductionModel
from pyrs.utilities import calibration_file_io

pytestmark = pytest.mark.integration


def _run_file(path, mrot):
    with h5py.File(path, "w") as run:
        run.create_dataset("entry/DASlogs/mrot/value", data=np.array([mrot]))
    return str(path)


@pytest.fixture
def shared_folder(tmp_path, monkeypatch):
    """A shared-calibration folder with a file per monochromator, standing in for the HFIR one."""
    folder = tmp_path / "shared"
    folder.mkdir()
    for name in ("HB2B_Latest.json", "HB2B_Latest_Si422.json", "HB2B_Latest_Si331.json"):
        (folder / name).write_text("{}")
    monkeypatch.setattr(
        model_module,
        "retrieve_calibration",
        lambda run_file: calibration_file_io.retrieve_calibration(run_file, str(folder)),
    )
    return folder


@pytest.fixture
def model_and_calls(monkeypatch):
    model = ManualReductionModel()
    calls = []

    def record(nexus_file, output_dir, progressbar, **kwargs):
        calls.append((nexus_file, kwargs["calibration"]))
        return object()

    monkeypatch.setattr(model._controller, "reduce_hidra_workflow", record)
    monkeypatch.setattr(model, "set_current_run", lambda label: None)
    return model, calls


def test_batch_gives_each_run_its_monochromator_calibration(tmp_path, shared_folder, model_and_calls):
    model, calls = model_and_calls
    si422 = _run_file(tmp_path / "HB2B_1.nxs.h5", mrot=-19.2)
    si331 = _run_file(tmp_path / "HB2B_2.nxs.h5", mrot=-168.8)

    model.reduce_runs([("1", si422), ("2", si331)], str(tmp_path), None, use_default_calibration=True)

    assert calls == [
        (si422, str(shared_folder / "HB2B_Latest_Si422.json")),
        (si331, str(shared_folder / "HB2B_Latest_Si331.json")),
    ]


def test_supplied_calibration_overrides_the_default_for_every_run(tmp_path, shared_folder, model_and_calls):
    model, calls = model_and_calls
    runs = [("1", _run_file(tmp_path / "HB2B_1.nxs.h5", -19.2)), ("2", _run_file(tmp_path / "HB2B_2.nxs.h5", -168.8))]
    manual = str(tmp_path / "my_calibration.json")

    model.reduce_runs(runs, str(tmp_path), None, calibration=manual, use_default_calibration=True)

    assert [calibration for _, calibration in calls] == [manual, manual]


def test_no_default_and_no_calibration_applies_none(tmp_path, shared_folder, model_and_calls):
    model, calls = model_and_calls

    model.reduce_runs([("1", _run_file(tmp_path / "HB2B_1.nxs.h5", -19.2))], str(tmp_path), None)

    assert [calibration for _, calibration in calls] == [None]


def test_missing_monochromator_file_falls_back_and_logs(tmp_path, shared_folder, model_and_calls, monkeypatch):
    model, calls = model_and_calls
    (shared_folder / "HB2B_Latest_Si331.json").unlink()
    logged = []

    class FakeLogger:
        def __init__(self, name):
            pass

        def warning(self, message):
            logged.append(message)

        def information(self, message):
            pass

    monkeypatch.setattr(model_module, "Logger", FakeLogger)
    run = _run_file(tmp_path / "HB2B_2.nxs.h5", mrot=-168.8)

    model.reduce_runs([("2", run)], str(tmp_path), None, use_default_calibration=True)

    assert calls == [(run, str(shared_folder / "HB2B_Latest.json"))]
    assert len(logged) == 1 and "HB2B_Latest_Si331.json" in logged[0]
