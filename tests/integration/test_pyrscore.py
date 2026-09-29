#!/usr/bin/python
import hashlib
import os
import shutil
from pathlib import Path
from typing import Callable

from pyrs.core import pyrscore, reduction_manager
from pyrs.core.workspaces import HidraWorkspace
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode  # type: ignore
import pytest

pytestmark = pytest.mark.integration


#: The one fixture still on the legacy capitalized '2Theta' schema. Named once so
#: the `xfail` reason below, `test_main` and `TestLoadDoesNotModifyTheFile` cannot
#: drift apart about which file they mean.
LEGACY_SCHEMA_FIXTURE = "Hidra_16-1_cor_log.h5"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _recording_project_file(monkeypatch):
    """Record the mode every `HidraProjectFile` is opened with, and each instance.

    `reduction_manager` resolves the name at call time, so patching the name in
    that module intercepts exactly the opens under test and nothing else.

    Returns `(opens, instances)`: a list of `(filename, mode)` and the objects
    themselves, whose `_project_h5` is `None` once closed.
    """
    opens: list[tuple[str, HidraProjectFileMode]] = []
    instances: list[HidraProjectFile] = []

    class Recording(HidraProjectFile):
        def __init__(self, name, mode=HidraProjectFileMode.READONLY):
            opens.append((Path(name).name, mode))
            super().__init__(name, mode=mode)
            instances.append(self)

    monkeypatch.setattr(reduction_manager, "HidraProjectFile", Recording)
    return opens, instances


class TestLoadDoesNotModifyTheFile:
    """Loading a project file is a read, and must leave it byte-identical.

    `ReductionManager.load_hidra_project` used to choose its HDF5 mode from
    `os.access(project_file_name, os.W_OK)` -- the file's *permissions* rather
    than what the operation needed -- so loading any writable project file
    opened it in append mode. HDF5 marks the superblock on a read-write open,
    so merely loading a file changed it on disk. The handle was also not closed
    when loading raised, and pytest's traceback keeps the frame alive, so the
    modification survived for the rest of the session. The visible symptom was
    committed fixtures under `tests/data/` appearing in `git status`.

    Both paths are covered, because only one of them leaked: a load that
    succeeds, and a load that raises part-way through.

    **The digest assertions alone cannot catch this, and did not.** Measured on
    h5py 3.16.0 / HDF5 1.14.6, a read-write open changes the bytes *only while
    the handle is open*::

        r+ open, handle OPEN : changed
        r+ after close       : unchanged
        r  open, handle OPEN : unchanged

    By the time a test takes its digest, the `HidraProjectFile` is unreferenced
    and CPython has closed it, so the mark is already reverted. Reverting
    `reduction_manager` to the full pre-fix state -- READWRITE *and* no
    `finally` -- left these tests green. The original symptom needed the handle
    held open across the whole session, which a digest taken after the call can
    never observe.

    So each test also asserts what the fix actually changed, through a recording
    `HidraProjectFile`: the **mode** it opens with, and that the handle is
    **closed** even when the load raises. Those fail against either mutation.
    The digests stay as the statement of intent -- they say what the code is
    for -- but they do not carry the guarantee.
    """

    def test_failing_load_leaves_the_file_unchanged(self, tmp_path: Path, test_data_dir, monkeypatch) -> None:
        # Arrange -- a writable copy; the bug cannot reproduce on a read-only file,
        # which is exactly why it went unnoticed on archive-mounted data.
        source = Path(test_data_dir) / LEGACY_SCHEMA_FIXTURE
        assert source.exists(), f"missing fixture {source}"
        target = tmp_path / source.name
        shutil.copy2(source, target)
        assert os.access(target, os.W_OK)
        digest_before = _digest(target)

        opens, instances = _recording_project_file(monkeypatch)

        # Act -- this fixture uses the legacy capitalized '2Theta' key, so the load
        # raises part-way through. That is the path that leaked the handle, and the
        # `match` pins it: the closure assertion below is only meaningful if the load
        # reached the point where a handle is open. A `RuntimeError` raised earlier,
        # or for an unrelated reason, would satisfy a bare `pytest.raises` while
        # exercising none of this.
        rs_core = pyrscore.PyRsCore()
        with pytest.raises(RuntimeError, match=r'non-empty "reduced diffraction data" entry'):
            rs_core.load_hidra_project(str(target), "no-mutation-on-failure", False, True)

        # Assert -- the digest states the intent; the two below carry the guarantee.
        assert _digest(target) == digest_before, "a failed load modified the project file on disk"
        assert opens == [(source.name, HidraProjectFileMode.READONLY)], (
            f"load must open READONLY regardless of the file's permissions; got {opens}"
        )
        assert all(handle._project_h5 is None for handle in instances), (
            "a load that raises must still close its handle -- an open read-write handle is "
            "what left the superblock marked for the rest of the session"
        )

    def test_successful_load_leaves_the_file_unchanged(
        self,
        write_minimal_h5_project: Callable[..., Path],
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        # Arrange -- a synthetic project that loads cleanly, so this covers the
        # normal path rather than only the error path above.
        project = write_minimal_h5_project(
            minimal_HidraWorkspace(with_instrument=True), filename="loadable.h5", with_instrument=True
        )
        assert os.access(project, os.W_OK)
        digest_before = _digest(Path(project))

        opens, instances = _recording_project_file(monkeypatch)

        # Act
        rs_core = pyrscore.PyRsCore()
        workspace = rs_core.load_hidra_project(str(project), "no-mutation-on-success", False, True)

        # Assert -- the digest states the intent; the two below carry the guarantee.
        assert workspace is not None
        assert _digest(Path(project)) == digest_before, "a successful load modified the project file on disk"
        assert opens == [(Path(project).name, HidraProjectFileMode.READONLY)], (
            f"load must open READONLY regardless of the file's permissions; got {opens}"
        )
        assert all(handle._project_h5 is None for handle in instances), "the handle must be closed after loading"


def broken_test_pole_figure_calculation():
    """
    main testing body to test the workflow to calculate pole figure
    :return:
    """
    # initialize core
    rs_core = pyrscore.PyRsCore()
    assert rs_core

    # import data file: detector ID and file name
    test_data_set = [
        (1, "tests/testdata/HB2B_exp129_Long_Al_222[1]_single.hdf5"),
        (2, "tests/testdata/HB2B_exp129_Long_Al_222[2]_single.hdf5"),
        (3, "tests/testdata/HB2B_exp129_Long_Al_222[3]_single.hdf5"),
        (4, "tests/testdata/HB2B_exp129_Long_Al_222[4]_single.hdf5"),
        (5, "tests/testdata/HB2B_exp129_Long_Al_222[5]_single.hdf5"),
        (6, "tests/testdata/HB2B_exp129_Long_Al_222[6]_single.hdf5"),
        (7, "tests/testdata/HB2B_exp129_Long_Al_222[7]_single.hdf5"),
    ]
    assert test_data_set


@pytest.mark.xfail(
    reason="tests/data/Hidra_16-1_cor_log.h5 uses the legacy capitalized '2Theta' "
    "coordinate key, which read_diffraction_2theta_array() no longer supports (it now "
    "raises a clear 'unsupported schema' RuntimeError instead of silently skipping "
    "reduced-diffraction data). This file needs migration to the current '2theta' "
    "schema -- deferred to a separate follow-up.",
    raises=RuntimeError,
    strict=True,
)
def test_main():
    """
    test main
    :return:
    """
    rs_core = pyrscore.PyRsCore()

    # pre-requisite is that the data file exists
    test_data = str(Path(__file__).resolve().parents[2] / "tests" / "data" / LEGACY_SCHEMA_FIXTURE)
    assert os.path.exists(test_data), "File {} does not exist".format(test_data)

    # load data
    test_hd_ws = rs_core.load_hidra_project(test_data, "test core", False, True)
    assert test_hd_ws is not None

    # Get sub runs
    sub_runs = rs_core.reduction_service.get_sub_runs("test core")
    assert sub_runs is not None

    # Get sample logs
    log_names = rs_core.reduction_service.get_sample_logs_names("test core")
    assert isinstance(log_names, list)

    return


# In order to test the core methods for peak fitting and thus strain/stress calculation
# default testing directory is ..../PyRS/


print(os.getcwd())
# therefore it is not too hard to locate testing data
# TODO: convert BD_Data_Log.hdf5 to Hidra_BD_Data.hdf
test_data = "tests/testdata/BD_Data_Log.hdf5"
print("Data file {0} exists? : {1}".format(test_data, os.path.exists(test_data)))


def create_test_data(rs_core, src_data_set, target_data_set):
    """
    fit peaks and output to h5 data sets as the input of strain/stress
    :param rs_core
    :param src_data_set:
    :param target_data_set:
    :return:
    """
    for direction in ["e11", "e22", "e33"]:
        # load
        src_h5_file = src_data_set[direction]
        data_key, message = rs_core.load_rs_raw(src_h5_file)
        print("Load {}: {}".format(src_h5_file, message))
        # fit
        rs_core.fit_peaks(data_key, None, "Gaussian", "Linear", [80, 90])
        # save
        target_h5_file = target_data_set[direction]
        rs_core.save_peak_fit_result(data_key, src_h5_file, target_h5_file)
    # END-FOR

    return


def modify_test_strain_calculation():
    """
    main testing body to test the workflow to calculate strain
    :return:
    """
    # initialize core
    rs_core = pyrscore.PyRsCore()

    # create testing files, aka, testing peak fitting module (core)
    target_data_set = {
        "e11": "tests/temp/LD_Data_Log.hdf5",
        "e22": "tests/temp/BD_Data_Log.hdf5",
        "e33": "tests/temp/ND_Data_Log.hdf5",
    }

    # start a session
    rs_core.new_strain_stress_session("test strain/stress module", is_plane_strain=False, is_plane_stress=False)

    # TODO - 20180810 - d0 might not be a single value but changes along grids.
    # TODO   continue   So make it possible to accept d0 in a n x 3 matrix as (x, y, z)
    rs_core.strain_stress_calculator.set_d0(d0=1.2345)
    rs_core.strain_stress_calculator.set_youngs_modulus(young_e=500.0)
    rs_core.strain_stress_calculator.set_poisson_ratio(poisson_ratio=0.23)

    # load data
    rs_core.strain_stress_calculator.load_reduced_file(file_name=target_data_set["e11"], direction="e11")
    rs_core.strain_stress_calculator.load_reduced_file(file_name=target_data_set["e22"], direction="e22")
    rs_core.strain_stress_calculator.load_reduced_file(file_name=target_data_set["e33"], direction="e33")

    # convert peak positions to d-spacing
    rs_core.strain_stress_calculator.convert_peaks_positions()

    # check and align measurement points around
    # TODO FIXME - 20181001 - Make an individual method for ....  ---> New workflow!
    if True:
        # TODO FIXME - 20181001 - This is a new suite of methods to analyze the sample grids
        # ... ...
        # set the name of the sample log for grid positions
        pos_x_sample_names = {"e11": "vx", "e22": "vx", "e33": "vx"}
        pos_y_sample_names = {"e11": "vy", "e22": "vy", "e33": "vy"}
        pos_z_sample_names = {"e11": "vz", "e22": "vz", "e33": "vz"}
        rs_core.strain_stress_calculator.set_grid_log_names(pos_x_sample_names, pos_y_sample_names, pos_z_sample_names)

        rs_core.strain_stress_calculator.check_grids_alignment()  # rename method
        rs_core.strain_stress_calculator.located_matched_grids(resolution=0.001)
    # END-Align-Grid

    # calculate unconstrained strain and stress
    strain_vec, stress_vec = rs_core.strain_stress_calculator.execute()

    # save
    rs_core.strain_stress_calculator.save_strain_stress("tests/temp/ss1.dat")

    # generate strain/stress grids
    grid_dict = {"Min": {"X": -130, "Y": None, "Z": 9.0}, "Max": {"X": 130.0, "Y": None, "Z": None}, "NotUsed": {}}
    grids, maps = rs_core.strain_stress_calculator.generate_grids(
        "e22", user_defined=False, grids_dimension_dict=grid_dict
    )

    # convert to image of slice view...
    rs_core.strain_stress_calculator.export_strain_2d(grids)

    # export
    #  rs_core.export_to_paraview(data_key, 'strain', '/tmp/stain_para.dat')

    return


def modify_test_copy_from_unconstrained_to_plane():
    """
    testing the use case such that the user copy a strain/stress calculator to another one
    :return:
    """
    # initialize core
    rs_core = pyrscore.PyRsCore()

    # create testing files, aka, testing peak fitting module (core)
    target_data_set = {
        "e11": "tests/temp/LD_Data_Log.hdf5",
        "e22": "tests/temp/BD_Data_Log.hdf5",
        "e33": "tests/temp/ND_Data_Log.hdf5",
    }

    # start a session
    rs_core.new_strain_stress_session("test strain/stress module", is_plane_strain=False, is_plane_stress=False)

    # set parameters
    rs_core.strain_stress_calculator.set_d0(d0=1.2345)
    rs_core.strain_stress_calculator.set_youngs_modulus(young_e=500.0)
    rs_core.strain_stress_calculator.set_poisson_ratio(poisson_ratio=0.23)

    # load data
    rs_core.strain_stress_calculator.load_reduced_file(file_name=target_data_set["e11"], direction="e11")
    rs_core.strain_stress_calculator.load_reduced_file(file_name=target_data_set["e22"], direction="e22")
    rs_core.strain_stress_calculator.load_reduced_file(file_name=target_data_set["e33"], direction="e33")

    # check and align measurement points around
    try:
        rs_core.strain_stress_calculator.check_grids_alignment()
    except RuntimeError as run_err:
        print("Measuring points are not aligned: {}".format(run_err))

    # create a plane strain/stress calculator
    ss_calculator = rs_core.strain_stress_calculator

    new_ss_calculator = ss_calculator.migrate(plane_strain=True, plane_stress=False)

    # re check
    new_ss_calculator.check_grids_alignment()
    new_ss_calculator.located_matched_grids(resolution=0.001)
    new_ss_calculator.execute()

    return


def modify_test_strain_stress_user_defined_grid():
    """
    Same as test_strain_calculation but using user-defined strain/stress grid
    :return:
    """
    # TESTME - 20180817 - Implement ASAP
    # initialize core
    rs_core = pyrscore.PyRsCore()

    # create testing files, aka, testing peak fitting module (core)
    target_data_set = {
        "e11": "tests/temp/LD_Data_Log.hdf5",
        "e22": "tests/temp/BD_Data_Log.hdf5",
        "e33": "tests/temp/ND_Data_Log.hdf5",
    }

    # start a session
    rs_core.new_strain_stress_session("test strain/stress module", is_plane_strain=False, is_plane_stress=False)

    # set up d0, E and nu
    rs_core.strain_stress_calculator.set_d0(d0=1.2345)
    rs_core.strain_stress_calculator.set_youngs_modulus(young_e=500.0)
    rs_core.strain_stress_calculator.set_poisson_ratio(poisson_ratio=0.23)

    # load data
    rs_core.strain_stress_calculator.load_reduced_file(file_name=target_data_set["e11"], direction="e11")
    rs_core.strain_stress_calculator.load_reduced_file(file_name=target_data_set["e22"], direction="e22")
    rs_core.strain_stress_calculator.load_reduced_file(file_name=target_data_set["e33"], direction="e33")

    # check and align measurement points around
    try:
        rs_core.strain_stress_calculator.check_grids_alignment()
    except RuntimeError as run_err:
        print("Measuring points are not aligned: {}".format(run_err))
    rs_core.strain_stress_calculator.located_matched_grids(resolution=0.001)

    # convert peak positions to d-spacing
    rs_core.strain_stress_calculator.convert_peaks_positions()

    # generate strain/stress grids
    grid_dict = {
        "Min": {"X": -130, "Y": 0.0, "Z": 9.0},
        "Max": {"X": 130.0, "Y": 0.0, "Z": 19},
        "Resolution": {"X": 5.0, "Y": None, "Z": 1.0},
    }
    grids, maps = rs_core.strain_stress_calculator.generate_grids(
        None, user_defined=True, grids_dimension_dict=grid_dict
    )
    center_d_vec = rs_core.strain_stress_calculator.align_peak_parameter_on_grids(grids, "center_d", maps)

    # calculate unconstrained strain and stress
    strain_vec, stress_vec = rs_core.strain_stress_calculator.execute(ss_grid_vec=grids, peak_pos_d_vec=center_d_vec)

    # export
    #  rs_core.export_to_paraview(data_key, 'strain', '/tmp/stain_para.dat')

    return


def modify_test_plane_strain():
    """
    Same as test_strain_calculation but using plane strain
    :return:
    """
    # TODO - 20180817 - Continue to implement
    # initialize core
    rs_core = pyrscore.PyRsCore()

    # create testing files, aka, testing peak fitting module (core)
    target_data_set = {
        "e11": "tests/temp/LD_Data_Log.hdf5",
        "e22": "tests/temp/BD_Data_Log.hdf5",
        "e33": "tests/temp/ND_Data_Log.hdf5",
    }

    # start a session
    rs_core.new_strain_stress_session("test strain/stress module", is_plane_strain=True, is_plane_stress=False)

    # set up d0, E and nu
    rs_core.strain_stress_calculator.set_d0(d0=1.2345)
    rs_core.strain_stress_calculator.set_youngs_modulus(young_e=500.0)
    rs_core.strain_stress_calculator.set_poisson_ratio(poisson_ratio=0.23)

    # load data
    rs_core.strain_stress_calculator.load_reduced_file(file_name=target_data_set["e11"], direction="e11")
    rs_core.strain_stress_calculator.load_reduced_file(file_name=target_data_set["e22"], direction="e22")

    # check and align measurement points around
    try:
        rs_core.strain_stress_calculator.check_grids_alignment()
    except RuntimeError as run_err:
        print("Measuring points are not aligned: {}".format(run_err))
    rs_core.strain_stress_calculator.located_matched_grids(resolution=0.001)

    # convert peak positions to d-spacing
    rs_core.strain_stress_calculator.convert_peaks_positions()

    # generate strain/stress grids
    grid_dict = {
        "Min": {"X": -130, "Y": 0.0, "Z": 9.0},
        "Max": {"X": 130.0, "Y": 0.0, "Z": 19},
        "Resolution": {"X": 5.0, "Y": None, "Z": 1.0},
    }
    grids, maps = rs_core.strain_stress_calculator.generate_grids(
        None, user_defined=True, grids_dimension_dict=grid_dict
    )
    center_d_vec = rs_core.strain_stress_calculator.align_peak_parameter_on_grids(grids, "center_d", maps)

    # calculate unconstrained strain and stress
    strain_vec, stress_vec = rs_core.strain_stress_calculator.execute(ss_grid_vec=grids, peak_pos_d_vec=center_d_vec)

    assert strain_vec
    assert stress_vec

    return


def modify_test_plane_stress():
    """
    Same as test_strain_calculation but using plane stress
    :return:
    """
    # TODO - 20180817 - Continue to implement
    # initialize core
    rs_core = pyrscore.PyRsCore()

    # create testing files, aka, testing peak fitting module (core)
    target_data_set = {
        "e11": "tests/temp/LD_Data_Log.hdf5",
        "e22": "tests/temp/BD_Data_Log.hdf5",
        "e33": "tests/temp/ND_Data_Log.hdf5",
    }

    # start a session
    rs_core.new_strain_stress_session("test strain/stress module", is_plane_strain=False, is_plane_stress=True)

    # set up d0, E and nu
    rs_core.strain_stress_calculator.set_d0(d0=1.2345)
    rs_core.strain_stress_calculator.set_youngs_modulus(young_e=500.0)
    rs_core.strain_stress_calculator.set_poisson_ratio(poisson_ratio=0.23)

    # load data
    rs_core.strain_stress_calculator.load_reduced_file(file_name=target_data_set["e11"], direction="e11")
    rs_core.strain_stress_calculator.load_reduced_file(file_name=target_data_set["e22"], direction="e22")

    # check and align measurement points around
    try:
        rs_core.strain_stress_calculator.check_grids_alignment()
    except RuntimeError as run_err:
        print("Measuring points are not aligned: {}".format(run_err))
    rs_core.strain_stress_calculator.located_matched_grids(resolution=0.001)

    # convert peak positions to d-spacing
    rs_core.strain_stress_calculator.convert_peaks_positions()

    # generate strain/stress grids
    grid_dict = {
        "Min": {"X": -130, "Y": 0.0, "Z": 9.0},
        "Max": {"X": 130.0, "Y": 0.0, "Z": 19},
        "Resolution": {"X": 5.0, "Y": None, "Z": 1.0},
    }
    grids, maps = rs_core.strain_stress_calculator.generate_grids(
        None, user_defined=True, grids_dimension_dict=grid_dict
    )
    center_d_vec = rs_core.strain_stress_calculator.align_peak_parameter_on_grids(grids, "center_d", maps)

    # calculate unconstrained strain and stress
    strain_vec, stress_vec = rs_core.strain_stress_calculator.execute(ss_grid_vec=grids, peak_pos_d_vec=center_d_vec)

    assert strain_vec
    assert stress_vec

    return


if __name__ == "__main__":
    pytest.main()
