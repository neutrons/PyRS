"""NXstress sample-position export, against a real HB2B dataset.

This is spec 04's `## Verification` step 2 as an executable test: write a `.nxs`
from real data and confirm `peaks/sx,sy,sz` are non-NaN.

The fixture it runs on, `tests/data/HB2B_1327_with_instrument.h5`, exists because
no file in `tests/data` could satisfy that step -- writing NXstress needs
instrument geometry and a parseable peak tag, while non-NaN positions need finite
`vx`/`vy`/`vz`, and those two sets of files were disjoint. It is built by
`tests/scripts/make_nxstress_sample_position_fixture.py`, which documents exactly
what is real in it and what was supplied; the sample coordinates under test are
copied verbatim from `HB2B_1327.h5`.

See spec 04's Follow-up 7 (F7.8, F7.10) and Decisions Log rows 24 and 26 in
`plans/NXstress-prod/README.md`.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest

from pyrs.core.workspaces import HidraWorkspace
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode  # type: ignore
from pyrs.utilities.NXstress.NXstress import NXstress

pytestmark = pytest.mark.integration

#: Resolved against the repository root, not the process working directory. A bare
#: `"tests/data/..."` works under `pixi run test-integration` -- which happens to
#: start at the root -- and errors every test here when pytest is invoked on this
#: file directly, or by an IDE runner with a different cwd.
FIXTURE_NAME = "HB2B_1327_with_instrument.h5"
GENERATOR = "tests/scripts/make_nxstress_sample_position_fixture.py"


@pytest.fixture(scope="module")
def nxstress_from_real_data(tmp_path_factory, test_data_dir):
    """Write a `.nxs` from the real HB2B fixture and hand back both sides."""
    fixture = Path(test_data_dir) / FIXTURE_NAME
    assert fixture.is_file(), f"missing fixture {fixture}; rebuild it with {GENERATOR}"

    project_file = HidraProjectFile(str(fixture), HidraProjectFileMode.READONLY)
    workspace = HidraWorkspace(str(fixture))
    workspace.load_hidra_project(project_file, load_raw_counts=False, load_reduced_diffraction=True)
    peak_collections = [project_file.read_peak_parameters(tag) for tag in project_file.read_peak_tags()]
    project_file.close()

    nxs_path = tmp_path_factory.mktemp("nxstress") / "HB2B_1327_with_instrument.nxs"
    with NXstress(nxs_path, mode="w") as nxs:
        nxs.write([workspace], [peak_collections])

    with h5py.File(nxs_path, "r") as handle:
        yield workspace, peak_collections, handle


def test_sample_position_is_not_nan(nxstress_from_real_data):
    """Verify peaks/sx,sy,sz carry real positions, not NaN placeholders."""
    # Arrange
    _, _, nxs = nxstress_from_real_data

    # Act / Assert
    for axis in ("sx", "sy", "sz"):
        values = nxs[f"entry/peaks/{axis}"][()]
        assert values.size > 0, f"peaks/{axis} is empty"
        assert not np.any(np.isnan(values)), f"peaks/{axis} still contains NaN"


def test_sample_position_matches_sample_coordinates(nxstress_from_real_data):
    """Verify the written positions are the vx/vy/vz sample coordinates."""
    # Arrange
    workspace, peak_collections, nxs = nxstress_from_real_data
    logs = workspace._sample_logs

    # The peaks index is flattened: one row per (PeakCollection, scan point), in
    # the writer's sort order. Rebuild the expected column the same way.
    from pyrs.utilities.NXstress._peaks import _Peaks

    expected = {"sx": [], "sy": [], "sz": []}
    for collection in sorted(peak_collections, key=_Peaks.PeakIndex.sort_key):
        scan_point = collection.sub_runs.raw_copy()
        point_list = logs.get_pointlist(scan_point)
        expected["sx"].append(np.asarray(point_list.vx, dtype=float))
        expected["sy"].append(np.asarray(point_list.vy, dtype=float))
        expected["sz"].append(np.asarray(point_list.vz, dtype=float))

    # Act / Assert
    for axis, blocks in expected.items():
        np.testing.assert_allclose(nxs[f"entry/peaks/{axis}"][()], np.concatenate(blocks))


def test_sample_position_is_not_the_stage_log(nxstress_from_real_data):
    """Verify the positions are NOT the identically-named sx/sy/sz logs.

    This is the distinction Decisions row 24 turns on, and it is only visible on
    real data. The two are not unrelated -- they describe the same motion in
    different frames -- which is exactly why a "not NaN" check cannot tell them
    apart. In this dataset `sx` tracks `vx` with correlation +1 and an offset of
    ~22 mm, while `sz` tracks `vz` with correlation **-1**: the stage axis runs
    opposite to the sample axis. Writing the stage logs would put a sign-flipped,
    offset coordinate into a field the schema defines as the sample position.
    """
    # Arrange
    workspace, _, nxs = nxstress_from_real_data
    logs = workspace._sample_logs
    for axis in ("sx", "sy", "sz"):
        if axis not in logs:
            pytest.skip(f"fixture carries no `{axis}` stage log to contrast against")

    # Act: build both candidate columns with the writer's own flattening, so the
    # comparison is elementwise rather than between differently-shaped value sets.
    from pyrs.utilities.NXstress._peaks import _Peaks

    _, peak_collections, _ = nxstress_from_real_data

    def flattened(log_name):
        blocks = []
        for collection in sorted(peak_collections, key=_Peaks.PeakIndex.sort_key):
            scan_point = collection.sub_runs.raw_copy()
            blocks.append(np.asarray(logs[(log_name, scan_point)], dtype=float))
        return np.concatenate(blocks)

    # Assert
    for peak_axis, coord_axis in (("sx", "vx"), ("sy", "vy"), ("sz", "vz")):
        written = nxs[f"entry/peaks/{peak_axis}"][()]
        stage = flattened(peak_axis)
        coord = flattened(coord_axis)

        # Precondition: the two logs really are distinct quantities in this file.
        assert not np.allclose(stage, coord), (
            f"precondition: `{peak_axis}` and `{coord_axis}` are indistinguishable here, "
            f"so this fixture cannot demonstrate the difference"
        )
        np.testing.assert_allclose(written, coord)
        assert not np.allclose(written, stage), (
            f"peaks/{peak_axis} matches the `{peak_axis}` stage log, not `{coord_axis}`"
        )


def test_stage_and_sample_axes_differ_in_frame(nxstress_from_real_data):
    """Verify the z stage axis runs opposite to the z sample axis.

    Recorded as a test rather than prose because it is the concrete evidence for
    `open-questions/04` Q5: the schema expresses `peaks/sx,sy,sz` in the frame
    defined by SAMPLE_DESCRIPTION's NXtransformations, which PyRS now writes, and
    this anti-correlation is what an external consumer needs that group for. If a future change makes the two frames coincide, this test fails and
    that open question needs revisiting rather than silently lapsing.
    """
    # Arrange
    workspace, _, _ = nxstress_from_real_data
    logs = workspace._sample_logs
    if not all(axis in logs for axis in ("sz", "vz")):
        pytest.skip("fixture carries no sz/vz pair")

    # Act
    stage_z = np.asarray(logs["sz"], dtype=float)
    sample_z = np.asarray(logs["vz"], dtype=float)
    correlation = np.corrcoef(stage_z, sample_z)[0, 1]

    # Assert
    # Not exactly -1: `sz` is an encoder readback and carries sub-micron noise,
    # while `vz` is the commanded grid. The sign is the point, not the precision.
    assert correlation < -0.999, f"expected the z stage axis to run opposite the z sample axis, got r={correlation}"


def test_sample_position_records_its_source(nxstress_from_real_data):
    """Verify each field names the log it came from, since the names disagree."""
    # Arrange
    _, _, nxs = nxstress_from_real_data

    # Act / Assert
    for axis, source in (("sx", "vx"), ("sy", "vy"), ("sz", "vz")):
        local_name = nxs[f"entry/peaks/{axis}"].attrs["local_name"]
        if isinstance(local_name, bytes):
            local_name = local_name.decode("utf-8")
        assert local_name == source


def test_transformations_all_reachable(nxstress_from_real_data):
    """Verify every transformation is reached from the detector, on a real file.

    The unit test asserts this on a synthetic workspace; this one confirms the
    written file is traversable as an external NXstress consumer would read it.
    """
    # Arrange
    _, _, nxs = nxstress_from_real_data
    detector = nxs["entry/instrument/DETECTOR"]
    transformations = detector["transformations"]

    def as_text(value):
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    # Act: follow depends_on from the detector until '.'
    reached = []
    current = as_text(detector["depends_on"][()])
    while current and current != ".":
        name = current.rsplit("/", 1)[-1]
        assert name in transformations, f"depends_on points outside the group: {current!r}"
        assert name not in reached, f"cycle in the depends_on chain at {name!r}"
        reached.append(name)
        current = as_text(transformations[name].attrs.get("depends_on", "."))

    # Assert
    assert set(reached) == set(transformations), f"unreachable: {sorted(set(transformations) - set(reached))}"


def test_pv_log_names_round_trip(nxstress_from_real_data):
    """Verify real PV-log names survive identifier encoding, via local_name."""
    # Arrange
    from pyrs.utilities.NXstress._definitions import decode_identifier

    _, _, nxs = nxstress_from_real_data
    logs = nxs["entry/SAMPLE_DESCRIPTION/logs"]

    # Act / Assert
    escaped = 0
    for field_name in logs:
        original = logs[field_name].attrs.get("local_name")
        if original is None:
            continue
        if isinstance(original, bytes):
            original = original.decode("utf-8")
        assert decode_identifier(field_name) == original
        escaped += field_name != original

    assert escaped > 0, "fixture exercises no escaped log names"
