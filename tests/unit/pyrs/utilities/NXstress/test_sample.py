"""
Tests for pyrs/utilities/NXstress/_sample.py
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
from nexusformat.nexus import NXcollection, NXentry, NXsample, nxopen
import pytest

from pyrs.core.polefigurecalculator import PoleFigureCalculator
from pyrs.core.workspaces import HidraWorkspace
from pyrs.dataobjects.constants import HidraConstants
from pyrs.utilities.NXstress._sample import _Sample
from pyrs.utilities.NXstress._definitions import FIELD_DTYPE


def _rodrigues(axis, degrees: float) -> np.ndarray:
    """Right-handed rotation of `degrees` about `axis`.

    Written here rather than imported: the point of
    `test_the_rotations_compose_in_write_order` is to check `_sample.py`'s chain
    against an independent construction, so sharing one would make it circular
    (`plans/audit-process/process.md` section 5.3).
    """
    k = np.asarray(axis, dtype=float)
    k = k / np.linalg.norm(k)
    theta = np.radians(degrees)
    cross = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * cross + (1.0 - np.cos(theta)) * (cross @ cross)


def _lab_from_sample(omega: float, chi: float, phi: float) -> np.ndarray:
    """The sample -> lab matrix implied by `rotate_project_q`, for one angle set.

    The *referent*, so it is driven through the real function rather than
    reimplemented. `rotate_project_q` returns `(90 - alpha, beta)` for a probe
    vector fixed by its `theta`/`eta` arguments; aiming that probe vector along
    each basis vector in turn reads off the columns of the lab -> sample matrix,
    and `alpha`/`beta` invert to `(h1, h2)` exactly. The third row is the cross
    product of the first two, which is forced once they are orthonormal.

    Derived in `plans/NXstress-prod/probes/a5_sample_rotation_axes.py`.
    """
    calculator = PoleFigureCalculator()
    columns = []
    for theta, eta in ((90.0, 0.0), (0.0, 90.0), (0.0, 0.0)):
        elevation, beta = calculator.rotate_project_q(theta, omega, chi, phi, eta)
        length = np.sin(np.radians(elevation))
        columns.append([length * np.cos(np.radians(beta)), length * np.sin(np.radians(beta))])
    rows = np.asarray(columns).T
    lab_from_sample = np.vstack([rows, np.cross(rows[0], rows[1])]).T
    assert np.allclose(lab_from_sample @ lab_from_sample.T, np.eye(3), atol=1e-9)
    return lab_from_sample


class TestSample:
    """Test suite for _sample.py"""

    def test_Sample_scan_point_and_coordinates(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
    ):
        """Verify scan_point matches subruns and vx,vy,vz have correct shape/dtype"""
        ws = minimal_HidraWorkspace(with_instrument=False)

        sample = _Sample.init_group([ws._sample_logs])

        assert isinstance(sample, NXsample)
        assert "scan_point" in sample
        assert "vx" in sample
        assert "vy" in sample
        assert "vz" in sample

        # Verify scan_point matches subruns
        subruns = ws._sample_logs.subruns.raw_copy()
        N_scan = len(subruns)

        np.testing.assert_array_equal(sample["scan_point"].nxdata, subruns.astype(FIELD_DTYPE.INT_DATA.value))

        # Verify coordinate arrays have correct shape and dtype
        for coord in ["vx", "vy", "vz"]:
            assert sample[coord].shape == (N_scan,)
            assert sample[coord].dtype == FIELD_DTYPE.FLOAT_DATA.value

    def test_Sample_chemical_formula_present(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
    ):
        """Verify chemical_formula field when CHEMICAL_FORMULA log is present"""
        ws = minimal_HidraWorkspace(with_instrument=False)

        # Add chemical formula to logs - must match number of subruns
        subruns = ws._sample_logs.subruns.raw_copy()
        N_scan = len(subruns)
        ws._sample_logs[HidraConstants.CHEMICAL_FORMULA] = ["Fe3O4"] * N_scan

        sample = _Sample.init_group([ws._sample_logs])

        assert "chemical_formula" in sample
        # _Sample takes the first value from the log
        assert sample["chemical_formula"] == "Fe3O4"

    def test_Sample_chemical_formula_absent(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
    ):
        """Verify chemical_formula defaults to 'unknown' when not in logs"""
        ws = minimal_HidraWorkspace(with_instrument=False)

        # Ensure chemical formula is not in logs
        if HidraConstants.CHEMICAL_FORMULA in ws._sample_logs:
            del ws._sample_logs[HidraConstants.CHEMICAL_FORMULA]

        sample = _Sample.init_group([ws._sample_logs])

        assert "chemical_formula" in sample
        assert sample["chemical_formula"] == "unknown"

    def test_Sample_temperature_present(self, minimal_HidraWorkspace: Callable[..., HidraWorkspace]):
        """Verify temperature field and units when TEMPERATURE log is present"""
        ws = minimal_HidraWorkspace(with_instrument=False)

        # Add temperature data to logs with units using tuple syntax
        subruns = ws._sample_logs.subruns.raw_copy()
        N_scan = len(subruns)
        temp_values = np.linspace(300, 400, N_scan)

        # Use tuple (key, units) to set value with units
        ws._sample_logs[(HidraConstants.TEMPERATURE, "K")] = temp_values

        sample = _Sample.init_group([ws._sample_logs])

        assert "temperature" in sample
        assert sample["temperature"].shape == (N_scan,)
        assert sample["temperature"].dtype == FIELD_DTYPE.FLOAT_DATA.value
        assert sample["temperature"].attrs["units"] == "K"

    def test_Sample_temperature_absent(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
    ):
        """Verify no temperature field when TEMPERATURE log is absent"""
        ws = minimal_HidraWorkspace(with_instrument=False)

        # Ensure temperature is not in logs
        if HidraConstants.TEMPERATURE in ws._sample_logs:
            del ws._sample_logs[HidraConstants.TEMPERATURE]

        sample = _Sample.init_group([ws._sample_logs])

        assert "temperature" not in sample

    def test_Sample_stress_field_present(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
    ):
        """Verify stress_field field, shape, and direction attr when present"""
        ws = minimal_HidraWorkspace(with_instrument=False)

        # Add stress field data to logs
        subruns = ws._sample_logs.subruns.raw_copy()
        N_scan = len(subruns)
        # Local, freshly-seeded generator -- not the implicit global numpy RNG state, which is
        # shared process-wide and would couple this test's values to unrelated tests' draws.
        stress_values = np.random.default_rng(seed=0).standard_normal((N_scan, 3))

        ws._sample_logs[HidraConstants.STRESS_FIELD] = stress_values
        # Direction is stored as array with same value for each subrun
        ws._sample_logs[HidraConstants.STRESS_FIELD_DIRECTION] = np.array(["z"] * N_scan)

        sample = _Sample.init_group([ws._sample_logs])

        assert "stress_field" in sample
        assert sample["stress_field"].shape[0] == N_scan
        assert sample["stress_field"].dtype == FIELD_DTYPE.FLOAT_DATA.value
        # The direction is entry-wide and is written as a SCALAR. This assertion used
        # to accommodate a list, because the writer stored the whole per-scan-point
        # array in the attribute -- see `TestStressFieldDirectionIsEntryWide`.
        assert sample["stress_field"].attrs["direction"] == "z"

    def test_Sample_stress_field_shape_mismatch(self, minimal_HidraWorkspace: Callable[..., HidraWorkspace]):
        """Verify RuntimeError when stress_field first axis != N_scan"""
        ws = minimal_HidraWorkspace(with_instrument=False)

        # Add stress field with wrong shape
        subruns = ws._sample_logs.subruns.raw_copy()
        N_scan = len(subruns)
        # Local, freshly-seeded generator -- see test_Sample_stress_field_present for why.
        wrong_shape_stress = np.random.default_rng(seed=0).standard_normal((N_scan + 5, 3))  # Wrong first dimension

        # Set `_data` dict directly, otherwise `SampleLogs.__setitem__` itself will raise an exception.
        ws._sample_logs._data[HidraConstants.STRESS_FIELD] = wrong_shape_stress

        with pytest.raises(RuntimeError, match=r".*unexpected shape.*"):
            _Sample.init_group([ws._sample_logs])

    def test_Sample_coordinate_shape_mismatch(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
    ):
        """Verify RuntimeError when coordinate array axis != N_scan"""
        ws = minimal_HidraWorkspace(with_instrument=False)

        # Corrupt vx to have wrong size by directly manipulating the logs
        subruns = ws._sample_logs.subruns.raw_copy()
        N_scan = len(subruns)

        # Create bad coordinate data with wrong length
        # Set `_data` dict directly, otherwise `SampleLogs.__setitem__` itself will raise an exception.
        ws._sample_logs._data["vx"] = np.zeros(N_scan + 5)  # Wrong size
        ws._sample_logs._data["vy"] = np.zeros(N_scan + 5)
        ws._sample_logs._data["vz"] = np.zeros(N_scan + 5)

        with pytest.raises(RuntimeError, match=r".*unexpected shape.*"):
            _Sample.init_group([ws._sample_logs])

    def test_Sample_extra_logs(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
    ):
        """Verify logs not in NXstress_logs go to logs NXcollection with local_name"""
        ws = minimal_HidraWorkspace(with_instrument=False)

        # Add a custom log with ':' in the name and units using tuple syntax
        custom_log_name = "HB2B:CS:CustomValue"
        subruns = ws._sample_logs.subruns.raw_copy()
        N_scan = len(subruns)
        custom_log_value = np.full(N_scan, 42.0)
        ws._sample_logs[(custom_log_name, "mm")] = custom_log_value

        sample = _Sample.init_group([ws._sample_logs])

        assert "logs" in sample
        assert isinstance(sample["logs"], NXcollection)

        # ':' is escaped as '__3A'. The literal is spelled out rather than computed
        # from allowed_identifier, so this test fails if the encoding changes rather
        # than silently tracking it. (It previously expected 'HB2B_CS_CustomValue',
        # from a conversion that collided distinct log names onto one.)
        expected_field_name = "HB2B__3ACS__3ACustomValue"
        assert expected_field_name in sample["logs"]

        # Verify attributes: local_name still recovers the original PV name
        assert sample["logs"][expected_field_name].attrs["local_name"] == custom_log_name
        assert sample["logs"][expected_field_name].attrs["units"] == "mm"


class TestSampleTransformationChain:
    """`SAMPLE_DESCRIPTION` defines the frame its positions are expressed in.

    NXstress defines `peaks/sx,sy,sz` as the sample position *in the sample
    reference frame*, and defines that frame by the sample transformations.
    PyRS wrote no such group, so the positions were numerically present and
    semantically unanchored -- fine for a PyRS-to-PyRS round trip, which shares
    the convention implicitly, and uninterpretable for the external consumer the
    format exists to serve. Recorded as `open-questions/04` Q5.

    Two things are pinned here. That the chain is **traversable** -- it must
    never repeat the detector-side defect of Follow-up 7 F7.5, where
    `depends_on` named the *first* link and traversal stopped immediately. And
    that the **rotation axes are the ones decomposition recovered**, since those
    were derived from `rotate_project_q` rather than read off a document: a
    later edit to a `vector` is a change to the recovered convention and has to
    be deliberate.
    """

    #: Logs the rotation links are built from, with values that are distinct
    #: per axis and per scan point, so a test cannot pass by cross-wiring them.
    ROTATION_LOGS = {"omega": 10.0, "chi": 20.0, "phi": 30.0}

    @classmethod
    def _with_rotations(cls, ws):
        """Add `omega`/`chi`/`phi` logs, which the minimal fixture does not carry."""
        logs = ws._sample_logs
        n_scan = len(logs.subruns)
        for key, base in cls.ROTATION_LOGS.items():
            logs[key] = base + np.arange(n_scan, dtype=float)
        return logs

    @staticmethod
    def _traverse(sd) -> list[str]:
        """Walk the chain the way a NeXus reader does: entry point, then back to '.'."""
        reached: list[str] = []
        current = str(sd["depends_on"].nxvalue)
        while current != ".":
            assert current.startswith("./transformations/"), f"depends_on leaves the group: {current!r}"
            name = current.rsplit("/", 1)[-1]
            assert name not in reached, f"cycle at {name!r}"
            reached.append(name)
            current = str(sd["transformations"][name].attrs["depends_on"])
        return reached

    def test_every_link_is_reachable_from_the_entry_point(self, minimal_HidraWorkspace: Callable[..., HidraWorkspace]):
        """The F7.5 defect, asserted on the sample side before it can happen."""
        # Arrange
        logs = self._with_rotations(minimal_HidraWorkspace(with_instrument=False))
        sd = _Sample.init_group([logs])

        # Act
        reached = self._traverse(sd)

        # Assert -- every written link, not merely the one `depends_on` names
        assert set(reached) == set(sd["transformations"])
        assert set(reached) == {name for name, *_ in _Sample.TRANSFORMATION_CHAIN}

    def test_the_entry_point_is_the_last_link_not_the_first(
        self, minimal_HidraWorkspace: Callable[..., HidraWorkspace]
    ):
        """Naming the first link leaves a chain that terminates at once."""
        # Arrange
        logs = self._with_rotations(minimal_HidraWorkspace(with_instrument=False))
        sd = _Sample.init_group([logs])
        first, last = _Sample.TRANSFORMATION_CHAIN[0][0], _Sample.TRANSFORMATION_CHAIN[-1][0]

        # Assert
        assert str(sd["depends_on"].nxvalue) == f"./transformations/{last}"
        assert str(sd["transformations"][first].attrs["depends_on"]) == "."

    def test_traversal_order_reverses_write_order(self, minimal_HidraWorkspace: Callable[..., HidraWorkspace]):
        """Write order is composition order, so traversal must run against it.

        Write order `[A, B, C]` entered at `C` composes to `A @ B @ C`. If this
        reverses, the chain still traverses and still reaches every link -- and
        describes a different placement.
        """
        # Arrange
        logs = self._with_rotations(minimal_HidraWorkspace(with_instrument=False))
        sd = _Sample.init_group([logs])

        # Act
        reached = self._traverse(sd)

        # Assert
        assert reached == [name for name, *_ in reversed(_Sample.TRANSFORMATION_CHAIN)]

    def test_each_link_carries_its_source_values_and_provenance(
        self, minimal_HidraWorkspace: Callable[..., HidraWorkspace]
    ):
        # Arrange
        ws = minimal_HidraWorkspace(with_instrument=False)
        logs = self._with_rotations(ws)
        sd = _Sample.init_group([logs])

        # Assert
        n_scan = len(logs.subruns)
        for name, source, vector, transformation_type, units in _Sample.TRANSFORMATION_CHAIN:
            link = sd["transformations"][name]
            if source is None:
                expected = np.full(n_scan, _Sample.FRAME_OFFSET_DEGREES)
            elif source in HidraConstants.SAMPLE_COORDINATE_NAMES:
                expected = sd[source].nxdata
            else:
                expected = logs[source]
            np.testing.assert_allclose(link.nxdata, np.asarray(expected, dtype=float))
            assert tuple(link.attrs["vector"]) == vector, name
            assert link.attrs["transformation_type"] == transformation_type, name
            assert link.attrs["units"] == units, name
            # The NeXus name is `translation_x`; the PyRS value's name is `vx`.
            # `frame_offset` is a constant, so it has no log to point at.
            assert link.attrs.get("local_name") == source, name

    def test_the_rotation_axes_are_the_decomposed_ones(self):
        """The recovered convention, pinned so a later edit has to be deliberate.

        `PoleFigureCalculator.rotate_project_q` composes these angles as a
        closed-form projection rather than as matrices, so the axes appear
        nowhere in the source. They were recovered by driving that function
        through its public signature and identifying the matrix it implements;
        `plans/NXstress-prod/probes/a5_sample_rotation_axes.py` is the evidence
        and `open-questions/04` Q5 carries its output. The result is exact --

            sample -> lab == M0 @ R_omega(0,-1,0) @ R_chi(1,0,0) @ R_phi(0,0,-1)

        -- so unlike most conventions recovered after the fact there is nothing
        here left to assume.
        """
        # Arrange
        chain = {name: (vector, kind) for name, _s, vector, kind, _u in _Sample.TRANSFORMATION_CHAIN}

        # Assert
        assert chain["omega"] == ((0.0, -1.0, 0.0), "rotation")
        assert chain["chi"] == ((1.0, 0.0, 0.0), "rotation")
        assert chain["phi"] == ((0.0, 0.0, -1.0), "rotation")
        # M0: the projection is `diag(1, -1, -1)` at zero stage angles, not the
        # identity. Omitting it would assert the sample frame is the lab frame.
        assert chain["frame_offset"] == ((1.0, 0.0, 0.0), "rotation")
        assert _Sample.FRAME_OFFSET_DEGREES == 180.0

    def test_the_rotations_compose_in_write_order(self, minimal_HidraWorkspace: Callable[..., HidraWorkspace]):
        """The written chain reproduces the matrix `rotate_project_q` implements.

        The axes alone do not pin the convention -- the same three axes in a
        different order are a different rotation. This composes what was
        actually written and compares it against the referent, so an edit to
        either the order or an axis is caught by the thing it would break.
        """
        # Arrange -- one scan point, at angles with no symmetry to hide behind
        ws = minimal_HidraWorkspace(with_instrument=False, sub_runs=np.array([1]))
        logs = ws._sample_logs
        omega, chi, phi = 37.0, -52.0, 113.0
        for key, value in (("omega", omega), ("chi", chi), ("phi", phi)):
            logs[key] = np.array([value])
        sd = _Sample.init_group([logs])

        # Act -- compose the rotation links in write order, which is composition order
        composed = np.eye(3)
        for name, _source, _vector, kind, _units in _Sample.TRANSFORMATION_CHAIN:
            if kind != "rotation":
                continue
            link = sd["transformations"][name]
            composed = composed @ _rodrigues(link.attrs["vector"], float(link.nxdata[0]))

        # Assert -- against the projection itself, sample -> lab being its transpose
        expected = _lab_from_sample(omega, chi, phi)
        np.testing.assert_allclose(composed, expected, atol=1e-9)

    def test_rotations_are_omitted_when_their_logs_are_absent(
        self, minimal_HidraWorkspace: Callable[..., HidraWorkspace]
    ):
        """A synthetic workspace carries no stage angles; the chain is still valid."""
        # Arrange
        sd = _Sample.init_group([minimal_HidraWorkspace(with_instrument=False)._sample_logs])

        # Assert -- translations only, and the entry point follows
        assert set(sd["transformations"]) == {"translation_x", "translation_y", "translation_z"}
        assert str(sd["depends_on"].nxvalue) == "./transformations/translation_z"
        assert self._traverse(sd) == ["translation_z", "translation_y", "translation_x"]

    def test_a_rotation_in_some_inputs_only_raises(self, minimal_HidraWorkspace: Callable[..., HidraWorkspace]):
        """Part-filling it would misalign every scan point after the first gap."""
        # Arrange
        first = minimal_HidraWorkspace(with_instrument=False, sub_runs=np.array([1, 2, 3]))
        second = minimal_HidraWorkspace(with_instrument=False, sub_runs=np.array([4, 5, 6]))
        self._with_rotations(first)

        # Act / Assert
        with pytest.raises(RuntimeError, match=r"log 'omega' is present in some input workspaces"):
            _Sample.init_group([first._sample_logs, second._sample_logs])

    def test_the_chain_grows_on_append(self, minimal_HidraWorkspace: Callable[..., HidraWorkspace]):
        """A chain shorter than the axes it describes would silently misdescribe them."""
        # Arrange
        first = minimal_HidraWorkspace(with_instrument=False, sub_runs=np.array([1, 2, 3]))
        second = minimal_HidraWorkspace(with_instrument=False, sub_runs=np.array([4, 5]))
        self._with_rotations(first)
        self._with_rotations(second)
        sd = _Sample.init_group([first._sample_logs])
        before = np.asarray(sd["transformations"]["omega"].nxdata).copy()
        entry_point = str(sd["depends_on"].nxvalue)

        # Act
        _Sample.init_group([second._sample_logs], data=sd)

        # Assert
        for name, source, *_ in _Sample.TRANSFORMATION_CHAIN:
            link = sd["transformations"][name]
            assert link.shape[0] == 5, name
            if source in HidraConstants.SAMPLE_COORDINATE_NAMES:
                np.testing.assert_allclose(link.nxdata, sd[source].nxdata)
        # Existing rows untouched, and an append adds scan points to a frame
        # rather than redefining one.
        np.testing.assert_allclose(np.asarray(sd["transformations"]["omega"].nxdata)[:3], before)
        assert str(sd["depends_on"].nxvalue) == entry_point

    def test_append_that_would_change_the_chain_raises(self, minimal_HidraWorkspace: Callable[..., HidraWorkspace]):
        """A link gained or lost part-way means the frame changes definition mid-entry."""
        # Arrange
        first = minimal_HidraWorkspace(with_instrument=False, sub_runs=np.array([1, 2, 3]))
        second = minimal_HidraWorkspace(with_instrument=False, sub_runs=np.array([4, 5]))
        self._with_rotations(first)
        sd = _Sample.init_group([first._sample_logs])

        # Act / Assert
        with pytest.raises(RuntimeError, match=r"transformation chain has rotation links"):
            _Sample.init_group([second._sample_logs], data=sd)

    def test_only_some_of_the_three_angles_raises(self, minimal_HidraWorkspace: Callable[..., HidraWorkspace]):
        """Two of the three angles describe no frame, so a partial set is refused."""
        # Arrange
        ws = minimal_HidraWorkspace(with_instrument=False)
        logs = ws._sample_logs
        logs["omega"] = np.zeros(len(logs.subruns))
        logs["chi"] = np.zeros(len(logs.subruns))

        # Act / Assert
        with pytest.raises(RuntimeError, match=r"needs all of \['omega', 'chi', 'phi'\]"):
            _Sample.init_group([logs])


class TestNonFiniteCoordinatesAreAsked:
    """Non-finite sample positions are detected by asking, not by catching.

    `PointList` enforces finiteness with a bare `assert`, so the only way to
    detect it through `get_pointlist` was to catch `AssertionError` and match
    its **message text** -- `"some coordinates do not have finite values"`.
    A reworded upstream assertion would silently stop matching and turn a
    handled case into an unhandled failure part-way through a write.

    A *missing* coordinate log is a different thing and still raises
    `ValueError`, naming the log: NaN-filling it would hide a workspace with no
    positions at all.
    """

    def test_non_finite_coordinates_are_written_as_nan(self, minimal_HidraWorkspace: Callable[..., HidraWorkspace]):
        # Arrange
        ws = minimal_HidraWorkspace(with_instrument=False)
        logs = ws._sample_logs
        n_scan = len(logs.subruns)
        logs["vy"] = np.full(n_scan, np.nan)

        # Act
        sample = _Sample.init_group([logs])

        # Assert -- every axis NaN-filled, and the group is still well formed
        for axis in HidraConstants.SAMPLE_COORDINATE_NAMES:
            assert np.all(np.isnan(np.asarray(sample[axis].nxdata)))
            assert sample[axis].shape == (n_scan,)

    def test_finite_coordinates_are_unchanged(self, minimal_HidraWorkspace: Callable[..., HidraWorkspace]):
        # Arrange
        ws = minimal_HidraWorkspace(with_instrument=False)
        logs = ws._sample_logs

        # Act
        sample = _Sample.init_group([logs])

        # Assert
        assert np.all(np.isfinite(np.asarray(sample["vx"].nxdata)))

    def test_a_missing_coordinate_log_still_raises(self, minimal_HidraWorkspace: Callable[..., HidraWorkspace]):
        """Not NaN-filled: a workspace with no positions at all must be reported."""
        # Arrange
        ws = minimal_HidraWorkspace(with_instrument=False)
        logs = ws._sample_logs
        del logs._data["vz"]

        # Act / Assert
        with pytest.raises(ValueError, match=r"Failed to find positions in logs.*vz"):
            _Sample.init_group([logs])

    def test_the_check_does_not_depend_on_the_assertion_message(
        self, minimal_HidraWorkspace: Callable[..., HidraWorkspace]
    ):
        """The point of the change: the NaN path must survive a reworded assertion."""
        # Arrange
        ws = minimal_HidraWorkspace(with_instrument=False)
        logs = ws._sample_logs
        logs["vx"] = np.full(len(logs.subruns), np.inf)

        # Act -- `get_pointlist` is never reached, so its message is irrelevant
        sample = _Sample.init_group([logs])

        # Assert
        assert np.all(np.isnan(np.asarray(sample["vx"].nxdata)))


class TestStressFieldDirectionIsEntryWide:
    """The `stress_field` direction is one value per entry, not one per scan point.

    It is *held* as a per-scan-point sample log, like `SampleName` and
    `chemical_formula`, but the NXstress schema defines it as a scalar attribute
    on `stress_field`. The writer used to store `logss[0][key]` -- the whole
    array -- into that attribute, and `sampleLogsFromNexus` then broadcast it
    again, so a 3-point direction round-tripped from shape `(3,)` to `(3, 3)`
    **without raising**: `SampleLogs.__setitem__` only checks the first axis, and
    `len((3, 3)) == 3`.

    The same line also skipped the cross-input agreement check that `name` and
    `chemical_formula` get, so two workspaces could disagree and the first would
    silently win.
    """

    DIRECTION = HidraConstants.STRESS_FIELD_DIRECTION

    @staticmethod
    def _with_stress(ws, direction, *, n_axes: int = 3):
        subruns = ws._sample_logs.subruns.raw_copy()
        rng = np.random.default_rng(seed=0)
        ws._sample_logs[HidraConstants.STRESS_FIELD] = rng.standard_normal((len(subruns), n_axes))
        ws._sample_logs[HidraConstants.STRESS_FIELD_DIRECTION] = np.asarray(direction)
        return ws._sample_logs

    def test_the_attribute_is_a_scalar(self, minimal_HidraWorkspace: Callable[..., HidraWorkspace]):
        # Arrange
        ws = minimal_HidraWorkspace(with_instrument=False)
        n_scan = len(ws._sample_logs.subruns)
        logs = self._with_stress(ws, ["z"] * n_scan)

        # Act
        sample = _Sample.init_group([logs])

        # Assert
        assert sample["stress_field"].attrs["direction"] == "z"
        assert np.asarray(sample["stress_field"].attrs["direction"]).shape == ()

    def test_it_round_trips_one_dimensional(self, minimal_HidraWorkspace: Callable[..., HidraWorkspace]):
        """The defect: shape (3,) in, shape (3, 3) out, and nothing raised."""
        # Arrange
        ws = minimal_HidraWorkspace(with_instrument=False)
        n_scan = len(ws._sample_logs.subruns)
        sample = _Sample.init_group([self._with_stress(ws, ["z"] * n_scan)])

        # Act
        recovered = _Sample.sampleLogsFromNexus(sample)[self.DIRECTION]

        # Assert
        assert np.asarray(recovered).shape == (n_scan,)
        assert list(np.asarray(recovered)) == ["z"] * n_scan

    def test_inputs_disagreeing_on_direction_raises(self, minimal_HidraWorkspace: Callable[..., HidraWorkspace]):
        """`name` and `chemical_formula` already raised here; direction did not."""
        # Arrange
        first = minimal_HidraWorkspace(with_instrument=False, sub_runs=np.array([1, 2, 3]))
        second = minimal_HidraWorkspace(with_instrument=False, sub_runs=np.array([4, 5, 6]))
        logss = [self._with_stress(first, ["z"] * 3), self._with_stress(second, ["x"] * 3)]

        # Act / Assert
        with pytest.raises(RuntimeError, match=r"disagree on 'stress field direction'"):
            _Sample.init_group(logss)

    def test_a_direction_varying_within_one_input_raises(self, minimal_HidraWorkspace: Callable[..., HidraWorkspace]):
        """An entry-wide value that varies per scan point is as wrong as two inputs disagreeing."""
        # Arrange
        ws = minimal_HidraWorkspace(with_instrument=False, sub_runs=np.array([1, 2, 3]))
        logs = self._with_stress(ws, ["x", "y", "z"])

        # Act / Assert
        with pytest.raises(RuntimeError, match=r"disagree on 'stress field direction'"):
            _Sample.init_group([logs])


class TestEntryWideScalarsAreNormalised:
    """`_scalar` compares normalised values, so it must return a normalised one."""

    def test_a_bytes_valued_log_is_written_as_text(self, minimal_HidraWorkspace: Callable[..., HidraWorkspace]):
        # Arrange -- bytes reach `_scalar` directly, bypassing `SampleLogs`' own
        # normalisation, exactly as a value read straight off an `NXfield` would.
        ws = minimal_HidraWorkspace(with_instrument=False)
        logs = ws._sample_logs
        logs._data[HidraConstants.SAMPLE_NAME] = np.array([b"steel"] * len(logs.subruns))

        # Act
        sample = _Sample.init_group([logs])

        # Assert
        assert sample["name"].nxdata == "steel"


class TestSampleTailAppend:
    """`init_group(logss, data=...)` grows an existing SAMPLE_DESCRIPTION group.

    Per-family coverage, independent of the end-to-end round trip in
    `test_append.py`.
    """

    def test_append_grows_every_per_scan_point_field_by_the_same_count(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
    ):
        # Arrange
        first = minimal_HidraWorkspace(with_instrument=False, sub_runs=np.array([1, 2, 3]))
        second = minimal_HidraWorkspace(with_instrument=False, sub_runs=np.array([4, 5]))
        sd = _Sample.init_group([first._sample_logs])
        before = np.asarray(sd["vx"].nxdata).copy()

        # Act
        returned = _Sample.init_group([second._sample_logs], data=sd)

        # Assert
        assert returned is sd
        assert sd["scan_point"].nxdata.tolist() == [1, 2, 3, 4, 5]
        for axis in ("vx", "vy", "vz"):
            assert sd[axis].shape[0] == 5, axis
        for name in sd["logs"]:
            assert sd["logs"][name].shape[0] == 5, name
        assert np.array_equal(np.asarray(sd["vx"].nxdata)[:3], before)

    def test_append_with_a_different_retained_log_set_raises(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
    ):
        """Existing rows would have no value for a log the appended batch introduces."""
        # Arrange
        first = minimal_HidraWorkspace(with_instrument=False, sub_runs=np.array([1, 2, 3]))
        second = minimal_HidraWorkspace(with_instrument=False, sub_runs=np.array([4, 5]))
        second.set_sample_log("an_extra_log", np.array([4, 5]), np.zeros(2))
        sd = _Sample.init_group([first._sample_logs])

        # Act / Assert
        with pytest.raises(RuntimeError, match=r"retained sample logs do not match"):
            _Sample.init_group([second._sample_logs], data=sd)

    def test_append_of_a_longer_string_log_is_not_truncated(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        tmp_path: Path,
    ):
        """A fixed-width `|S` column would silently truncate; `_writable` prevents it.

        `Filename` is a bytes log, and a fixed-width HDF5 string column is sized by
        the longest value present when it is created. Appending a longer one raises
        nothing and lands a prefix -- which is why `_writable` coerces `|S` to the
        variable-length dtype. Round-tripped through a real file, because the
        truncation only happens on the h5py write path.
        """
        # Arrange
        first = minimal_HidraWorkspace(with_instrument=False, sub_runs=np.array([1, 2, 3]), name="s")
        longer = "a_considerably_longer_filename"
        second = minimal_HidraWorkspace(with_instrument=False, sub_runs=np.array([4, 5]), name=longer)
        sd = _Sample.init_group([first._sample_logs])
        _Sample.init_group([second._sample_logs], data=sd)

        path = tmp_path / "logs.nxs"
        with nxopen(path, "w") as root:
            root["entry"] = NXentry()
            root["entry"]["sample"] = sd

        # Act
        with nxopen(path, "r") as root:
            values = [
                v.decode("utf-8") if isinstance(v, bytes) else str(v)
                for v in root["entry"]["sample"]["logs"]["Filename"].nxdata
            ]

        # Assert
        assert values[-1] == f"{longer}.h5"
