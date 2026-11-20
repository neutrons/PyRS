# ruff: noqa: F841
"""
Tests for pyrs/utilities/NXstress/_instrument.py
"""

from collections.abc import Callable
import logging

import numpy as np
from nexusformat.nexus import NXcollection, NXinstrument, NXdetector_module, NXfield, nxload
import pytest

from pyrs.core.detector_model import DetectorModel, rotation_matrix
from pyrs.core.instrument_geometry import DENEXDetectorShift
from pyrs.core.workspaces import HidraWorkspace
from pyrs.utilities.NXstress._instrument import _Instrument, _Masks
from pyrs.utilities.NXstress._definitions import DEFAULT_TAG
from pyrs.utilities.NXstress._transformations import resolve_depends_on
from pyrs.utilities.NXstress.NXstress import NXstress
from tests.util.mask_helpers import add_named_detector_mask


class TestInstrument:
    """Test suite for _instrument.py"""

    def test_Masks_init(self):
        """Verify _Masks._init creates empty NXcollection with required fields"""
        masks = _Masks._init()

        assert isinstance(masks, NXcollection)
        assert "names" in masks
        assert "detector" in masks
        assert "solid_angle" in masks

        # Verify empty structure
        assert len(masks["names"]) == 0
        assert isinstance(masks["detector"], NXcollection)
        assert isinstance(masks["solid_angle"], NXcollection)

    def test_Masks_init_group_with_default_mask(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
    ):
        """Verify default mask appears in masks with DEFAULT_TAG name"""
        ws = minimal_HidraWorkspace(sub_runs=[1, 2, 3], with_instrument=True, with_masks=True)

        masks = _Masks.init_group(ws)

        assert isinstance(masks, NXcollection)
        assert DEFAULT_TAG in masks["names"]
        assert DEFAULT_TAG in masks["detector"]

    def test_Masks_init_group_append(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
    ):
        """Verify calling init_group twice (detector then solid_angle) populates both"""
        ws = minimal_HidraWorkspace(sub_runs=[1, 2, 3], with_instrument=True, with_masks=True)

        # First call for detector masks
        masks = _Masks.init_group(ws)
        initial_count = len(masks["names"])

        # Second call for solid angle masks (appending)
        # For this test, we'll use the same workspace but we'll just change the names.
        defaults = ws._diff_data_set[None], ws._var_data_set[None], ws._mask_dict.get(None, None)
        ws._diff_data_set = {f"{k}_2nd": v for k, v in ws._diff_data_set.items() if k is not None}
        ws._var_data_set = {f"{k}_2nd": v for k, v in ws._var_data_set.items() if k is not None}
        ws._mask_dict = {f"{k}_2nd": v for k, v in ws._mask_dict.items() if k is not None}
        # Re-add the default items:
        ws._diff_data_set[None], ws._var_data_set[None] = defaults[0:2]
        if defaults[2]:
            ws._mask_dict[None] = defaults[2]

        # In real usage, solid angle masks would be different data
        masks = _Masks.init_group(ws, masks=masks)

        # Names should have been appended
        assert len(masks["names"]) >= initial_count

    def test_Masks_init_group_duplicate_raises(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
    ):
        """Verify behavior when attempting to add duplicate masks

        A non-default named mask is added to the workspace.  The first call to
        `init_group` writes it; the second call must raise because the same
        name is already present in the masks group.
        """
        ws = minimal_HidraWorkspace(sub_runs=[1, 2, 3], with_instrument=True)

        # Add a non-default named mask so that `mask_keys(ws)` contains a
        # name other than DEFAULT_TAG.  The first `init_group` call will write
        # it; the second call will find it already in `names` and raise.
        add_named_detector_mask(ws, "test_mask")

        masks = _Masks.init_group(ws)

        with pytest.raises(RuntimeError, match=r".*Usage error: mask .* has already been written.*"):
            masks2 = _Masks.init_group(ws, masks=masks)

    def test_Instrument_init(self):
        """Verify _Instrument._init creates NXinstrument with name and short_name"""
        inst = _Instrument._init("HB2B", "HB2B")

        assert isinstance(inst, NXinstrument)
        assert "name" in inst
        assert inst["name"] == "HB2B"
        assert inst["name"].attrs["short_name"] == "HB2B"

    def test_Instrument_detector_module_fields(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
    ):
        """Verify NXdetector_module contains required fields"""
        ws = minimal_HidraWorkspace(sub_runs=[1, 2, 3], with_instrument=True)

        inst = _Instrument.init_group([ws])

        assert "DETECTOR" in inst
        detector = inst["DETECTOR"]
        assert "detector_bank" in detector

        det_module = detector["detector_bank"]
        assert isinstance(det_module, NXdetector_module)

        # Verify required fields
        assert "data_size" in det_module
        assert "fast_pixel_direction" in det_module
        assert "slow_pixel_direction" in det_module
        assert "depends_on" in det_module

        # Verify data_size is 2D array [rows, cols]
        assert len(det_module["data_size"]) == 2
        assert det_module["data_size"].dtype == np.int64

    def test_Instrument_transformations_chain(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
    ):
        """The detector chain is the calibration file's (#1099): its names, in DetectorModel's order, innermost first"""
        # Arrange
        ws = minimal_HidraWorkspace(sub_runs=[1, 2, 3], with_instrument=True)

        # Act
        inst = _Instrument.init_group([ws])

        # Assert
        detector = inst["DETECTOR"]
        trans = detector["transformations"]
        # spelled out here rather than imported from the writer, so this fails if the writer changes
        chain = [
            "shift_x",
            "shift_y",
            "rotation_z",
            "rotation_y",
            "rotation_x",
            "arm_length",
            "shift_z",
            "two_theta_0",
            "arm_rotation",
        ]
        for name in chain:
            assert "transformation_type" in trans[name].attrs
            assert "vector" in trans[name].attrs

        # the detector enters at the innermost transformation; each names the next outward, the last "."
        assert detector["depends_on"] == "transformations/shift_x"
        links = [str(trans[name].attrs["depends_on"]) for name in chain]
        assert links == chain[1:] + ["."]
        # 2theta_0 swings about Y (the scattering plane), as reduction applies it
        np.testing.assert_array_equal(trans["two_theta_0"].attrs["vector"], [0.0, 1.0, 0.0])
        # each calibration parameter has its error; the arm length and angle are not calibration parameters
        assert {name for name in trans if name.endswith("_errors")} == {
            f"{name}_errors" for name in chain if name not in ("arm_length", "arm_rotation")
        }

    # The instrument identity comes from config, not from the workspace: the Hidra
    # project format records no instrument name anywhere. Overriding it is what lets
    # this writer serve a beamline other than HB2B without a code change.

    # These go through `config_override`, not `monkeypatch.setattr(module, "Config", {...})`.
    # A dict stub makes `Config[key]` an ordinary dict lookup, so the real
    # `Config.__getitem__` and the real deep-merge are never exercised; and a stub has
    # no "unset" state, which is why `test_Instrument_name_defaults_when_not_overridden`
    # below cannot be written against one at all. Together the three show the value
    # *tracks* configuration, which no one of them shows alone.
    #
    # Measured, so the claim is not larger than it is: this does **not** catch
    # `_instrument.py` reverting to the forbidden `from neutrons_standard.config import
    # Config`. Both names reach the same singleton once `init("pyrs")` has run, and
    # `config_override` mutates that instance, so these pass either way. That import is
    # a load-*order* hazard -- it raises `ModuleNotFoundError: No module named 'None'`
    # only when it runs before `init()` -- and is guarded by `config.py`'s docstring and
    # by import order, not by these tests.

    def test_Instrument_name_from_config(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        config_override: Callable[..., None],
    ):
        """Verify instrument name and short_name are taken from configuration"""
        # Arrange
        config_override({"nxstress": {"instrument_name": "HB2A", "instrument_short_name": "PG3"}})
        ws = minimal_HidraWorkspace(sub_runs=[1, 2, 3], with_instrument=True)

        # Act
        inst = _Instrument.init_group([ws])

        # Assert
        assert str(inst["name"].nxvalue) == "HB2A"
        assert inst["name"].attrs["short_name"] == "PG3"

    def test_Instrument_name_defaults_when_not_overridden(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
    ):
        """And the shipped default is what arrives when nothing overrides it.

        The companion to the test above: together they show the value *tracks*
        configuration, which neither shows alone. A dict stub could not express
        this case at all -- there is no "unset" in a stub, only whatever the stub
        was given.
        """
        # Act
        inst = _Instrument.init_group([minimal_HidraWorkspace(sub_runs=[1, 2, 3], with_instrument=True)])

        # Assert
        assert str(inst["name"].nxvalue) == "HB2B"

    def test_Instrument_name_blank_falls_back(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        config_override: Callable[..., None],
        caplog: pytest.LogCaptureFixture,
    ):
        """Verify a blank configured name falls back to HB2B, and says so.

        Note this is reachable only by bypassing `validate_config`, which rejects
        a blank `instrument_name` at import. `_instrument_names` keeps the
        fallback as a backstop for a config that could not be read at all, and
        this covers that path.
        """
        # Arrange
        # `validate=False`: as the docstring says, this state is unreachable once
        # `validate_config` has run -- the fixture would otherwise refuse it first.
        config_override(
            {"nxstress": {"instrument_name": "   ", "instrument_short_name": ""}},
            validate=False,
        )
        ws = minimal_HidraWorkspace(sub_runs=[1, 2, 3], with_instrument=True)

        # Act
        with caplog.at_level(logging.WARNING):
            inst = _Instrument.init_group([ws])

        # Assert
        assert str(inst["name"].nxvalue) == "HB2B"
        assert inst["name"].attrs["short_name"] == "HB2B"
        assert "unset or blank" in caplog.text

    def test_Instrument_all_transformations_reachable(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
    ):
        """Verify every transformation written is reached from the detector.

        A transformation the chain never reaches cannot affect the geometry, however
        faithfully it was written. The traversal rule is spelled out here rather than
        imported from the writer, so this fails if the writer changes rather than
        tracking it.
        """
        # Arrange
        ws = minimal_HidraWorkspace(sub_runs=[1, 2, 3], with_instrument=True)
        inst = _Instrument.init_group([ws])
        detector = inst["DETECTOR"]
        trans = detector["transformations"]
        written = {name for name in trans if "transformation_type" in trans[name].attrs}  # not the `_errors`

        # Act: follow depends_on from the detector until '.', resolving each value
        # by the NeXus rule -- it raises on a link that does not resolve.
        reached = [link.nxname for link in resolve_depends_on(detector)]

        # Assert
        assert set(reached) == written, f"unreachable: {sorted(written - set(reached))}"


def _written_order(detector) -> list:
    """The chain's links outermost first, read from the file rather than assumed.

    Traversal from the entry point goes innermost first (#1099: the detector enters at the transformation
    applied first to a pixel), so composing the reversed traversal left to right gives outer @ ... @ inner.
    """
    return [link.nxname for link in reversed(resolve_depends_on(detector))]


def _rodrigues(axis, degrees: float):
    """Right-handed rotation of `degrees` about `axis`.

    Written here rather than imported from the writer, so
    `test_rotation_subchain_composes_to_the_reduction_convention` fails when the
    writer changes rather than tracking it (`plans/audit-process/process.md` 5.3).
    """
    k = np.asarray(axis, dtype=float)
    k = k / np.linalg.norm(k)
    theta = np.radians(degrees)
    cross = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * cross + (1.0 - np.cos(theta)) * (cross @ cross)


class TestDetectorRotationOrder:
    """The written rotation order is the reduction's, and that is now asserted.

    `_instrument.py` carried `# TODO: check order of rotations here!!!` from the
    first implementation until this review. The order turned out to be correct,
    which `probes/a5_transformations_chain.py` established — but a probe is run
    during an audit, not on every commit, and the two existing chain tests pin
    *reachability* and *pairwise links*, neither of which would catch a reordering
    that kept the chain connected. Swapping `rotation_x` and `rotation_z` in the
    writer leaves both of them green.

    So the claim is asserted against its referent: the composed sub-chain equals
    `pyrs.core.detector_model.rotation_matrix`, the rotation the reduction
    actually uses.

    `two_theta_0` and the arm angle are not rotations of the detector about its own centre, so they are left
    out here; `TestDetectorChainIsTheCalibrationFiles` checks the whole chain against pixel positions (#1099).
    """

    ROTATIONS = ("rotation_x", "rotation_y", "rotation_z")

    def test_rotation_subchain_composes_to_the_reduction_convention(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
    ):
        # Arrange -- angles large and asymmetric, so no two orderings coincide
        ws = minimal_HidraWorkspace(sub_runs=[1, 2, 3], with_instrument=True)
        rot_x, rot_y, rot_z = 37.0, -52.0, 113.0
        ws.set_detector_shift(
            DENEXDetectorShift(
                shift_x=0.01,
                shift_y=0.02,
                shift_z=0.03,
                rotation_x=rot_x,
                rotation_y=rot_y,
                rotation_z=rot_z,
                tth_0=0.0,
            )
        )
        detector = _Instrument.init_group([ws])["DETECTOR"]
        trans = detector["transformations"]

        # Act -- compose in the order the file actually says, which means reading
        # the chain rather than a list declared here. Iterating a hardcoded tuple
        # would compose in *this test's* order whatever the writer emitted, and
        # would pass against a reversed writer -- measured, not assumed.
        composed = np.eye(3)
        for name in _written_order(detector):
            if name not in self.ROTATIONS:
                continue
            link = trans[name]
            composed = composed @ _rodrigues(link.attrs["vector"], float(link.nxdata))

        # Assert -- against the referent, not against a restatement of it
        expected = rotation_matrix(np.deg2rad(rot_x), np.deg2rad(rot_y), np.deg2rad(rot_z))
        # Both sides are float64: measured max elementwise difference 5.6e-17,
        # while the reversed order (Rz @ Ry @ Rx) differs by 1.59, so this
        # tolerance still separates the orderings by many orders of magnitude.
        np.testing.assert_allclose(composed, expected, atol=1e-12)

    def test_the_rotation_links_are_written_in_xyz_order(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
    ):
        """The order the composition test depends on, stated separately.

        If the writer stops emitting the three rotations consecutively, the
        composition above would silently start testing something else.
        """
        # Arrange
        ws = minimal_HidraWorkspace(sub_runs=[1, 2, 3], with_instrument=True)

        # Act
        written = _written_order(_Instrument.init_group([ws])["DETECTOR"])

        # Assert
        rotations = [n for n in written if n in self.ROTATIONS]
        assert rotations == list(self.ROTATIONS)
        start = written.index("rotation_x")
        assert written[start : start + 3] == list(self.ROTATIONS), written


class TestMaskGroupHoldsOnlyRealMasks:
    """`masks/names` lists detector masks, and every one of them has an array.

    It used to list *diffractogram keys* too -- `_Masks.mask_keys` unioned
    `_diff_data_set` in -- so a texture entry advertised `eta_-5.0` as a mask and
    resolved it to nothing. The `NXlink` meant to patch that was never serialised,
    and had it worked it would have returned the default array under each composite
    key, inflating `_mask_dict` on read with entries the original workspace never
    had.
    """

    @staticmethod
    def _texture(ws):
        """Eta-only reductions: every one is the default mask ∩ an eta ROI."""
        sub_runs = ws.get_sub_runs().raw_copy()
        two_theta = np.tile(np.linspace(60.0, 120.0, 20), (len(sub_runs), 1))
        ones = np.ones((len(sub_runs), 20))
        keys = ("eta_-5.0", "eta_0.0", "eta_5.0")
        ws.set_reduced_diffraction_data_set(two_theta, {k: ones for k in keys}, {k: ones.copy() for k in keys})
        return keys

    def test_diffractogram_keys_are_not_listed_as_masks(self, minimal_HidraWorkspace: Callable[..., HidraWorkspace]):
        # Arrange
        ws = minimal_HidraWorkspace(sub_runs=[1, 2, 3], with_instrument=True, with_masks=True)
        self._texture(ws)

        # Act
        masks = _Masks.init_group(ws)
        # `nxvalue` returns a bare `str` for a length-1 array, not a 1-element list.
        names = [str(n) for n in np.atleast_1d(masks["names"].nxvalue)]

        # Assert
        assert names == [DEFAULT_TAG]
        assert not any("eta" in name for name in names)

    def test_every_listed_mask_resolves_to_an_array(self, minimal_HidraWorkspace: Callable[..., HidraWorkspace]):
        """The property the old `NXlink` was trying and failing to provide."""
        # Arrange
        ws = minimal_HidraWorkspace(sub_runs=[1, 2, 3], with_instrument=True, with_masks=True, mask_names=("mask_a",))
        self._texture(ws)

        # Act
        masks = _Masks.init_group(ws)
        names = [str(n) for n in np.atleast_1d(masks["names"].nxvalue)]

        # Assert
        assert set(names) == {DEFAULT_TAG, "mask_a"}
        for name in names:
            assert name in masks["detector"] or name in masks["solid_angle"], name

    def test_a_solid_angle_shaped_mask_raises_rather_than_being_guessed(
        self, minimal_HidraWorkspace: Callable[..., HidraWorkspace]
    ):
        """Nothing loads one yet, so writing one is refused instead of approximated.

        The old heuristic filed a mask under `solid_angle/` whenever it was 1-D,
        even-length and floating -- which is every square detector's mask if it is
        ever stored as float.
        """
        # Arrange
        ws = minimal_HidraWorkspace(sub_runs=[1, 2, 3], with_instrument=True, with_masks=True)
        ws.set_detector_mask(np.array([-5.0, 5.0]), False, "roi")

        # Act / Assert
        with pytest.raises(NotImplementedError, match=r"[Ss]olid-angle masks are not implemented"):
            _Masks.init_group(ws)

    def test_a_generated_default_mask_can_be_read_back(
        self,
        minimal_HidraWorkspace: Callable[..., HidraWorkspace],
        minimal_PeakCollection,
        tmp_path,
    ):
        """A workspace with no mask of its own wrote a file PyRS could not read.

        `_generate_default_mask` returned `np.ones(detector_size)` -- 2-D
        `(nrows, ncols)` -- while `set_detector_mask` refuses any 2-D array whose
        second axis is not 1. Every round-trip test used `with_masks=True`, so the
        generated-default path was never exercised.
        """
        # Arrange
        sub_runs = np.array([1, 2, 3])
        ws = minimal_HidraWorkspace(with_instrument=True, with_masks=False, sub_runs=sub_runs)
        peaks = [minimal_PeakCollection(N_subrun=3, sub_runs=sub_runs)]
        path = tmp_path / "generated_default.nxs"

        # Act
        with NXstress(path, "w") as nx:
            nx.write([ws], [peaks])
        with NXstress(path, "r") as nx:
            workspaces, _ = nx.read()

        # Assert
        recovered = np.asarray(workspaces[0].get_detector_mask(True))
        assert recovered.ndim == 1
        nrows, ncols = ws.get_instrument_setup().detector_size
        assert recovered.shape == (nrows * ncols,)


def _chain_matrix(transformations, entry: str, arm: float) -> np.ndarray:
    """The 4x4 transform of a whole chain, built from the fields' attributes alone (innermost first)."""
    total = np.eye(4)
    name = entry
    while name != ".":
        link = transformations[name]
        value = arm if name == "arm_rotation" else float(link.nxdata)
        vector = np.asarray(link.attrs["vector"], dtype=float)
        step = np.eye(4)
        if str(link.attrs["transformation_type"]) == "translation":
            step[:3, 3] = value * vector
        else:
            step[:3, :3] = _rodrigues(vector, value)
        total = step @ total
        name = str(link.attrs["depends_on"])
    return total


class TestDetectorChainIsTheCalibrationFiles:
    """NXstress describes the detector exactly as reduction places it (#1099): the calibration file's chain, the
    arm angle per scan point, the arm length reduction used, the applied calibration's errors."""

    SHIFT = DENEXDetectorShift(0.004, -0.003, 0.012, 0.3, -0.5, 1.2, 0.656)
    # seven distinct, non-default errors: an error written under another parameter's name shows
    ERRORS = DENEXDetectorShift(1.1e-4, 2.2e-4, 3.3e-4, 0.011, 0.022, 0.033, 0.0044)
    ERROR_BY_NAME = {
        "shift_x": 1.1e-4,
        "shift_y": 2.2e-4,
        "shift_z": 3.3e-4,
        "rotation_x": 0.011,
        "rotation_y": 0.022,
        "rotation_z": 0.033,
        "two_theta_0": 0.0044,
    }
    ARMS = (76.0, 90.0, 103.0)

    def _workspace(self, minimal_HidraWorkspace, l2=None):
        ws = minimal_HidraWorkspace(sub_runs=[1, 2, 3], with_instrument=True)
        ws.set_detector_shift(self.SHIFT)
        ws.set_sample_log("2theta", np.array([1, 2, 3]), np.array(self.ARMS))
        if l2 is not None:
            ws.set_sample_log("L2", np.array([1, 2, 3]), np.array(l2, dtype=float))
        return ws

    def test_chain_places_pixels_as_reduction_does(self, minimal_HidraWorkspace):
        """Applied without DetectorModel, the chain reproduces DetectorModel.lab_positions at every scan point"""
        # Arrange - an L2 log unlike the instrument's arm length, so the one reduction uses is checked
        ws = self._workspace(minimal_HidraWorkspace, l2=[1.95, 1.95, 1.95])
        detector = _Instrument.init_group([ws])["DETECTOR"]
        trans = detector["transformations"]
        model = DetectorModel(ws.get_instrument_setup())
        index_0, index_1 = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
        local = np.concatenate([model.local_positions(index_0, index_1), np.ones(index_0.shape + (1,))], axis=-1)
        entry = str(detector["depends_on"]).split("/")[-1]
        assert np.asarray(trans["arm_rotation"].nxdata).shape == (len(self.ARMS),)

        for scan_point, arm_log in enumerate(self.ARMS):
            # Act
            arm = float(trans["arm_rotation"].nxdata[scan_point])
            placed = (local @ _chain_matrix(trans, entry, arm).T)[..., :3]

            # Assert - the arm angle is -(the 2theta log), and the pixels are where reduction puts them
            assert arm == -arm_log
            expected = model.lab_positions(index_0, index_1, -arm_log, 1.95, self.SHIFT)
            np.testing.assert_allclose(placed, expected, rtol=0, atol=1e-12)

    def test_errors_are_the_recorded_calibrations(self, minimal_HidraWorkspace):
        """A workspace loaded from a project file carries the calibration reduction applied and its errors, each
        under its own parameter"""
        # Arrange - no shift of its own, the recorded calibration instead
        ws = minimal_HidraWorkspace(sub_runs=[1, 2, 3], with_instrument=True)
        ws.set_detector_shift(None)
        ws.set_recorded_calibration(self.SHIFT, self.ERRORS)

        # Act
        trans = _Instrument.init_group([ws])["DETECTOR"]["transformations"]

        # Assert
        assert bool(trans.attrs["calibrated"])
        assert float(trans["rotation_z"].nxdata) == pytest.approx(1.2)
        assert float(trans["two_theta_0"].nxdata) == pytest.approx(0.656)
        written = {name: float(trans[f"{name}_errors"].nxdata) for name in self.ERROR_BY_NAME}
        assert written == pytest.approx(self.ERROR_BY_NAME)

    def test_errors_unknown_are_not_refined(self, minimal_HidraWorkspace):
        """A shift with no recorded errors writes -1, not refined"""
        # Arrange
        ws = self._workspace(minimal_HidraWorkspace)

        # Act
        trans = _Instrument.init_group([ws])["DETECTOR"]["transformations"]

        # Assert
        assert {float(trans[name].nxdata) for name in trans if name.endswith("_errors")} == {-1.0}

    def test_own_shift_wins_over_another_record(self, minimal_HidraWorkspace):
        """A workspace's own shift is the one written; a recorded calibration that is not that shift lends it no
        errors"""
        # Arrange
        ws = self._workspace(minimal_HidraWorkspace)
        ws.set_recorded_calibration(DENEXDetectorShift(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1), self.ERRORS)

        # Act
        trans = _Instrument.init_group([ws])["DETECTOR"]["transformations"]

        # Assert
        assert float(trans["two_theta_0"].nxdata) == pytest.approx(0.656)
        assert {float(trans[name].nxdata) for name in trans if name.endswith("_errors")} == {-1.0}

    def test_record_of_the_own_shift_lends_its_errors(self, minimal_HidraWorkspace):
        """When the recorded calibration is the workspace's shift (NXstress read-back), its errors are written"""
        # Arrange
        ws = self._workspace(minimal_HidraWorkspace)
        ws.set_recorded_calibration(self.SHIFT, self.ERRORS)

        # Act
        trans = _Instrument.init_group([ws])["DETECTOR"]["transformations"]

        # Assert
        written = {name: float(trans[f"{name}_errors"].nxdata) for name in self.ERROR_BY_NAME}
        assert written == pytest.approx(self.ERROR_BY_NAME)

    def test_differing_l2_is_refused(self, minimal_HidraWorkspace):
        """One NXentry holds one detector placement; HB2B uses one distance"""
        # Arrange
        ws = self._workspace(minimal_HidraWorkspace, l2=[1.95, 1.95, 1.90])

        # Act / Assert
        with pytest.raises(RuntimeError, match="L2"):
            _Instrument.init_group([ws])

    def test_missing_arm_angle_is_nan(self, minimal_HidraWorkspace):
        """Without a 2theta log (synthetic data) the arm angle is unknown, not 0: one NaN per scan point"""
        # Arrange
        ws = minimal_HidraWorkspace(sub_runs=[1, 2, 3])

        # Act
        trans = _Instrument.init_group([ws])["DETECTOR"]["transformations"]

        # Assert
        arms = np.asarray(trans["arm_rotation"].nxdata)
        assert arms.shape == (3,)
        assert np.isnan(arms).all()

    @staticmethod
    def _as_old_layout(inst):
        """Rename an entry's chain to the layout NXstress wrote before #1099, as an old file has it"""
        trans = inst["DETECTOR"]["transformations"]
        old = {"shift_x": "translation_x", "shift_y": "translation_y", "shift_z": "translation_z"}
        old.update({"arm_length": "distance", "two_theta_0": "two_theta_zero"})
        for new, name in old.items():
            trans[name] = NXfield(float(trans[new].nxdata))
            del trans[new]
        for name in [name for name in trans if name.endswith("_errors")] + ["arm_rotation"]:
            del trans[name]
        return trans

    def test_old_layout_reads_back_the_same(self, minimal_HidraWorkspace):
        """An entry NXstress wrote before #1099 (translation_*, distance, two_theta_zero) reads as before, with no
        errors"""
        # Arrange
        inst = _Instrument.init_group([self._workspace(minimal_HidraWorkspace)])
        self._as_old_layout(inst)

        # Act
        geometry, shift, _ = _Instrument.instrumentFromNexus(inst)

        # Assert
        assert shift.convert_to_dict() == pytest.approx(self.SHIFT.convert_to_dict())
        assert geometry.arm_length == pytest.approx(2.0)
        assert _Instrument.shiftErrorFromNexus(inst) is None

    def test_new_names_win_over_old_ones(self, minimal_HidraWorkspace):
        """With both layouts' names present, the chain's (#1099) are read"""
        # Arrange
        inst = _Instrument.init_group([self._workspace(minimal_HidraWorkspace)])
        trans = inst["DETECTOR"]["transformations"]
        trans["translation_x"] = NXfield(9.0)
        trans["two_theta_zero"] = NXfield(9.0)
        trans["distance"] = NXfield(9.0)

        # Act
        geometry, shift, _ = _Instrument.instrumentFromNexus(inst)

        # Assert
        assert shift.center_shift_x == pytest.approx(0.004)
        assert shift.two_theta_0 == pytest.approx(0.656)
        assert geometry.arm_length == pytest.approx(2.0)


def test_errors_survive_a_read_back_and_rewrite(minimal_HidraWorkspace, minimal_PeakCollection, tmp_path):
    """Read back and written again (e.g. after refitting peaks), NXstress keeps the calibration's errors (#1099)"""
    # Arrange - a workspace carrying the recorded calibration and its errors, written once
    ws = minimal_HidraWorkspace(sub_runs=[1, 2, 3], with_instrument=True)
    ws.set_detector_shift(None)
    ws.set_recorded_calibration(
        TestDetectorChainIsTheCalibrationFiles.SHIFT, TestDetectorChainIsTheCalibrationFiles.ERRORS
    )
    peaks = [minimal_PeakCollection(N_subrun=3)]
    first, second = tmp_path / "first.nxs", tmp_path / "second.nxs"
    with NXstress(first, "w") as nx:
        nx.write([ws], [peaks])

    # Act - read back and written again
    with NXstress(first, "r") as nx:
        workspaces, peakss = nx.read()
    with NXstress(second, "w") as nx:
        nx.write(workspaces, peakss)
    read_errors = _Instrument.shiftErrorFromNexus(nxload(str(second), "r")["entry"]["instrument"])

    # Assert
    expected = TestDetectorChainIsTheCalibrationFiles.ERRORS.convert_to_dict()
    assert read_errors.convert_to_dict() == pytest.approx(expected)
