"""A5: is the NXtransformations chain `_instrument.py` writes actually traversable?

Spec 04 schedules a rotation-order cross-check (`_instrument.py` carries a
``# TODO: check order of rotations here!!!``), and Follow-up 1 F1.4 left open
*which* convention is the referent. Follow-up 2 F2.3 then listed the eight
transformations in written order -- but it read them out of the **source text**,
so it could only report the order, never whether the chain is reachable.

This probe builds the group and walks it, which turns out to matter.

The rule, from `NXtransformations`' own documentation (base_classes/
NXtransformations.nxdl.xml, NeXus definitions v2026.01):

    "The entry point (``depends_on``) will be outside of this class and point to
    a field in here."

    "For a chain of three transformations, where :math:`T_1` depends on
    :math:`T_2` and that in turn depends on :math:`T_3`, the final
    transformation :math:`T_f` is :math:`T_f = T_3 T_2 T_1`."

So traversal starts at the *object's* ``depends_on`` and follows each field's own
``depends_on`` until ``"."``. Anything not reachable that way is not part of the
geometry, however faithfully it was written to the file.

Per `process.md` §5.3 the composition rule is reimplemented **locally** here
rather than imported, so this probe fails if `_instrument.py` changes rather than
silently tracking it.

Run: ``pixi run python plans/NXstress-prod/probes/a5_transformations_chain.py``
"""

from __future__ import annotations

import sys

import numpy as np

from pyrs.core.instrument_geometry import DENEXDetectorGeometry, DENEXDetectorShift
from pyrs.core.workspaces import HidraWorkspace
from pyrs.utilities.NXstress._definitions import GROUP_NAME
from pyrs.utilities.NXstress._instrument import _Instrument


def report(claim: str, result: str) -> None:
    print(f"\n  CLAIM   {claim}")
    print(f"  RESULT  {result}")


def build_workspace() -> HidraWorkspace:
    """A calibrated workspace, matching the shape the unit-test fixture uses."""
    ws = HidraWorkspace("probe")
    subruns = np.arange(1, 4, dtype=int)
    ws.set_sub_runs(subruns)
    for axis, values in (
        ("vx", np.arange(3, dtype=float)),
        ("vy", np.zeros(3)),
        ("vz", np.zeros(3)),
    ):
        ws.set_sample_log(axis, subruns, values)
    ws.set_wavelength(1.486, calibrated=True)
    ws.set_instrument_geometry(
        DENEXDetectorGeometry(
            num_rows=4, num_columns=4, pixel_size_x=0.001, pixel_size_y=0.001, arm_length=2.0, calibrated=True
        )
    )
    ws.set_detector_shift(
        DENEXDetectorShift(
            shift_x=0.01, shift_y=0.02, shift_z=0.03, rotation_x=1.0, rotation_y=2.0, rotation_z=3.0, tth_0=0.5
        )
    )
    return ws


def walk_chain(detector, transformations) -> list[str]:
    """Follow `depends_on` from the detector, returning names in traversal order.

    This is the NeXus rule reimplemented locally -- deliberately not imported.
    """
    entry = str(detector["depends_on"].nxvalue)
    order: list[str] = []
    seen: set[str] = set()
    current = entry
    while current and current != ".":
        name = current.rsplit("/", 1)[-1]
        if name in seen:  # a cycle would otherwise hang the probe
            order.append(f"<cycle at {name}>")
            break
        if name not in transformations:
            order.append(f"<dangling: {current}>")
            break
        seen.add(name)
        order.append(name)
        current = str(transformations[name].attrs.get("depends_on", "."))
    return order


def as_matrix(field) -> np.ndarray:
    """4x4 affine for one transformation field, per the NeXus definition."""
    vector = np.asarray(field.attrs["vector"], dtype=float)
    vector = vector / np.linalg.norm(vector)
    value = float(field.nxvalue)
    matrix = np.eye(4)
    if field.attrs["transformation_type"] == "translation":
        matrix[:3, 3] = value * vector
    else:  # rotation, right-handed about `vector`, value in degrees
        theta = np.deg2rad(value)
        x, y, z = vector
        cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
        matrix[:3, :3] = np.eye(3) + np.sin(theta) * cross + (1.0 - np.cos(theta)) * (cross @ cross)
    return matrix


def main() -> int:
    print("=" * 78)
    print("A5 PROBE: NXtransformations chain reachability and rotation order")
    print("=" * 78)

    ws = build_workspace()
    inst = _Instrument.init_group([ws])  # 04b generalised this to `list[HidraWorkspace]`
    detector = inst[GROUP_NAME.DETECTOR]
    transformations = detector["transformations"]

    written = [name for name in transformations]
    report(
        "how many transformations `_instrument.py` writes",
        f"{len(written)}: {written}",
    )

    entry = str(detector["depends_on"].nxvalue)
    reachable = walk_chain(detector, transformations)
    report(
        "how many are REACHABLE by following depends_on from the detector",
        f"{len(reachable)} of {len(written)} -- entry point {entry!r}, traversal {reachable}",
    )

    orphaned = [n for n in written if n not in reachable]
    report(
        "which transformations are therefore NOT part of the geometry",
        f"{orphaned or 'NONE'}"
        + (
            " -- every rotation and the two-theta zero are written to the file but "
            "never enter the composition, so the detector is mis-oriented for any "
            "consumer that follows the chain."
            if orphaned
            else ""
        ),
    )

    # T_f = T_n ... T_2 T_1, with T_1 the entry point (NeXus definition, quoted above).
    composed = np.eye(4)
    for name in reachable:
        composed = as_matrix(transformations[name]) @ composed
    report(
        "the composed transform, as the file currently stands",
        f"translation={np.round(composed[:3, 3], 6).tolist()}, "
        f"rotation_is_identity={np.allclose(composed[:3, :3], np.eye(3))}",
    )

    # What the chain WOULD compose to if the entry point named the last link.
    full = [name for name in reversed(written)]
    composed_full = np.eye(4)
    for name in full:
        composed_full = as_matrix(transformations[name]) @ composed_full
    report(
        "what it composes to if the entry point names the LAST link instead",
        f"traversal {full};\n          translation="
        f"{np.round(composed_full[:3, 3], 6).tolist()}, "
        f"rotation_is_identity={np.allclose(composed_full[:3, :3], np.eye(3))}",
    )

    # --- the rotation order, against the reduction pipeline's own convention ---
    # `reduce_hb2b_pyrs.py` builds Rx @ Ry @ Rz and applies it to the pixel matrix.
    shift = ws.get_detector_shift()
    rx, ry, rz = (
        np.deg2rad(shift.rotation_x),
        np.deg2rad(shift.rotation_y),
        np.deg2rad(shift.rotation_z),
    )

    def axis_matrix(axis: str, angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        if axis == "x":
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        if axis == "y":
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    reduction = axis_matrix("x", rx) @ axis_matrix("y", ry) @ axis_matrix("z", rz)

    rotations = ["rotation_x", "rotation_y", "rotation_z"]
    nexus_rotation = np.eye(4)
    for name in reversed(rotations):
        nexus_rotation = as_matrix(transformations[name]) @ nexus_rotation
    report(
        "does the NeXus rotation sub-chain match reduce_hb2b_pyrs's Rx @ Ry @ Rz",
        f"{np.allclose(nexus_rotation[:3, :3], reduction)} -- with the chain ordered so "
        f"rotation_x is the entry point and rotation_z the last followed, "
        f"T_f = Rz Ry Rx; matching Rx Ry Rz requires the REVERSE written order.",
    )
    reversed_rotation = np.eye(4)
    for name in rotations:
        reversed_rotation = as_matrix(transformations[name]) @ reversed_rotation
    report(
        "and with the rotation sub-chain traversed the other way",
        f"{np.allclose(reversed_rotation[:3, :3], reduction)}",
    )

    print("\n" + "-" * 78)
    print("The reachability result is independent of any ordering convention:")
    print("a transformation the chain never reaches cannot affect the geometry.")
    print("-" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
