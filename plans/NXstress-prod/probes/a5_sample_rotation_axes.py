"""A5: which axes do PyRS's `omega`, `chi` and `phi` stage angles rotate about?

Spec 04's Q5 asks what the sample-frame `NXtransformations` chain should hold.
The translations are obvious -- `vx`/`vy`/`vz` are already written in millimetres
along the coordinate axes. The rotations are not: PyRS applies the three stage
angles in `PoleFigureCalculator.rotate_project_q`
(`pyrs/core/polefigurecalculator.py:186`), the **closed-form pole-figure
projection** of Bob He, *Two-Dimensional X-ray Diffraction*, ch. 8.3. It builds
two components of a unit vector directly out of sines and cosines. No matrix
appears anywhere in it, so there is no axis to read off, and the first reading of
this was that the convention was simply unrecoverable.

**That reading was wrong, and this probe is what establishes it.** A rotation
composition is a rotation composition however it is spelled. Two steps:

1. *Recover the matrix.* The formula rotates a probe vector fixed by `theta` and
   `eta`; choosing those two angles aims the probe vector along each basis
   vector in turn, so three calls read off three columns. This needs no
   reimplementation of the formula at all -- it is driven through its public
   signature, as a black box, which is the strongest form this evidence can take.
2. *Identify it.* Enumerate every candidate `(axis assignment, multiplication
   order, fixed frame offset)` over the six signed axes and keep those
   reproducing the recovered matrix for many random angle triples.

**The frame offset is why a first attempt at this found nothing.** Evaluating the
formula at `omega = chi = phi = 0` does not give the identity -- it gives
`diag(1, -1, -1)`, a 180-degree rotation about `+x` relating the formula's output
frame to the lab frame. An enumeration over stage rotations alone cannot match a
composition that has a constant factor in it, and reports zero candidates while
the convention is in fact fully determined.

Per `process.md` §5.3 the candidate construction is written **locally** rather
than imported, so this probe fails if the codebase changes rather than silently
tracking it. `rotate_project_q` is imported, because it is the referent.

The third row the formula discards is **not** a loss of information: the two it
returns are orthonormal, so the third is their cross product up to the sign fixed
by requiring a proper rotation (`det = +1`). The probe reconstructs it and checks
that the completed matrix really is orthogonal, rather than assuming it.

Run: ``pixi run python plans/NXstress-prod/probes/a5_sample_rotation_axes.py``
"""

from __future__ import annotations

import itertools
import sys

import numpy as np

from pyrs.core.polefigurecalculator import PoleFigureCalculator

#: The six signed unit axes a stage angle could plausibly turn about.
AXES = {
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0),
    "-z": (0.0, 0.0, -1.0),
}

LABELS = ("omega", "chi", "phi")

#: `(theta, eta)` aiming the formula's probe vector along x, y and z. The probe
#: vector is `(sin th, cos th cos g, -cos th sin g)` with `g = eta + 270`, which
#: is read off the formula's own grouping of its terms -- the one piece of its
#: internals this probe depends on, and a wrong choice here shows up immediately
#: as zero surviving candidates rather than as a wrong answer.
BASIS_DRIVE = ((90.0, 0.0), (0.0, 90.0), (0.0, 0.0))

N_TRIALS = 40
TOLERANCE = 1e-9

#: Triples used for the independent end-to-end check of the adopted form.
N_VERIFY = 200


def observed(calculator: PoleFigureCalculator, omega: float, chi: float, phi: float) -> np.ndarray:
    """The full lab -> sample rotation matrix the formula implements, for one angle set.

    Driven entirely through `rotate_project_q`'s public signature: each column is
    one call with the probe vector aimed along one basis vector. The returned
    `(90 - alpha, beta)` is converted back to `(h1, h2)` -- `alpha` carries the
    in-plane length and `beta` its direction, with `h2`'s sign folded into
    `beta`'s, so the inversion is exact rather than approximate. The third row is
    the cross product of the first two, which is forced once they are known to be
    orthonormal; `main` checks that they are.
    """
    columns = []
    for theta, eta in BASIS_DRIVE:
        elevation, beta = calculator.rotate_project_q(theta, omega, chi, phi, eta)
        length = np.sin(np.radians(elevation))
        columns.append([length * np.cos(np.radians(beta)), length * np.sin(np.radians(beta))])
    rows = np.asarray(columns).T
    return np.vstack([rows, np.cross(rows[0], rows[1])])


def rotation(axis: tuple[float, float, float], degrees: float) -> np.ndarray:
    """Right-handed rotation of `degrees` about `axis`, by Rodrigues' formula."""
    k = np.asarray(axis, dtype=float)
    k = k / np.linalg.norm(k)
    theta = np.radians(degrees)
    cross = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * cross + (1.0 - np.cos(theta)) * (cross @ cross)


def main() -> int:
    calculator = PoleFigureCalculator()
    rng = np.random.default_rng(seed=0)
    trials = [tuple(rng.uniform(-180.0, 180.0, size=3)) for _ in range(N_TRIALS)]
    # Random and plentiful, so a candidate cannot survive by coinciding at a
    # symmetric special case -- which `chi = phi = 0`, the value every HB2B file
    # in `tests/data` happens to carry, very much is.
    matrices = [observed(calculator, *trial) for trial in trials]

    print("STEP 1 -- recover the matrix, through the public signature only\n")
    offset = observed(calculator, 0.0, 0.0, 0.0)
    print("at omega = chi = phi = 0 the formula gives")
    for row in offset:
        print("   ", np.array2string(row, precision=6, sign="+", floatmode="fixed"))
    print(f"  identity?                      {np.allclose(offset, np.eye(3))}")
    print(f"  180 degrees about +x?          {np.allclose(offset, rotation((1.0, 0.0, 0.0), 180.0))}")
    print("  ^ a constant frame offset. A search over stage rotations alone cannot match it.")

    orthogonal = all(np.allclose(m @ m.T, np.eye(3), atol=TOLERANCE) for m in matrices)
    proper = all(abs(np.linalg.det(m) - 1.0) < 1e-9 for m in matrices)
    print(f"\n  every recovered matrix orthogonal: {orthogonal}, det +1: {proper}")
    print("  ^ so the discarded third row was reconstructible, not lost.")

    print(f"\nSTEP 2 -- identify it, over {N_TRIALS} random (omega, chi, phi) triples\n")
    survivors, considered = [], 0
    for names in itertools.permutations(AXES, 3):
        for order in itertools.permutations(range(3)):
            for pre in (False, True):
                for post in (False, True):
                    considered += 1
                    if all(
                        np.allclose(candidate(names, order, pre, post, offset, t), m, atol=TOLERANCE, rtol=0.0)
                        for t, m in zip(trials, matrices)
                    ):
                        survivors.append((names, order, pre, post))

    print(f"  candidates considered: {considered}")
    print(f"  exact matches:         {len(survivors)}\n")
    for names, order, pre, post in survivors:
        composition = " @ ".join(f"R_{LABELS[i]}" for i in order)
        formula = ("M0 @ " if pre else "") + composition + (" @ M0" if post else "")
        print(f"  {', '.join(f'{LABELS[i]} about {names[i]}' for i in range(3))}")
        print(f"      lab <- sample reading:  M = {formula}")

    if not survivors:
        print("\n  No candidate matches: the formula is not a composition of three axis rotations.")
        return 1
    if len(survivors) > 1:
        print(
            "\n  The survivors are conjugate by M0 (a 180-degree rotation about x, which\n"
            "  negates the y and z axes), so they are the same physical rotation written\n"
            "  with the constant factor on either side -- not a genuine ambiguity."
        )

    print(f"\nSTEP 3 -- the form NXtransformations needs, checked on {N_VERIFY} fresh triples\n")
    print("  NXtransformations describes sample -> lab, which is the transpose of the")
    print("  lab -> sample matrix recovered above.\n")
    verify = np.random.default_rng(seed=7)
    worst = 0.0
    for _ in range(N_VERIFY):
        omega, chi, phi = verify.uniform(-180.0, 180.0, size=3)
        adopted = (
            offset
            @ rotation((0.0, -1.0, 0.0), omega)
            @ rotation((1.0, 0.0, 0.0), chi)
            @ rotation((0.0, 0.0, -1.0), phi)
        )
        worst = max(worst, float(np.abs(adopted - observed(calculator, omega, chi, phi).T).max()))
    print("  sample -> lab  ==  M0 @ R_omega(0,-1,0) @ R_chi(1,0,0) @ R_phi(0,0,-1)")
    print(f"  worst elementwise error over {N_VERIFY} triples: {worst:.3e}")
    if worst > 1e-9:
        print("  MISMATCH -- the adopted form is wrong.")
        return 1
    print(
        "\n  Write order is composition order for a NeXus chain entered at its last link,\n"
        "  so `_sample.py` writes frame_offset, omega, chi, phi in that order, with\n"
        "  `depends_on` naming phi. Nothing here is left to choose."
    )
    return 0


def candidate(names, order, pre, post, offset, trial) -> np.ndarray:
    """The candidate composition's 3x3 matrix for one angle set."""
    matrices = [rotation(AXES[names[i]], trial[i]) for i in range(3)]
    composed = matrices[order[0]] @ matrices[order[1]] @ matrices[order[2]]
    if pre:
        composed = offset @ composed
    if post:
        composed = composed @ offset
    return composed


if __name__ == "__main__":
    sys.exit(main())
