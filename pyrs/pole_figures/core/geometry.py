"""
HB-2B Eulerian-cradle geometry: mount conventions, beams, rotation, poles and the equal-area projection.

Ported from the reference script (``rot``, ``Mount``, ``project``). Its globals ``CHI_OFF`` and ``VIEW`` are explicit
arguments here.

Frames: the sample frame is the STL frame, with the virtual axes vx, vy, vz (mm). The cradle (instrument) frame at
chi 0, phi 0 has e_x vertical (the phi axis), e_z along Q at omega = theta, and e_y = e_z x e_x. All angles are in
degrees.

- incident ``ki = (0, sb cos w, -sin w)``, diffracted ``kf = (0, sb cos(2t - w), sin(2t - w))``
- sample to cradle: ``R(chi, phi) = rot(e_y, s_chi chi) @ rot(e_x, s_phi phi) @ M0``
- pole in the sample frame: ``h = R^T q``

Typical workflow:

1. Describe how the sample sits on the cradle with :class:`MountConventions`.
2. Build a :class:`Mount` for one omega and 2theta.
3. Get the pole of each cradle chi, phi with :meth:`Mount.pole` or :meth:`Mount.poles`.
4. Place the poles on a pole figure with :func:`project`.

Example:
    >>> m = Mount(MountConventions(), omega=45.0, tth=90.0)
    >>> m.pole(0.0, 0.0)
    array([0., 0., 1.])
    >>> project(m.pole(0.0, 0.0))
    array([[0., 0.]])
"""

import difflib
import typing
from dataclasses import dataclass, fields, replace
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
PhiSense = Literal["clockwise_from_above", "counterclockwise_from_above"]
IncidentFrom = Literal["+vy", "-vy"]
ViewFrom = Literal["+vz", "-vz", "+vx"]
SampleAxis = Literal["+vx", "-vx", "+vy", "-vy", "+vz", "-vz"]
SAMPLE_AXES: tuple[SampleAxis, ...] = ("+vx", "-vx", "+vy", "-vy", "+vz", "-vz")


def axis_vector(a: str) -> FloatArray:
    """
    Unit vector of a signed sample axis: '+vx' -> (1, 0, 0), '-vz' -> (0, 0, -1).

    Args:
        a: One of ``SAMPLE_AXES``.

    Returns:
        The unit vector in the sample frame.

    Raises:
        ValueError: If ``a`` is not one of ``SAMPLE_AXES``.
    """
    if a not in SAMPLE_AXES:
        raise ValueError(f"sample axis must be one of {', '.join(SAMPLE_AXES)}, not {a!r}")
    v = np.zeros(3)
    v["xyz".index(a[2])] = 1.0 if a[0] == "+" else -1.0
    return v


def check_rotation(m: npt.ArrayLike, *, tol: float = 2e-3) -> None:
    """
    Check that m is a proper rotation (orthonormal rows, determinant +1) within tol.

    Args:
        m: A 3 x 3 sample-to-instrument matrix.
        tol: Largest allowed element of ``|m m^T - I|`` (default: 2e-3, so values typed to 3 or 4 decimals pass).

    Raises:
        ValueError: If m is not 3 x 3 finite numbers, its rows are not orthonormal within tol, or it is a reflection
            (negative determinant).
    """
    m = np.asarray(m, float)
    if m.shape != (3, 3) or not np.all(np.isfinite(m)):
        raise ValueError(f"the sample-to-instrument matrix must be 3 x 3 finite numbers, got {m.tolist()!r}")
    err = float(np.abs(m @ m.T - np.eye(3)).max())
    if err > tol:
        raise ValueError(f"the sample-to-instrument matrix is not a rotation: rows must be unit length and at right "
                         f"angles (off by {err:.3g})")  # fmt: skip
    det = float(np.linalg.det(m))
    if det < 0:
        raise ValueError(f"the sample-to-instrument matrix is a reflection (determinant {det:.3g}): flip the sign of "
                         "one row")  # fmt: skip


def nearest_rotation(m: npt.ArrayLike) -> FloatArray:
    """
    The rotation closest to m (polar decomposition), so a matrix typed to 3 or 4 decimals is exactly orthonormal.

    A matrix that is already orthonormal (to 1e-12) is returned as it is, so the identity stays exact.

    Args:
        m: A 3 x 3 matrix that is nearly a rotation.

    Returns:
        The orthonormal 3 x 3 matrix ``U V^T`` from the singular value decomposition of m.
    """
    m = np.asarray(m, float)
    if np.abs(m @ m.T - np.eye(3)).max() < 1e-12:
        return m.copy()
    u, _, vt = np.linalg.svd(m)
    return np.asarray(u @ vt)


#: stage (instrument) axes at chi 0, phi 0: e_x up along the phi axis, e_y across (the chi axis), e_z along Q
STAGE_AXES: tuple[str, ...] = ("phi axis (up)", "chi axis (across)", "Q (omega = theta)")
IDENTITY9: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def rotate_mount(m: npt.ArrayLike, stage_axis: int, deg: float) -> tuple[float, ...]:
    """
    Turn the mounted sample about a stage axis by deg (right hand).

    The sample-to-instrument matrix M0 becomes rot(e_axis, deg) @ M0, so presses add up in the stage frame. The
    result is made exactly orthonormal and values below 1e-12 are set to 0.

    Args:
        m: The current M0, 9 values row by row (or 3 x 3).
        stage_axis: Stage axis at chi 0, phi 0: 0 the phi axis (e_x, up), 1 the chi axis (e_y), 2 Q (e_z).
        deg: Rotation angle (deg).

    Returns:
        The new M0 as 9 values, row by row.

    Example:
        >>> rotate_mount(IDENTITY9, 2, 90.0)
        (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    """
    m3 = np.reshape(np.asarray(m, float), (3, 3))
    new = nearest_rotation(rot(np.eye(3)[stage_axis], deg) @ m3)
    new[np.abs(new) < 1e-12] = 0.0
    return tuple(float(x) for x in new.ravel())


def parse_matrix(text: str) -> tuple[float, ...] | None:
    """
    Read a 3 x 3 matrix typed as text, rows split by ';' or new lines and numbers by spaces or commas.

    Args:
        text: The matrix, e.g. '0 0 1; 1 0 0; 0 1 0'.

    Returns:
        The 9 numbers row by row, or None for empty text (no matrix given: the mount uses up_axis and q_axis).

    Raises:
        ValueError: If the text is not 3 rows of 3 numbers.

    Example:
        >>> parse_matrix("0 0 1; 1 0 0; 0 1 0")
        (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    """
    rows = [r for r in text.replace("\n", ";").split(";") if r.strip()]
    if not rows:
        return None
    vals = [[float(x) for x in r.replace(",", " ").split()] for r in rows]
    if len(vals) != 3 or any(len(r) != 3 for r in vals):
        raise ValueError(f"the sample-to-instrument matrix needs 3 rows of 3 numbers, e.g. 0 0 1; 1 0 0; 0 1 0, "
                         f"got {text!r}")  # fmt: skip
    return tuple(float(x) for r in vals for x in r)


def format_matrix(m: npt.ArrayLike) -> str:
    """
    Write a 3 x 3 matrix as text that :func:`parse_matrix` reads back.

    Args:
        m: The matrix.

    Returns:
        Rows split by '; ', 4 significant digits, values within 5e-5 of 0 shown as 0 (so no '-0').

    Example:
        >>> format_matrix(np.eye(3))
        '1 0 0; 0 1 0; 0 0 1'
    """
    m = np.asarray(m, float) + 0.0
    return "; ".join(" ".join(f"{v:.4g}" if abs(v) > 5e-5 else "0" for v in row) for row in m)


def axis_name(v: npt.ArrayLike) -> str:
    """
    Name of the sample axis nearest a vector: (0, -1, 0) -> '-vy'.

    Args:
        v: A vector in the sample frame. Only its largest component counts.

    Returns:
        The signed axis name, one of ``SAMPLE_AXES``.
    """
    v = np.asarray(v, float)
    i = int(np.argmax(np.abs(v)))
    return ("+" if v[i] > 0 else "-") + "v" + "xyz"[i]


# Display frame for "pf_view_from": "+vx": (vy, vz, vx), vx toward the viewer.
_VIEW_PLUS_VX = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])


def rot(axis: npt.ArrayLike, deg: float) -> FloatArray:
    """
    Rotation matrix about an axis (right hand, Rodrigues formula).

    Args:
        axis: Rotation axis, any length.
        deg: Rotation angle (deg).

    Returns:
        The 3 x 3 rotation matrix.
    """
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    t = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


@dataclass(frozen=True, kw_only=True)
class MountConventions:
    """
    How the sample sits on the cradle (the ``conventions`` block of a reference-script config).

    At chi 0, phi 0 the cradle axes are e_x (up, the phi axis), e_y (horizontal, across the beams) and e_z (Q at
    omega = theta). phi turns about e_x, chi about e_y, and chi 90 puts the phi axis along Q. By default +vx is
    up, +vy along e_y and +vz along Q. ``up_axis`` and ``q_axis`` mount the sample another way, e.g. a cylinder
    turned about its long axis vz by phi: up_axis "+vz". A full ``sample_to_instrument`` matrix replaces both for a
    tilted mount.

    Attributes:
        phi_sense: Sense of +phi seen from above (check with the top camera): "clockwise_from_above" (default)
            or "counterclockwise_from_above".
        vx_up: +vx points up at chi 0 (default: True). If False, vx and vy are reversed (legacy key, applied after
            up_axis and q_axis, or after sample_to_instrument).
        chi_tilts_up_axis_toward_Q: Sense of chi: +chi tilts the upward phi axis toward Q (default: True).
        incident_from: Side the incident beam comes from at chi 0, phi 0, "+vy" or "-vy" (default: "-vy").
        chi_offset_deg: Cradle chi = logged chi + chi_offset_deg, for a goniometer whose chi zero is not the cradle
            one (deg, default: 0).
        eta_up_sign: Sense of eta for eta-sliced runs, +1 or -1 (default: +1).
        up_axis: Sample axis along the phi axis (up) at chi 0 (default: "+vx").
        q_axis: Sample axis along Q (e_z) at chi 0, phi 0, at right angles to up_axis (default: "+vz"). The third
            axis (along e_y) follows from the right-hand rule.
        sample_to_instrument: The 3 x 3 sample-to-instrument matrix M0, 9 values row by row, or None to build M0
            from up_axis and q_axis (default: None).

    Raises:
        ValueError: If a value is not one of its allowed options, up_axis and q_axis are not at right angles, or
            sample_to_instrument is not a rotation.

    Example:
        >>> MountConventions(up_axis="+vz", q_axis="+vx").axes_at_zero()
        {'up': '+vz', 'across': '-vy', 'Q': '+vx'}
    """

    phi_sense: PhiSense = "clockwise_from_above"
    vx_up: bool = True
    chi_tilts_up_axis_toward_Q: bool = True  # noqa: N815 (config key name)
    incident_from: IncidentFrom = "-vy"
    chi_offset_deg: float = 0.0
    eta_up_sign: float = 1.0
    up_axis: SampleAxis = "+vx"
    q_axis: SampleAxis = "+vz"
    sample_to_instrument: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        """Check every value and that the mount is a rotation (raises ValueError)."""
        allowed = {"phi_sense": typing.get_args(PhiSense), "incident_from": typing.get_args(IncidentFrom),
                   "up_axis": typing.get_args(SampleAxis), "q_axis": typing.get_args(SampleAxis)}  # fmt: skip
        for name, values in allowed.items():
            if getattr(self, name) not in values:
                raise ValueError(f"{name} must be one of {', '.join(values)}, not {getattr(self, name)!r}")
        for name in ("vx_up", "chi_tilts_up_axis_toward_Q"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError(f"{name} must be true or false, not {getattr(self, name)!r}")
        if self.eta_up_sign not in (1, -1):
            raise ValueError(f"eta_up_sign must be +1 or -1, not {self.eta_up_sign!r}")
        if not np.isfinite(self.chi_offset_deg):
            raise ValueError(f"chi_offset_deg must be a number, not {self.chi_offset_deg!r}")
        if abs(float(axis_vector(self.up_axis) @ axis_vector(self.q_axis))) > 0.5:
            raise ValueError(f"the up axis ({self.up_axis}) and the axis along Q ({self.q_axis}) must be different "
                             "sample axes (at right angles)")  # fmt: skip
        if self.sample_to_instrument is not None:
            check_rotation(np.reshape(self.sample_to_instrument, (3, 3)))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MountConventions":
        """
        Build from a ``conventions`` dict (the reference script's config block).

        Args:
            d: Field name -> value. Keys starting with ``_`` are notes and are ignored. Missing keys take the
                defaults.

        Returns:
            The mount conventions.

        Raises:
            ValueError: If a key is unknown (a misspelled key would otherwise give the default mount; the message
                suggests the closest name) or a value is not allowed.
        """
        names = {f.name for f in fields(cls)}
        unknown = [k for k in d if k not in names and not k.startswith("_")]
        if unknown:
            hints = [f"{k} (did you mean {m[0]}?)" if (m := difflib.get_close_matches(k, names, 1)) else k
                     for k in unknown]  # fmt: skip
            raise ValueError(f"unknown mount convention key: {', '.join(hints)}")
        return cls(**{k: v for k, v in d.items() if k in names})

    @property
    def s_phi(self) -> float:
        """Sign applied to phi in ``R``: -1 for clockwise from above, +1 for counterclockwise."""
        return -1.0 if self.phi_sense == "clockwise_from_above" else 1.0

    @property
    def s_chi(self) -> float:
        """Sign applied to chi in ``R``: -1 when +chi tilts the up axis toward Q, else +1."""
        return -1.0 if self.chi_tilts_up_axis_toward_Q else 1.0

    @property
    def sb(self) -> float:
        """Beam side sign ``sb`` in ki and kf: -1 when the incident beam comes from +vy, else +1."""
        return -1.0 if self.incident_from == "+vy" else 1.0

    @property
    def M0(self) -> FloatArray:  # noqa: N802 (matches the physics notation)
        """
        Sample-to-instrument (cradle) rotation at chi 0, phi 0: instrument vector = M0 @ sample vector.

        Rows are e_x (up, phi axis), e_y (across) and e_z (Q) in sample coordinates. Identity for the default mount.
        A given ``sample_to_instrument`` matrix is used as it is (made exactly orthonormal), otherwise it is built from
        ``up_axis`` and ``q_axis``. ``vx_up`` = False is applied on top.
        """
        if self.sample_to_instrument is not None:
            m = nearest_rotation(np.reshape(self.sample_to_instrument, (3, 3)))
        else:
            u, w = axis_vector(self.up_axis), axis_vector(self.q_axis)
            m = np.vstack([u, np.cross(w, u), w])
        return m if self.vx_up else np.diag([-1.0, -1.0, 1.0]) @ m

    def axes_at_zero(self) -> dict[str, str]:
        """
        Sample axis nearest each instrument direction at chi 0, phi 0.

        Returns:
            Keys 'up' (e_x), 'across' (e_y) and 'Q' (e_z), values signed sample axes such as '+vx'. A leading '~'
            marks an axis more than 1 deg from that direction (a tilted mount).
        """
        m = self.M0
        out = {}
        for key, row in zip(("up", "across", "Q"), m, strict=True):
            out[key] = axis_name(row) if np.abs(row).max() > np.cos(np.radians(1.0)) else "~" + axis_name(row)
        return out

    def cradle_chi(self, logged_chi: float | FloatArray) -> float | FloatArray:
        """
        Cradle chi used in the geometry from the logged chi: logged chi + ``chi_offset_deg``.

        Args:
            logged_chi: Chi from the run logs or goniometer (deg).

        Returns:
            Cradle chi (deg), same shape as the input.
        """
        return logged_chi + self.chi_offset_deg

    def logged_chi(self, cradle_chi: float | FloatArray) -> float | FloatArray:
        """
        Logged chi (run file, goniometer) from the cradle chi: cradle chi - ``chi_offset_deg``.

        Args:
            cradle_chi: Cradle chi used in the geometry (deg).

        Returns:
            Logged chi (deg), same shape as the input.
        """
        return cradle_chi - self.chi_offset_deg

    def mirrored(self) -> "MountConventions":
        """Same mount with the beams on the other side (for a background check with mirrored beams)."""
        other: IncidentFrom = "-vy" if self.incident_from == "+vy" else "+vy"
        return replace(self, incident_from=other)


class Mount:
    """
    Beams and sample orientation for one omega and 2theta.

    The beam vectors are unit vectors in the cradle frame at chi 0, phi 0.

    Args:
        conv: Mount conventions.
        omega: Omega (deg).
        tth: Scattering angle 2theta (deg).
        eta_deg: Eta slice: kf turned out of the scattering plane toward +e_x by eta (deg, default: 0 for the full
            detector).

    Attributes:
        conv: The mount conventions.
        omega: Omega (deg).
        tth: 2theta (deg).
        eta: Eta slice (deg).
        ki: Incident beam direction (cradle frame).
        kf: Diffracted beam direction (cradle frame).
        q: Unit scattering vector ``(kf - ki) / |kf - ki|`` (cradle frame).
        s_phi: Sign applied to phi (from ``conv``).
        s_chi: Sign applied to chi (from ``conv``).
        M0: Sample-to-cradle rotation at chi 0, phi 0 (from ``conv``).
        band_offset: Shift of the diffracted band center along e_x x kf (mm), set by :meth:`set_band_offset`.

    Example:
        >>> m = Mount(MountConventions(), omega=45.0, tth=90.0)
        >>> m.q
        array([0., 0., 1.])
    """

    def __init__(self, conv: MountConventions, *, omega: float, tth: float, eta_deg: float = 0.0) -> None:
        self.conv = conv
        self.omega, self.tth = float(omega), float(tth)
        self.s_phi, self.s_chi, self.M0 = conv.s_phi, conv.s_chi, conv.M0
        sb = conv.sb
        w, t2 = np.radians(omega), np.radians(tth)
        self.ki: FloatArray = np.array([0.0, sb * np.cos(w), -np.sin(w)])
        self.kf: FloatArray = np.array([0.0, sb * np.cos(t2 - w), np.sin(t2 - w)])
        self.eta = float(eta_deg)
        if self.eta:
            # kf on the Debye cone, turned out of the scattering plane toward +e_x (up) by eta, so
            # kf . e_x = sin 2theta sin eta as for the pyRS pixel eta.
            e = np.radians(self.eta) * conv.eta_up_sign
            uh = (self.kf - np.cos(t2) * self.ki) / np.sin(t2)
            self.kf = np.cos(t2) * self.ki + np.sin(t2) * (np.cos(e) * uh + np.sin(e) * np.array([1.0, 0.0, 0.0]))
        q = self.kf - self.ki
        self.q: FloatArray = q / np.linalg.norm(q)
        self.band_offset = 0.0  # mm along e_x x kf; see set_band_offset

    def set_band_offset(self, offset_mm: float) -> None:
        """
        Shift the diffracted band off the gauge center (``gauge.set_slit_offset`` computes the shift).

        Args:
            offset_mm: Shift of the band center along e_x x kf (mm). 0 centers the band on the gauge.
        """
        self.band_offset = float(offset_mm)

    def R(self, chi: float, phi: float) -> FloatArray:  # noqa: N802 (physics notation)
        """
        Sample-to-cradle rotation: ``rot(e_y, s_chi chi) @ rot(e_x, s_phi phi) @ M0``.

        Args:
            chi: Cradle chi (deg).
            phi: Phi (deg).

        Returns:
            The 3 x 3 matrix: cradle vector = R @ sample vector.
        """
        return rot([0, 1, 0], self.s_chi * chi) @ rot([1, 0, 0], self.s_phi * phi) @ self.M0

    def pole(self, chi: float, phi: float) -> FloatArray:
        """
        Unit scattering vector (the pole) in the sample frame: ``R(chi, phi)^T q``.

        Args:
            chi: Cradle chi (deg).
            phi: Phi (deg).

        Returns:
            The pole (vx, vy, vz components).
        """
        return self.R(chi, phi).T @ self.q

    def poles(self, chi: npt.ArrayLike, phi: npt.ArrayLike) -> FloatArray:
        """
        :meth:`pole` for arrays of cradle chi and phi, vectorized.

        Same rotations as ``R`` written out: h = M0^T Rx(s_phi phi)^T Ry(s_chi chi)^T q.

        Args:
            chi: Cradle chi (deg), any shape (flattened).
            phi: Phi (deg), same size as chi.

        Returns:
            Poles in the sample frame, shape (n, 3).
        """
        c = np.radians(self.s_chi * np.asarray(chi, float).ravel())
        p = np.radians(self.s_phi * np.asarray(phi, float).ravel())
        qx, qy, qz = self.q
        cc, sc = np.cos(c), np.sin(c)
        ax, ay, az = cc * qx - sc * qz, np.full_like(c, qy), sc * qx + cc * qz  # Ry(chi)^T q
        cp, sp = np.cos(p), np.sin(p)
        v = np.c_[ax, cp * ay + sp * az, -sp * ay + cp * az]  # Rx(phi)^T
        return np.asarray(v @ self.M0)  # M0^T applied to row vectors

    def phi_axis(self, chi: float) -> FloatArray:
        """
        Phi axis in the cradle frame at a cradle chi (e_x at chi 0).

        Args:
            chi: Cradle chi (deg).

        Returns:
            Unit vector of the phi axis.
        """
        return rot([0, 1, 0], self.s_chi * chi) @ np.array([1.0, 0.0, 0.0])


def check_view(view: str) -> None:
    """
    Check that ``view`` is one of the pole-figure views (``ViewFrom``).

    Args:
        view: "+vz", "-vz" or "+vx".

    Raises:
        ValueError: If view is anything else.
    """
    allowed = typing.get_args(ViewFrom)
    if view not in allowed:
        raise ValueError(f"view must be one of {', '.join(allowed)}, got {view!r}")


def to_view(h: npt.ArrayLike, *, view: ViewFrom = "+vz") -> FloatArray:
    """
    Sample-frame vectors in the display frame (x right, y up, z toward the viewer).

    Args:
        h: Sample-frame vectors, shape (..., 3).
        view: "+vz", "-vz" or "+vx" (default: "+vz"). Only "+vx" changes the frame: (vy, vz, vx). The "-vz" mirror
            is applied in :func:`project`.

    Returns:
        The vectors in the display frame, same shape.

    Raises:
        ValueError: If view is not one of the pole-figure views.
    """
    check_view(view)
    h = np.asarray(h, float)
    return h @ _VIEW_PLUS_VX.T if view == "+vx" else h


def from_view(d: npt.ArrayLike, *, view: ViewFrom = "+vz") -> FloatArray:
    """
    Display-frame vectors back to the sample frame (inverse of :func:`to_view`).

    Args:
        d: Display-frame vectors, shape (..., 3).
        view: "+vz", "-vz" or "+vx" (default: "+vz").

    Returns:
        The vectors in the sample frame, same shape.

    Raises:
        ValueError: If view is not one of the pole-figure views.
    """
    check_view(view)
    d = np.asarray(d, float)
    return d @ _VIEW_PLUS_VX if view == "+vx" else d


def project(h: npt.ArrayLike, *, view: ViewFrom = "+vz", fold: bool = True) -> FloatArray:
    """
    Equal-area (Schmidt) projection of sample-frame unit vectors.

    Args:
        h: Sample-frame unit vectors, shape (3,) or (n, 3).
        view: "+vz": seen from +vz, +vx right, +vy up (default). "-vz": as the 2D maps (vx to the left). "+vx": seen
            down the phi axis, +vy right, +vz up.
        fold: Replace poles on the far hemisphere with their antipodes (default: True).

    Returns:
        Projected points, shape (n, 2), radius 1 at the equator.

    Raises:
        ValueError: If view is not one of the pole-figure views.
    """
    check_view(view)
    sgn = -1.0 if view == "-vz" else 1.0
    h = np.atleast_2d(np.asarray(h, float))
    h = to_view(h, view=view)
    if fold:
        h = np.where(sgn * h[:, 2:3] < -1e-9, -h, h)
    alpha = np.arccos(np.clip(sgn * h[:, 2], -1, 1))
    r = np.sqrt(2) * np.sin(alpha / 2)
    xy = np.c_[sgn * h[:, 0], h[:, 1]]
    n = np.linalg.norm(xy, axis=1)
    n[n < 1e-12] = 1
    return np.asarray(xy / n[:, None] * r[:, None])
