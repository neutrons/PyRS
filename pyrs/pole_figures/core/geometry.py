"""
HB-2B Eulerian-cradle geometry: mount conventions, beams, rotation and poles.

Ported from the reference script (``rot``, ``Mount``). Its global ``CHI_OFF`` is the explicit ``chi_offset_deg``
here. The pole-figure projection is in :mod:`.projection`.

Frames: the sample frame is the STL frame, with the virtual axes vx, vy, vz (mm). The cradle (instrument) frame at
chi 0, phi 0 has e_x vertical (the phi axis), e_z along Q at omega = theta, and e_y = e_z x e_x. All angles are in
degrees.

- incident ``ki = (0, sb cos w, -sin w)``, diffracted ``kf = (0, sb cos(2t - w), sin(2t - w))``
- sample to cradle: ``R(chi, phi) = rot(e_y, s_chi chi) @ rot(e_x, s_phi phi) @ M0`` (``rotation.rot``)
- pole in the sample frame: ``h = R^T q``

Typical workflow:

1. Describe how the sample sits on the cradle with :class:`MountConventions`.
2. Build a :class:`Mount` for one omega and 2theta.
3. Get the pole of each cradle chi, phi with :meth:`Mount.pole` or :meth:`Mount.poles`.
4. Place the poles on a pole figure with ``projection.project``.

Example:
    >>> m = Mount(MountConventions(), omega=45.0, tth=90.0)
    >>> m.pole(0.0, 0.0)
    array([0., 0., 1.])
"""

import difflib
import typing
from dataclasses import dataclass, fields, replace
from typing import Any, Literal, NamedTuple

import numpy as np
import numpy.typing as npt

from .rotation import SampleAxis, axis_name, axis_vector, check_rotation, nearest_rotation, rot
from .types import FloatArray

PhiSense = Literal["clockwise_from_above", "counterclockwise_from_above"]
IncidentFrom = Literal["+vy", "-vy"]


class AxesAtZero(NamedTuple):
    """
    Sample axis nearest each instrument direction at chi 0, phi 0 (see ``MountConventions.axes_at_zero``).

    Attributes:
        up: Along e_x, the phi axis (up).
        across: Along e_y, the chi axis.
        q: Along e_z, Q at omega = theta.
    """

    up: str
    across: str
    q: str


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
        AxesAtZero(up='+vz', across='-vy', q='+vx')
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

    def axes_at_zero(self) -> "AxesAtZero":
        """
        Sample axis nearest each instrument direction at chi 0, phi 0.

        Returns:
            The signed sample axes (such as '+vx') along e_x (up), e_y (across) and e_z (Q). A leading '~' marks an
            axis more than 1 deg from that direction (a tilted mount).
        """
        names = [axis_name(row) if np.abs(row).max() > np.cos(np.radians(1.0)) else "~" + axis_name(row)
                 for row in self.M0]  # fmt: skip
        return AxesAtZero(up=names[0], across=names[1], q=names[2])

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
