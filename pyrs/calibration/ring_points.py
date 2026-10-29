"""
Ring points: where on the detector each fitted Debye ring was found.

:class:`RingPoints` is what :func:`pyrs.calibration.ring_extraction.extract_ring_points` produces and
:class:`pyrs.calibration.pixel_refinement.PixelCalibration` refines from; it can be saved to and
loaded from ``.npz`` so a refinement can be re-run without re-fitting rings. Part of
:mod:`pyrs.calibration.pixel_pos_calibration`.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np

__all__ = ["RingPoints"]


_RING_FIELDS = ("sub_run", "index_0", "index_1", "det_2theta", "l2", "d_spacing", "hkl_squared", "powder", "sigma")


@dataclass
class RingPoints:
    """Points on fitted Debye rings: where on the detector each ring was found.

    All attributes are 1D arrays of equal length, one entry per point.

    Attributes:
        sub_run: Sub-run the point was measured in.
        index_0: Fractional pixel index across columns (see :class:`DetectorModel`).
        index_1: Fractional pixel index along rows.
        det_2theta: Detector arm 2theta (degrees) **as logged by DAS**, i.e. ``get_detector_2theta``.
        l2: Sample-to-detector distance (m) of the sub-run.
        d_spacing: True d-spacing (Angstrom) of the reflection, from the powder standard.
        hkl_squared: ``h^2 + k^2 + l^2`` of the reflection.
        powder: Powder standard name.
        sigma: Uncertainty of the fitted ring position in 2theta (degrees); NaN if unknown.

    Example:
        >>> points = RingPoints(sub_run=[1], index_0=[500.2], index_1=[512.0], det_2theta=[85.0],
        ...                     l2=[0.985], d_spacing=[1.2467], hkl_squared=[8], powder=["Ni"])
        >>> len(points)
        1
    """

    sub_run: np.ndarray
    index_0: np.ndarray
    index_1: np.ndarray
    det_2theta: np.ndarray
    l2: np.ndarray
    d_spacing: np.ndarray
    hkl_squared: np.ndarray
    powder: np.ndarray
    sigma: Optional[np.ndarray] = field(default=None)

    def __post_init__(self) -> None:
        self.sub_run = np.asarray(self.sub_run, dtype=int)
        for name in ("index_0", "index_1", "det_2theta", "l2", "d_spacing"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=float))
        self.hkl_squared = np.asarray(self.hkl_squared, dtype=int)
        self.powder = np.asarray(self.powder, dtype=str)
        if self.sigma is None:
            self.sigma = np.full(self.sub_run.shape, np.nan)
        self.sigma = np.asarray(self.sigma, dtype=float)

        lengths = {name: getattr(self, name).shape for name in _RING_FIELDS}
        if any(len(shape) != 1 for shape in lengths.values()) or len(set(lengths.values())) != 1:
            raise ValueError(f"RingPoints fields must be 1D arrays of equal length, got shapes {lengths}")
        if np.any(self.d_spacing <= 0):
            raise ValueError("RingPoints d_spacing must be positive")

    def __len__(self) -> int:
        return int(self.sub_run.size)

    def select(self, mask: np.ndarray) -> "RingPoints":
        """Subset of the points, by boolean mask or index array."""
        return RingPoints(**{name: getattr(self, name)[mask] for name in _RING_FIELDS})

    def save(self, path: Union[str, Path]) -> None:
        """Write the points to a ``.npz`` file, so refinement can be re-run without re-fitting rings."""
        np.savez(path, **{name: getattr(self, name) for name in _RING_FIELDS})

    @classmethod
    def load(cls, path: Union[str, Path]) -> "RingPoints":
        """Read points written by :meth:`save`."""
        with np.load(path, allow_pickle=False) as data:
            missing = set(_RING_FIELDS) - set(data.files)
            if missing:
                raise ValueError(f"{path} is not a RingPoints file: missing {sorted(missing)}")
            return cls(**{name: data[name] for name in _RING_FIELDS})
