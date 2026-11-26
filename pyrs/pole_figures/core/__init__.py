"""
Physics core: pure numpy/scipy/h5py/gemmi, no GUI or plotting imports.

The "reference script" in the docstrings is the original HB-2B pole-figure script (``pole_figures.py``) this code
was ported from and is tested against. "As the reference script" means the numerics are unchanged from it. The
parity tests compare with its outputs.

Frames and units: the sample frame is the STL frame with the virtual axes vx, vy, vz (mm). The cradle (instrument)
frame is described in :mod:`.geometry`. Angles are in degrees, wavelengths in Å, mu in 1/mm and rates in counts/s.

Typical workflow:

1. Load the sample (:mod:`.stl_mesh`) and the phase (:mod:`.phase`), and set the mount (:mod:`.geometry`).
2. Predict the corrections with no data (:mod:`.prediction`) and plan table scans (:mod:`.table_scan`).
3. Read the runs (:mod:`.runs`) and build the corrected pole figures (:mod:`.polefig`, :mod:`.combine`).

Example:
    >>> from pyrs.pole_figures.core.geometry import Mount, MountConventions
    >>> Mount(MountConventions(), omega=45.0, tth=90.0).pole(0.0, 0.0)
    array([0., 0., 1.])
"""
