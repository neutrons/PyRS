"""
Closed triangle mesh of the sample and the ray tracer for beam path lengths.

Ported from the reference script (``read_stl``, ``SampleMesh``) with no change to the numerics. The STL frame is the
sample frame (vx, vy, vz) and its units are read as mm.

Typical workflow:

1. Load the sample with :meth:`SampleMesh.from_stl` and show :meth:`SampleMesh.describe` and
   :meth:`SampleMesh.warnings`.
2. :meth:`SampleMesh.check` before tracing (prediction and analysis do this unless told not to).
3. :meth:`SampleMesh.trace` gives the inside test and the path ahead of many points along one direction.

Example:
    >>> cube = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
    >>> faces = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
    ...          (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    >>> mesh = SampleMesh(cube[faces], name="cube")
    >>> mesh.closed, mesh.consistent, mesh.volume
    (True, True, 1.0)
    >>> wind, length = mesh.trace([[0.3, 0.6, 0.25]], [0.0, 0.0, 1.0])
    >>> wind, length
    (array([1.]), array([0.75]))
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

GRID_FACTOR = 1.5  # grid cells per side = GRID_FACTOR * sqrt(triangles in the footprint); fastest over the fixtures


def read_stl(path: str | Path) -> FloatArray:
    """
    Read the triangles of a binary or ASCII STL.

    A file whose size matches the binary layout (84 + 50 n bytes) is read as binary, any other as ASCII.

    Args:
        path: STL file.

    Returns:
        Triangle vertices in the file's frame and units, shape (n, 3, 3).

    Raises:
        ValueError: If the file is neither a valid binary nor a valid ASCII STL.
    """
    data = Path(path).read_bytes()
    if len(data) >= 84:
        n = int(np.frombuffer(data, "<u4", 1, 80)[0])
        if 84 + 50 * n == len(data):
            rec = np.frombuffer(data, np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), n, 84)
            return rec["v"].astype(float)
    tok = data.decode("ascii", "replace").split()
    v = [tok[i + 1 : i + 4] for i, s in enumerate(tok) if s == "vertex"]
    if not v or len(v) % 3:
        raise ValueError(f"{Path(path).name} is not a valid binary or ASCII STL")
    try:
        return np.array(v, float).reshape(-1, 3, 3)
    except ValueError as exc:
        raise ValueError(f"{Path(path).name} is not a valid binary or ASCII STL ({exc})") from exc


class SampleMesh:
    """
    Closed triangle mesh of the sample in the sample frame (vx, vy, vz in mm, the frame of the gauge center).

    Rays are traced with the Moller-Trumbore test against every triangle. For a ray ``p + t d`` (t > 0) each
    crossing adds ``s = sign(n . d)`` (+1 leaving, -1 entering, n the outward normal). The sum of s is 1 for a point
    inside a closed mesh and 0 outside. The sum of ``s t`` is the material path ahead of an inside point, re-entries
    included, so concave shapes, holes and cavities are handled.

    Args:
        triangles: Triangle vertices, shape (n, 3, 3) (mm). Inward winding is detected from the signed volume and
            flipped.
        name: Label used by :meth:`describe` (default: "mesh", usually the file name).

    Attributes:
        v0: First vertex of each triangle.
        e1: Edge from v0 to the second vertex (after any winding flip).
        e2: Edge from v0 to the third vertex (after any winding flip).
        n: Outward normal of each triangle (not unit length).
        volume: Enclosed volume (mm3).
        ntri: Number of triangles.
        name: Label.
        lo: Lower corner of the bounding box (mm).
        hi: Upper corner of the bounding box (mm).
        size: Bounding box size (mm).
        closed: Every edge is shared by exactly two triangles.
        consistent: Every triangle is wound the same way (each directed edge once, and its reverse once).

    Raises:
        ValueError: If there are no triangles.
    """

    def __init__(self, triangles: npt.ArrayLike, *, name: str = "mesh") -> None:
        T = np.asarray(triangles, float).reshape(-1, 3, 3)
        if len(T) == 0:
            raise ValueError(f"the mesh has no triangles, got an array of shape {np.shape(triangles)}")
        if not np.isfinite(T).all():
            bad = int((~np.isfinite(T)).any(axis=(1, 2)).sum())
            raise ValueError(f"{name}: {bad} triangles have non-finite vertex coordinates (inf or nan)")
        e1, e2 = T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]
        vol = float(np.einsum("ij,ij->", T[:, 0], np.cross(T[:, 1], T[:, 2])) / 6)
        if vol < 0:  # inward winding: flip every triangle
            e1, e2, vol = e2, e1, -vol
        self.v0: FloatArray = T[:, 0]
        self.e1: FloatArray = e1
        self.e2: FloatArray = e2
        self.n: FloatArray = np.cross(e1, e2)
        self.volume = vol
        self.ntri = len(T)
        self.name = name
        P = T.reshape(-1, 3)
        self.lo: FloatArray = P.min(axis=0)
        self.hi: FloatArray = P.max(axis=0)
        self.size: FloatArray = self.hi - self.lo
        key = np.round(P, 6)
        idx = np.unique(key, axis=0, return_inverse=True)[1].reshape(-1, 3)
        directed = np.concatenate([idx[:, [0, 1]], idx[:, [1, 2]], idx[:, [2, 0]]])
        edges = np.sort(directed, axis=1)
        self.closed = bool(np.all(np.unique(edges, axis=0, return_counts=True)[1] == 2))
        # consistent winding: every directed edge once, and its reverse once (a few flipped triangles pass the
        # closed test but give wrong crossing signs, so wrong or even negative path lengths)
        uniq = np.unique(directed, axis=0)
        fwd = {tuple(e) for e in uniq.tolist()}
        self.consistent = bool(len(uniq) == len(directed) and all((b, a) in fwd for a, b in fwd))

    @classmethod
    def from_stl(cls, path: str | Path) -> "SampleMesh":
        """
        Load a mesh from an STL file.

        Args:
            path: STL file, in mm in the sample frame.

        Returns:
            The mesh, named after the file.

        Raises:
            ValueError: If the file is not a valid STL or has no triangles.
        """
        return cls(read_stl(path), name=Path(path).name)

    def triangles(self) -> FloatArray:
        """Triangle vertices (n, 3, 3) with outward winding."""
        return np.stack([self.v0, self.v0 + self.e1, self.v0 + self.e2], axis=1)

    def trace(
        self, points: npt.ArrayLike, d: npt.ArrayLike, *, chunk: int = 2_000_000
    ) -> tuple[FloatArray, FloatArray]:
        """
        Trace rays from many points along one direction, with a 2D spatial index.

        Points and triangles are projected on the plane normal to d and binned on a grid. A ray from a point can only
        cross triangles whose projection covers the point, so each point is tested only against the triangles that
        overlap its grid cell. The test for each point-triangle pair is the same as in :meth:`trace_brute` (the
        reference script's tracer), so the hits are the same; it is many times faster on meshes with many triangles.

        Args:
            points: Ray start points, shape (n, 3) (mm).
            d: Ray direction, shape (3,). Any length: the path is in units of ``|d|``, so pass a unit vector for mm.
            chunk: Point-triangle pairs per step of the crossing test (default: 2,000,000). The index of candidate
                pairs is built in full first, so peak memory still grows with the number of pairs.

        Returns:
            The sum of crossing signs per point (1 inside a closed mesh, 0 outside) and the material path ahead of
            each point (meaningful for inside points).
        """
        p = np.atleast_2d(np.asarray(points, float))
        d = np.asarray(d, float)
        n_pts = len(p)
        wind, length = np.zeros(n_pts), np.zeros(n_pts)
        if n_pts == 0:
            return wind, length
        h = np.cross(d, self.e2)
        a = np.einsum("ij,ij->i", self.e1, h)
        b1 = np.cross(d, [1.0, 0, 0] if abs(d[0]) < 0.9 else [0, 1.0, 0])
        b1 /= np.linalg.norm(b1)
        B = np.c_[b1, np.cross(d / np.linalg.norm(d), b1)]
        pp = p @ B  # (n, 2)
        tv = np.stack([self.v0, self.v0 + self.e1, self.v0 + self.e2]) @ B  # (3, m, 2)
        tmin, tmax = tv.min(axis=0), tv.max(axis=0)
        lo, hi = pp.min(axis=0), pp.max(axis=0)
        tol = 1e-9 * (1 + np.abs(pp).max())
        use = (np.abs(a) > 1e-12) & np.all((tmax >= lo - tol) & (tmin <= hi + tol), axis=1)
        tri = np.flatnonzero(use)
        if len(tri) == 0:
            return wind, length
        # grid over the points' footprint: about one cell per triangle, at most 256 x 256
        g = int(np.clip(GRID_FACTOR * np.sqrt(len(tri)), 1, 256))
        span = np.maximum(hi - lo, 1e-9)
        cell = span / g

        def cell_of(x: FloatArray) -> npt.NDArray[np.int64]:
            return np.clip(np.floor((x - lo) / cell).astype(np.int64), 0, g - 1)

        pc = cell_of(pp)
        pid = pc[:, 0] * g + pc[:, 1]
        order = np.argsort(pid, kind="stable")
        counts = np.bincount(pid, minlength=g * g)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        # cells covered by each triangle's projected bounding box (widened by tol so a touching edge counts)
        c0, c1 = cell_of(tmin[tri] - tol), cell_of(tmax[tri] + tol)
        nx, ny = c1[:, 0] - c0[:, 0] + 1, c1[:, 1] - c0[:, 1] + 1
        ncell = nx * ny
        t_rep = np.repeat(np.arange(len(tri)), ncell)
        k = np.arange(len(t_rep)) - np.repeat(np.cumsum(ncell) - ncell, ncell)  # index within each triangle
        cx = c0[t_rep, 0] + k // ny[t_rep]
        cy = c0[t_rep, 1] + k % ny[t_rep]
        tc_cell = cx * g + cy
        keep = counts[tc_cell] > 0
        t_rep, tc_cell, cx, cy = t_rep[keep], tc_cell[keep], cx[keep], cy[keep]
        # drop cells that the projected triangle misses (all four corners outside one edge): long thin triangles at
        # a slant cover far fewer cells than their bounding box
        tp = tv[:, tri[t_rep]]  # (3, k, 2) projected vertices
        x0, y0 = lo[0] + cx * cell[0] - tol, lo[1] + cy * cell[1] - tol
        x1, y1 = x0 + cell[0] + 2 * tol, y0 + cell[1] + 2 * tol
        area2 = (tp[1, :, 0] - tp[0, :, 0]) * (tp[2, :, 1] - tp[0, :, 1]) - (tp[1, :, 1] - tp[0, :, 1]) * (
            tp[2, :, 0] - tp[0, :, 0]
        )
        orient = np.sign(area2)  # +1 counterclockwise in the projection
        inside_all = orient != 0  # an edge-on triangle (zero area) cannot be crossed: drop it
        for e in range(3):
            ax, ay = tp[e, :, 0], tp[e, :, 1]
            bx, by = tp[(e + 1) % 3, :, 0], tp[(e + 1) % 3, :, 1]
            ex, ey = bx - ax, by - ay
            # signed side of each corner times the orientation: < 0 is outside this edge
            out_all = np.ones(len(t_rep), bool)
            for qx, qy in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
                out_all &= orient * (ex * (qy - ay) - ey * (qx - ax)) < -tol * (np.abs(ex) + np.abs(ey))
            inside_all &= ~out_all
        t_rep, tc_cell = t_rep[inside_all], tc_cell[inside_all]
        # point-triangle pairs: every point of the cell with every triangle over it
        npair = counts[tc_cell]
        pair_tri = np.repeat(t_rep, npair)
        j = np.arange(len(pair_tri)) - np.repeat(np.cumsum(npair) - npair, npair)
        pair_pt = order[np.repeat(starts[tc_cell], npair) + j]
        ti = tri[pair_tri]
        v0, e1, e2, hh = self.v0[ti], self.e1[ti], self.e2[ti], h[ti]
        f = 1.0 / a[ti]
        s_t = np.sign(self.n[ti] @ d)
        for k0 in range(0, len(ti), max(1, chunk)):
            sl = slice(k0, k0 + chunk)
            s = p[pair_pt[sl]] - v0[sl]
            u = f[sl] * np.einsum("ij,ij->i", s, hh[sl])
            q = np.cross(s, e1[sl])
            v = f[sl] * (q @ d)
            t = f[sl] * np.einsum("ij,ij->i", q, e2[sl])
            hit = (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
            idx = pair_pt[sl][hit]
            wind += np.bincount(idx, weights=s_t[sl][hit], minlength=n_pts)
            length += np.bincount(idx, weights=s_t[sl][hit] * t[hit], minlength=n_pts)
        return wind, length

    def trace_brute(
        self, points: npt.ArrayLike, d: npt.ArrayLike, *, chunk: int = 2_000_000
    ) -> tuple[FloatArray, FloatArray]:
        """
        The reference script's tracer: every point against every triangle in the ray bundle's footprint.

        Args:
            points: Ray start points, shape (n, 3) (mm).
            d: Ray direction, shape (3,). Any length: the path is in units of ``|d|``, so pass a unit vector for mm.
            chunk: Work-array size limit in points x triangles per step (default: 2,000,000).

        Returns:
            The sum of crossing signs per point (1 inside a closed mesh, 0 outside) and the material path ahead of
            each point (meaningful for inside points).
        """
        p = np.atleast_2d(np.asarray(points, float))
        d = np.asarray(d, float)
        h = np.cross(d, self.e2)
        a = np.einsum("ij,ij->i", self.e1, h)
        use = np.abs(a) > 1e-12  # skip triangles parallel to the ray
        # skip triangles outside the footprint of the ray bundle (projected on the plane normal to d)
        b1 = np.cross(d, [1.0, 0, 0] if abs(d[0]) < 0.9 else [0, 1.0, 0])
        b1 /= np.linalg.norm(b1)
        B = np.c_[b1, np.cross(d / np.linalg.norm(d), b1)]
        pp = p @ B
        tv = np.stack([self.v0, self.v0 + self.e1, self.v0 + self.e2]) @ B  # (3, m, 2)
        tol = 1e-9 * (1 + np.abs(pp).max())
        use &= np.all((tv.max(axis=0) >= pp.min(axis=0) - tol) & (tv.min(axis=0) <= pp.max(axis=0) + tol), axis=1)
        v0, e1, e2, h, f = self.v0[use], self.e1[use], self.e2[use], h[use], 1.0 / a[use]
        s_t = np.sign(self.n[use] @ d)
        wind, length = np.zeros(len(p)), np.zeros(len(p))
        step = max(1, chunk // max(1, len(p)))
        for k in range(0, len(v0), step):
            sl = slice(k, k + step)
            s = p[:, None, :] - v0[None, sl]  # (n, m, 3)
            u = f[sl] * np.einsum("nmk,mk->nm", s, h[sl])
            q = np.cross(s, e1[None, sl])
            v = f[sl] * (q @ d)
            t = f[sl] * np.einsum("nmk,mk->nm", q, e2[sl])
            hit = (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
            wind += (hit * s_t[sl]).sum(axis=1)
            length += (hit * s_t[sl] * t).sum(axis=1)
        return wind, length

    def ray_segments(self, p: npt.ArrayLike, d: npt.ArrayLike) -> list[tuple[float, float]]:
        """
        Material segments ahead of one point along d, from every crossing (for drawing beam paths).

        Args:
            p: Start point, shape (3,) (mm).
            d: Ray direction. The segment ends are in units of ``|d|``.

        Returns:
            ``[(t0, t1), ...]``: each stretch of the ray inside the sample. A start point inside gives t0 = 0 first.
        """
        d = np.asarray(d, float)
        h = np.cross(d, self.e2)
        a = np.einsum("ij,ij->i", self.e1, h)
        use = np.abs(a) > 1e-12
        f = 1.0 / a[use]
        s = np.asarray(p, float) - self.v0[use]
        u = f * np.einsum("ij,ij->i", s, h[use])
        q = np.cross(s, self.e1[use])
        v = f * (q @ d)
        t = f * np.einsum("ij,ij->i", q, self.e2[use])
        hit = (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
        t, sg = t[hit], np.sign(self.n[use][hit] @ d)
        order = np.argsort(t)
        t, sg = t[order], sg[order]
        w, start, segs = sg.sum(), 0.0, []  # winding at p: 1 inside, 0 outside
        for ti, si in zip(t, sg, strict=True):
            if w > 0.5 >= w - si:
                segs.append((start, float(ti)))
            elif w <= 0.5 < w - si:
                start = float(ti)
            w -= si
        return segs

    def contains(self, p: npt.ArrayLike) -> npt.NDArray[np.bool_]:
        """
        True for points inside the mesh (winding test along an oblique direction).

        Args:
            p: Points, shape (n, 3) or (3,) (mm).

        Returns:
            Inside flag per point.
        """
        wind, _ = self.trace(p, np.array([0.5773502691896258, 0.5773502691896258, 0.5773502691896258]))
        return wind > 0.5

    def describe(self) -> str:
        """One-line summary: triangles, closed, volume and bounds."""
        return (
            f"{self.name}: {self.ntri} triangles, {'closed' if self.closed else 'NOT closed'}, "
            f"volume {self.volume:.2f} mm3, bounds vx {self.lo[0]:g} to {self.hi[0]:g}, vy {self.lo[1]:g} to "
            f"{self.hi[1]:g}, vz {self.lo[2]:g} to {self.hi[2]:g} mm"
        )

    def warnings(self) -> list[str]:
        """
        Problems the user should know about before trusting path lengths.

        Returns:
            Messages for an open mesh, inconsistent winding, or a size below 1 or above 1000 (wrong units).
        """
        out = []
        if not self.closed:
            out.append(
                "The STL is not closed (an edge is not shared by exactly two triangles), so inside tests and path "
                "lengths can be wrong."
            )
        elif not self.consistent:
            out.append(
                "The STL's triangles are not all wound the same way (some face inward), so inside tests and path "
                "lengths can be wrong. Repair the normals in the CAD program."
            )
        if self.size.max() < 1.0 or self.size.max() > 1000.0:
            out.append(f"The STL is {self.size.max():g} across. Lengths are read as mm: check the units.")
        return out

    def check(self) -> None:
        """
        Check that the mesh gives correct path lengths.

        Raises:
            ValueError: If the mesh is not closed or its winding is inconsistent.
        """
        if not (self.closed and self.consistent):
            raise ValueError(f"{self.name}: {self.warnings()[0]} Pass allow_open_mesh=True to use it anyway.")
