"""
Read HB-2B reduced (pyRS) project files for pole figures.

A reduced file ``HB2B_<run>.h5`` holds, per sub-run, the motor logs (chi, phi, omega, 2theta), the count time, the
incident slit gaps (ISR:X/Y:Gap), and the reduced spectrum (counts against 2theta). Three kinds of sub-run are
skipped, and counted, so the run table can say so:

* still counting: a file reduced while the run is measured ("live") has a placeholder duration in its last sub-run
  (1000 s or more). Only the last sub-run can be live, so a real count of 1000 s or more elsewhere is kept.
* too short: under 10 % of the median count time (a final sub-run of a few ms after the scan returns home).
* invalid: a non-finite duration or motor log (chi, phi, omega, 2theta).

chi, phi, omega and 2theta always come from the logs, never from the run title. The virtual sample axes vx, vy, vz
(the gauge position in the sample frame) are read when logged. They are NaN when the virtual frame was not set up.

Typical workflow:

1. Read each file with :func:`read_run`.
2. Show :meth:`RunData.describe` in the run table and pass the run to ``polefig.analyze_run``.

Example:
    >>> run_number("HB2B_1234.h5")
    '1234'
    >>> run = read_run("HB2B_1234.h5")  # doctest: +SKIP
    >>> run.describe()  # doctest: +SKIP
    'run 1234: 649 sub-runs, lambda 1.5401 A, omega 56.00, arm 2theta 112.00, slits 2 x 2 mm'
"""

import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .types import FloatArray

# Paths and log names in a reduced (pyRS) project file: the strings of pyrs.dataobjects.constants.HidraConstants.
LOGS_PATH = "raw data/logs"
REDUCED_MAIN_PATH = "reduced diffraction data/main"
REDUCED_2THETA_PATH = "reduced diffraction data/2theta"
WAVELENGTH_PATH = "instrument/monochromator setting/wave length"
DURATION_LOG = "sub-run duration"
SUB_RUNS_LOG = "sub-runs"
POSITION_KEYS = ("vx", "vy", "vz")  # optional logs: the gauge position in the sample (STL) frame, mm

LIVE_PLACEHOLDER_S = 1000.0  # a live file's last (counting) sub-run carries a placeholder duration at or above this
MOTOR_TOL_DEG = 0.01  # omega or the arm 2theta moving more than this during a run is reported
SHORT_FRACTION = 0.1  # sub-runs shorter than this fraction of the median count time are skipped
LOG_KEYS = ("chi", "phi", "omega", "2theta", DURATION_LOG, SUB_RUNS_LOG, "ISR:X:Gap", "ISR:Y:Gap")


@dataclass(kw_only=True)
class RunData:
    """
    One reduced run, finished sub-runs only.

    Attributes:
        run: Run number (from the file name ``HB2B_<run>.h5``, else the file stem).
        path: The file.
        title: Run title from the logs ("" when absent).
        wavelength_A: Wavelength from the file (Å).
        chi: Logged (goniometer) chi per sub-run (deg).
        phi: Logged phi per sub-run (deg).
        omega: Logged omega per sub-run (deg).
        two_theta_arm: Logged detector arm 2theta per sub-run (deg).
        count_s: Count time per sub-run (s).
        subruns: Sub-run numbers.
        slit_x: Incident slit width per sub-run, ISR:X:Gap (mm).
        slit_y: Incident slit height per sub-run, ISR:Y:Gap (mm).
        x: 2theta of the reduced spectra (deg), only bins that are finite in every sub-run kept.
        y: Counts per bin, shape (n_subruns, n_bins).
        live_skipped: Sub-runs left out because they were still counting.
        short_skipped: Sub-runs left out as too short.
        invalid_skipped: Sub-runs left out for a non-finite duration or motor log.
        position: Logged vx, vy, vz per sub-run (mm), shape (n_subruns, 3). NaN where not logged or not defined.
        notes: Things the user should know about the run (omega or the arm moved during the run).
        position_note: Why part of the position is missing (a log absent or of the wrong length, NaN sub-runs).
            Empty when the position is fully logged or not logged at all.
    """

    run: str
    path: Path
    title: str
    wavelength_A: float
    chi: FloatArray
    phi: FloatArray
    omega: FloatArray
    two_theta_arm: FloatArray
    count_s: FloatArray
    subruns: FloatArray
    slit_x: FloatArray
    slit_y: FloatArray
    x: FloatArray
    y: FloatArray
    live_skipped: int
    short_skipped: int
    invalid_skipped: int
    position: FloatArray
    notes: list[str]
    position_note: str

    def _logged_rows(self) -> FloatArray:
        """Sub-runs with all of vx, vy, vz finite."""
        return self.position[np.isfinite(self.position).all(axis=1)]

    @property
    def logged_position(self) -> FloatArray | None:
        """
        Median vx, vy, vz (mm) over the sub-runs that logged all three.

        None means the run has no logged position: no sub-run logged all of vx, vy and vz (the virtual frame was not
        set up, or a log is absent). That is a real state of a run, not a missing value, so callers check for it.
        """
        rows = self._logged_rows()
        if not len(rows):
            return None
        return np.asarray(np.median(rows, axis=0), dtype=float)

    @property
    def position_spread(self) -> float:
        """Largest range of the logged vx, vy or vz over the sub-runs (mm). 0 when not logged."""
        rows = self._logged_rows()
        return float(np.ptp(rows, axis=0).max()) if len(rows) else 0.0

    @property
    def n_subruns(self) -> int:
        """Number of sub-runs kept."""
        return len(self.count_s)

    @property
    def omega_median(self) -> float:
        """Omega used in the geometry: the median of the log (deg)."""
        return float(np.median(self.omega))

    @property
    def arm_two_theta(self) -> float:
        """Detector arm 2theta: the median of the log (deg)."""
        return float(np.median(self.two_theta_arm))

    @property
    def incident_width(self) -> float:
        """Incident beam width: the median of ISR:X:Gap (mm)."""
        return float(np.median(self.slit_x))

    @property
    def incident_height(self) -> float:
        """Incident beam height: the median of ISR:Y:Gap (mm)."""
        return float(np.median(self.slit_y))

    @property
    def live(self) -> bool:
        """True when the file was reduced while the run was still counting (its last sub-run was skipped)."""
        return self.live_skipped > 0

    def describe(self) -> str:
        """One-line summary for the run table: sub-runs, skipped sub-runs, wavelength, omega, arm, slits, notes."""
        skipped = []
        if self.live_skipped:
            skipped.append(f"{self.live_skipped} still counting")
        if self.short_skipped:
            skipped.append(f"{self.short_skipped} too short")
        if self.invalid_skipped:
            skipped.append(f"{self.invalid_skipped} with a non-finite duration or motor log")
        extra = f" ({', '.join(skipped)} skipped)" if skipped else ""
        notes = "".join(f". {n}" for n in self.notes)
        return (f"run {self.run}: {self.n_subruns} sub-runs{extra}, lambda {self.wavelength_A:.4f} A, omega "
                f"{self.omega_median:.2f}, arm 2theta {self.arm_two_theta:.2f}, slits {self.incident_width:g} x "
                f"{self.incident_height:g} mm{notes}")  # fmt: skip


def run_number(path: str | Path) -> str:
    """
    Run number from a file name.

    Args:
        path: The file, e.g. 'HB2B_1234.h5' (case is ignored).

    Returns:
        The digits after 'HB2B_', e.g. '1234', or the file stem for other names.
    """
    m = re.search(r"HB2B_(\d+)", Path(path).name, re.IGNORECASE)
    return m.group(1) if m else Path(path).stem


def read_run(path: str | Path) -> RunData:
    """
    Read a reduced HB-2B (pyRS) file, keeping only the finished sub-runs.

    Args:
        path: The ``HB2B_<run>.h5`` file.

    Returns:
        The run, with the skipped sub-runs counted and notes on omega or the arm moving during the run.

    Raises:
        ValueError: If the file cannot be read, is not a reduced HB-2B file, lacks a needed log, has logs or spectra
            of different lengths or no sub-runs, has a wavelength outside 0.5 to 5 Å, or has no finished sub-runs.
    """
    path = Path(path)
    try:
        with h5py.File(path, "r") as h:
            needed = (LOGS_PATH, REDUCED_MAIN_PATH, REDUCED_2THETA_PATH, WAVELENGTH_PATH)
            absent = [k for k in needed if k not in h]
            if absent:
                raise ValueError(f"{path.name} is not a reduced HB-2B (pyRS) file: no {', '.join(absent)}")
            L = h[LOGS_PATH]
            missing = [k for k in LOG_KEYS if k not in L]
            if missing:
                raise ValueError(f"{path.name} lacks the logs {', '.join(missing)}")
            d = {k: np.asarray(L[k][()], float).reshape(-1) for k in LOG_KEYS}
            x = np.asarray(h[REDUCED_2THETA_PATH][()], float)
            y = np.asarray(h[REDUCED_MAIN_PATH][()], float)
            n_log = _check_lengths(path, d, x=x, y=y)
            pos, notes = _read_position(L, n_log)
            title = _read_title(L)
            lam = float(np.asarray(h[WAVELENGTH_PATH][()], float).reshape(-1)[0])
    except OSError as exc:
        raise ValueError(f"cannot read {path.name}: {exc}") from exc
    if not (np.isfinite(lam) and 0.5 <= lam <= 5.0):
        raise ValueError(f"{path.name}: wavelength {lam} A is not a valid HB-2B wavelength (0.5 to 5 A)")
    x = x[0]
    t = d[DURATION_LOG]
    valid = np.isfinite(t) & np.all([np.isfinite(d[k]) for k in ("chi", "phi", "omega", "2theta")], axis=0)
    live = np.zeros(n_log, bool)
    live[-1] = bool(valid[-1] and t[-1] >= LIVE_PLACEHOLDER_S)  # only the last sub-run can still be counting
    counted = valid & ~live
    med = float(np.median(t[counted])) if counted.any() else 0.0
    short = counted & (t < SHORT_FRACTION * med)
    done = counted & ~short
    if not done.any():
        raise ValueError(f"{path.name} has no finished sub-runs")
    ok = np.isfinite(y[done]).all(axis=0)  # bins finite in every sub-run used
    run_notes = [f"{name} moved by {np.ptp(d[k][done]):.3f} deg during the run (the median is used)"
                 for k, name in (("omega", "omega"), ("2theta", "the arm 2theta"))
                 if np.ptp(d[k][done]) > MOTOR_TOL_DEG]  # fmt: skip
    return RunData(
        run=run_number(path),
        path=path,
        title=title,
        wavelength_A=lam,
        chi=d["chi"][done],
        phi=d["phi"][done],
        omega=d["omega"][done],
        two_theta_arm=d["2theta"][done],
        count_s=t[done],
        subruns=d[SUB_RUNS_LOG][done],
        slit_x=d["ISR:X:Gap"][done],
        slit_y=d["ISR:Y:Gap"][done],
        x=x[ok],
        y=y[done][:, ok],
        live_skipped=int(live.sum()),
        short_skipped=int(short.sum()),
        invalid_skipped=int((~valid).sum()),
        notes=run_notes,
        position=pos[done],
        position_note=_position_note(pos[done], notes),
    )


def _check_lengths(path: Path, logs: dict[str, FloatArray], *, x: FloatArray, y: FloatArray) -> int:
    """
    The number of sub-runs, checked to be the same in every needed log and in the spectra.

    Raises:
        ValueError: If the spectra are not 2D (sub-run x bin), 2theta has no row of the spectra's width, the logs and
            spectra have different lengths, or there are no sub-runs. The message names the file.
    """
    if y.ndim != 2 or x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != y.shape[1]:
        raise ValueError(f"{path.name}: the reduced spectra must be (sub-run, bin) arrays with a 2theta row of the "
                         f"same width, got main {y.shape} and 2theta {x.shape}")  # fmt: skip
    lengths = {k: v.size for k, v in logs.items()} | {"reduced spectra": y.shape[0]}
    if len(set(lengths.values())) != 1:
        shown = ", ".join(f"{k} {n}" for k, n in lengths.items())
        raise ValueError(f"{path.name}: the logs and spectra must have one value per sub-run, got lengths {shown}")
    n = y.shape[0]
    if n == 0:
        raise ValueError(f"{path.name} has no sub-runs")
    return n


def _read_title(logs: h5py.Group) -> str:
    """The run title from the logs: the first value, decoded as UTF-8 (bad bytes replaced), "" when absent."""
    if "run_title" not in logs:
        return ""
    v = np.asarray(logs["run_title"][()]).reshape(-1)
    if not v.size:
        return ""
    first = v[0]
    return first.decode("utf-8", errors="replace") if isinstance(first, bytes) else str(first)


def _read_position(logs: h5py.Group, n_log: int) -> tuple[FloatArray, list[str]]:
    """vx, vy, vz per sub-run (NaN where absent) and notes on logs that are absent or of the wrong length."""
    pos = np.full((n_log, 3), np.nan)
    absent, notes = [], []
    for j, k in enumerate(POSITION_KEYS):
        if k not in logs:
            absent.append(k)
            continue
        v = np.asarray(logs[k][()], float).reshape(-1)  # (n,) or (n, 1)
        if v.size != n_log:
            notes.append(f"{k} log has {v.size} values for {n_log} sub-runs")
        else:
            pos[:, j] = v
    if absent and len(absent) < len(POSITION_KEYS):
        notes.append(f"{', '.join(absent)} not logged")
    return pos, notes


def _position_note(pos: FloatArray, notes: list[str]) -> str:
    """The notes, plus a count of sub-runs without a full position when some sub-runs have one."""
    full = np.isfinite(pos).all(axis=1)
    if full.any() and not full.all():
        notes = [*notes, f"{int((~full).sum())} of {len(full)} sub-runs have no full vx, vy, vz"]
    elif not full.any() and np.isfinite(pos).any() and not notes:
        notes = [*notes, "no sub-run has all of vx, vy, vz"]
    return "; ".join(notes)
