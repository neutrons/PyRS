"""
Worker processes for independent pieces of work, such as the Monte Carlo orientations.

Each orientation's intensity factor is computed exactly as in the serial loop, so the results are identical. Small
jobs stay serial: starting the processes costs a few seconds. No Qt here; a GUI calls this from a worker thread.

Typical workflow:

1. :func:`intensity_factors` picks the number of processes with :func:`n_workers` (or the ``HIDRA_PF_WORKERS``
   environment variable).
2. :func:`run_chunks` starts spawned workers at low priority and hands back each chunk's result as it finishes.

Example:
    >>> n_workers(1e6)
    1
    >>> rows = intensity_factors(m, chi=chi, phi=phi, points=pts, volume=vol, mesh=mesh,
    ...                          center=(0.0, 0.0, 0.0), mu=0.1)  # doctest: +SKIP
"""

import ctypes
import multiprocessing
import os
import pickle
import sys
import threading
import warnings
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from .gauge import IntensityFactor, intensity_factor
from .geometry import Mount
from .stl_mesh import SampleMesh
from .types import CountProgress, FloatArray, no_count_progress

MAX_WORKERS = 6  # gentle on a desktop or a shared analysis node; HIDRA_PF_WORKERS overrides (1 = serial)
#: orientations x Monte Carlo points x triangles below this run serially (about 8 s of tracing on a desktop)
MIN_PARALLEL_WORK = 3e9

_ONE_THREAD = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
_STATE: dict[str, Any] = {}
#: worker_start changes process-wide state (environment, __main__): one job at a time may do so
_START_LOCK = threading.Lock()


def usable_cpus() -> int:
    """Cores this process may run on (CPU affinity or cgroup limits on a shared node), else all cores."""
    if hasattr(os, "process_cpu_count"):  # Python 3.13+
        return os.process_cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0)) or 1
    return os.cpu_count() or 1


Workers = int | Literal["auto"]


def n_workers(work: float, *, requested: Workers = "auto", min_work: float = MIN_PARALLEL_WORK) -> int:
    """
    Processes to use for a job of the given size.

    Args:
        work: Size of the job: orientations x Monte Carlo points x triangles.
        requested: A positive whole number to use as it is, or "auto" (default): the HIDRA_PF_WORKERS environment
            variable (a positive whole number), else 1 below ``min_work``, else up to MAX_WORKERS and at most half
            the cores this process may use (the workers run at low priority).
        min_work: Jobs smaller than this run serially (default: ``MIN_PARALLEL_WORK``).

    Returns:
        The number of processes, 1 for serial.

    Raises:
        ValueError: If requested is neither "auto" nor a positive whole number.
    """
    if requested != "auto":
        if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
            raise ValueError(f"workers must be a positive whole number or 'auto', got {requested!r}")
        return requested
    env = os.environ.get("HIDRA_PF_WORKERS", "").strip()
    if env:
        if env.isascii() and env.isdigit() and int(env) > 0:  # isdigit alone accepts "²", which int() refuses
            return int(env)
        warnings.warn(f"HIDRA_PF_WORKERS={env!r} is not a positive whole number; choosing automatically",
                      stacklevel=2)  # fmt: skip
    if work < min_work:
        return 1
    return max(1, min(MAX_WORKERS, usable_cpus() // 2))


def lower_priority() -> None:
    """Run this process below normal priority, so interactive work on the machine comes first."""
    if sys.platform == "win32":
        below_normal = 0x00004000
        # a failed call returns 0 and leaves normal priority: the results are the same, only less polite
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), below_normal)
        return
    try:
        os.nice(10)
    except OSError:  # not permitted here (e.g. a restricted container): normal priority gives the same results
        pass


@contextmanager
def worker_start() -> Iterator[None]:
    """
    Settings for starting worker processes (they inherit them): no main-module import, one numpy thread each.

    A spawned process imports the parent's main module (by its spec or file). For a GUI that is the whole Qt app,
    loaded in every worker before it can start, one after another, so the main module is hidden meanwhile. Each
    worker's numpy is limited to one thread: a dozen workers each starting a thread per core slow each other down.
    The changes are process-wide, so only one job at a time makes them (a lock), and they are always undone.
    """
    with _START_LOCK:
        with _hidden_main():
            yield


@contextmanager
def _hidden_main() -> Iterator[None]:
    old_env = {k: os.environ.get(k) for k in _ONE_THREAD}
    os.environ.update(dict.fromkeys(_ONE_THREAD, "1"))
    main = sys.modules.get("__main__")
    saved: dict[str, Any] = {}
    if main is not None:
        saved = {k: main.__dict__[k] for k in ("__spec__", "__file__") if k in main.__dict__}
        main.__dict__["__spec__"] = None
        main.__dict__.pop("__file__", None)
    try:
        yield
    finally:
        if main is not None:
            main.__dict__.pop("__spec__", None)
            main.__dict__.update(saved)
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def run_chunks(chunks: Sequence[Any], *, init: Callable[[bytes], None], task: Callable[[Any], Any], payload: bytes,
               workers: int, on_done: Callable[[Any, Any], None]) -> None:  # fmt: skip
    """
    Run ``task(chunk)`` for every chunk in spawned worker processes.

    Args:
        chunks: The pieces of work, each picklable.
        init: Called once in each worker with ``payload`` (module-level, so it pickles).
        task: Called in a worker for each chunk (module-level, so it pickles).
        payload: Pickled data every worker needs (sent once per worker, not per chunk).
        workers: Number of processes.
        on_done: ``on_done(chunk, result)``, called in this process as chunks finish, in completion order. If it
            raises (e.g. the user cancelled), the remaining chunks are dropped and the exception is passed on.
    """
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=init, initargs=(payload,)) as pool:
        with worker_start():  # processes start as tasks are submitted
            futures = {pool.submit(task, c): c for c in chunks}
        try:
            for fut in as_completed(futures):
                on_done(futures[fut], fut.result())
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise


def _init_if(payload: bytes) -> None:
    lower_priority()
    _STATE["if"] = pickle.loads(payload)


def _run_if(chunk: list[tuple[int, float, float]]) -> list[tuple[int, IntensityFactor]]:
    m, pts, vol, mesh, center, mu = _STATE["if"]
    return [(k, intensity_factor(m, chi=c, phi=f, points=pts, volume=vol, mesh=mesh, center=center, mu=mu))
            for k, c, f in chunk]  # fmt: skip


def intensity_factors(
    m: Mount,
    *,
    chi: npt.ArrayLike,
    phi: npt.ArrayLike,
    points: FloatArray,
    volume: float,
    mesh: SampleMesh,
    center: npt.ArrayLike,
    mu: float,
    progress: CountProgress = no_count_progress,
    workers: Workers = "auto",
) -> list[IntensityFactor]:
    """
    ``gauge.intensity_factor`` at every (chi, phi), in order; in parallel processes when the job is large.

    Args:
        m: Beams and rotation conventions.
        chi: Cradle chi of each orientation (deg).
        phi: Phi of each orientation (deg), same length as chi.
        points: Gauge points from ``gauge.gauge_points`` (cradle frame, mm).
        volume: Gauge volume (mm3).
        mesh: Sample in its own frame (mm).
        center: Gauge center in the sample frame (mm).
        mu: Linear attenuation coefficient (1/mm).
        progress: ``progress(n_done)``, called as orientations finish. It may raise to stop the job.
        workers: Number of processes, or "auto" (default) to choose from the job size (:func:`n_workers`).

    Returns:
        One result per orientation, in the order of chi and phi.

    Raises:
        ValueError: If workers is neither "auto" nor a positive whole number.
        RuntimeError: If an orientation returned no result.
    """
    pts, vol = points, volume
    chi = np.asarray(chi, float).ravel()
    phi = np.asarray(phi, float).ravel()
    n = len(chi)
    w = n_workers(float(n) * len(pts) * mesh.ntri, requested=workers)
    out: list[IntensityFactor | None] = [None] * n
    if w <= 1 or n < 2 * w:
        for k in range(n):
            out[k] = intensity_factor(m, chi=float(chi[k]), phi=float(phi[k]), points=pts, volume=vol, mesh=mesh,
                                      center=center, mu=mu)  # fmt: skip
            progress(k + 1)
        return _complete(out)
    jobs = [(k, float(chi[k]), float(phi[k])) for k in range(n)]
    size = max(1, min(40, n // (6 * w)))
    chunks = [jobs[j : j + size] for j in range(0, n, size)]
    done = 0

    def on_done(_chunk: Any, result: list[tuple[int, IntensityFactor]]) -> None:
        nonlocal done
        for k, r in result:
            out[k] = r
        done += len(result)
        progress(done)

    payload = pickle.dumps((m, pts, vol, mesh, np.asarray(center, float), mu))
    run_chunks(chunks, init=_init_if, task=_run_if, payload=payload, workers=w, on_done=on_done)
    return _complete(out)


def _complete(out: list[IntensityFactor | None]) -> list[IntensityFactor]:
    """The results in order. Raises if any orientation has none (a shorter list would misalign chi and phi)."""
    missing = sum(r is None for r in out)
    if missing:
        raise RuntimeError(f"{missing} of {len(out)} orientations returned no result")
    return [r for r in out if r is not None]
