"""
Provenance of a detector calibration: when, by whom and how it was obtained (#1053).

Both calibration methods attach a ``"provenance"`` block to the calibration JSON, beside the values
(``pyrs.utilities.calibration_file_io.write_calibration_to_json``), so a calibration can be judged
and traced after ``HB2B_Latest.json`` has been overwritten. The block holds:

1. a header (:func:`header`): schema version, creation time, PyRS version, user, host, method;
2. the inputs: run and mask files with their SHA-256 (:func:`file_record`), powders, settings;
3. the starting calibration;
4. ``steps``: one :func:`step_record` per refinement stage, in the order run;
5. ``fit_quality`` (:func:`fit_quality`): residual count, degrees of freedom, RMSE, chi^2/dof,
   the residual scatter the errors are scaled by (#1049), and residuals per reflection and run;
6. the error model and the parameters reported undetermined (-2) with the reason (:func:`undetermined`).

See docs/ground_truths.md.
"""

import datetime
import enum
import getpass
import hashlib
import json
import math
import os
import socket
import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

import pyrs
from pyrs.calibration.powder_standards import hkl_label

__all__ = [
    "METHODS",
    "RESIDUAL_DESCRIPTION",
    "SCHEMA_VERSION",
    "assemble",
    "file_record",
    "fit_quality",
    "header",
    "jsonable",
    "reflection_label",
    "safe_build",
    "step_record",
    "undetermined",
]

SCHEMA_VERSION = 1
METHODS = ("peak-fit", "ring")
RESIDUAL_DESCRIPTION = "peak position minus Bragg angle 2*asin(lambda/2d), degrees"
# pixel_refinement.UNDETERMINED; not imported, because pixel_refinement imports this module
_UNDETERMINED = -2.0

# a group of residuals: sub-run, powder, reflection label, and the eta mask (peak-fit) or None (ring)
ResidualLabel = Tuple[int, str, str, Optional[str]]


def _number(value: Any) -> Optional[float]:
    """A JSON-safe float: NaN and infinity become None (``json`` would write non-standard NaN)."""
    value = float(value)
    return value if math.isfinite(value) else None


def header(method: str) -> Dict[str, Any]:
    """Schema version, creation time (ISO 8601 with time zone), PyRS version, user, host and method.

    Raises:
        ValueError: If ``method`` is not one of :data:`METHODS`.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    return {
        "schema": SCHEMA_VERSION,
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "pyrs_version": pyrs.__version__,
        "user": getpass.getuser(),
        "host": socket.gethostname(),
        "method": method,
    }


def file_record(path: Optional[Union[str, os.PathLike]]) -> Optional[Dict[str, Optional[str]]]:
    """Absolute path and SHA-256 of an input file; if it cannot be read, ``sha256`` is None and
    ``sha256_error`` says why.

    Returns:
        None if ``path`` is None (e.g. no mask).
    """
    if path is None:
        return None
    path = os.path.abspath(os.fspath(path))
    try:
        sha = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                sha.update(block)
    except OSError as read_err:
        return {"file": path, "sha256": None, "sha256_error": f"{type(read_err).__name__}: {read_err}"}
    return {"file": path, "sha256": sha.hexdigest()}


def reflection_label(powder: str, structure: Optional[str], hkl_squared: int) -> str:
    """A reflection as ``"<powder> <hkl>"``, e.g. ``"Ni 111"``; ``"<powder> h2+k2+l2=<n>"`` if the
    structure is unknown or has no such reflection (a label never raises)."""
    try:
        return f"{powder} {hkl_label(int(hkl_squared), str(structure))}"
    except (TypeError, ValueError):
        return f"{powder} h2+k2+l2={hkl_squared}"


def step_record(
    step: int,
    routine: str,
    free_parameters: Sequence[str],
    values: Union[Sequence[float], np.ndarray],
    errors: Union[Sequence[float], np.ndarray],
    status: int,
    n: Optional[int],
    rmse: Optional[float],
    nfev: Optional[int],
    message: str = "",
    undetermined_reasons: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """One refinement stage: the routine run, its free parameters and their values and errors after it.

    Args:
        step: 1-based position in the order the stages ran.
        routine: Name of the routine, e.g. ``"CalibrateShift"``.
        free_parameters: Names of the refined parameters.
        values: Their values after the stage.
        errors: Their one-sigma errors (-2: undetermined).
        status: ``scipy.optimize.least_squares`` status (> 0 converged, 0 stopped at ``max_nfev``).
        n: Number of residuals, if known.
        rmse: Residual RMS in degrees, if known.
        nfev: Number of residual evaluations, if known.
        message: The optimizer's termination message.
        undetermined_reasons: Why each undetermined parameter is undetermined.

    Raises:
        ValueError: If ``values`` or ``errors`` do not match ``free_parameters``.
    """
    names = [str(name) for name in free_parameters]
    if len(values) != len(names) or len(errors) != len(names):
        raise ValueError(f"{len(names)} free parameters but {len(values)} values and {len(errors)} errors")
    return {
        "step": int(step),
        "routine": str(routine),
        "free_parameters": names,
        "values": {name: _number(value) for name, value in zip(names, values)},
        "errors": {name: _number(error) for name, error in zip(names, errors)},
        "status": int(status),
        "n": None if n is None else int(n),
        "rmse_deg": None if rmse is None else _number(rmse),
        "nfev": None if nfev is None else int(nfev),
        "message": str(message),
        "undetermined_reasons": dict(undetermined_reasons or {}),
    }


def fit_quality(residual: np.ndarray, n_free: int, labels: Optional[Sequence[ResidualLabel]] = None) -> Dict[str, Any]:
    """Overall fit quality at the final calibration, and residuals per reflection and run.

    Args:
        residual: Final residuals (degrees), one per observation.
        n_free: Number of parameters free in the final stage.
        labels: One :data:`ResidualLabel` per residual, or None to skip the breakdown. If they do not
            match the residuals one to one, the breakdown is left empty and
            ``by_reflection_and_run_unavailable`` says why (it is never guessed).
    """
    residual = np.asarray(residual, dtype=float)
    n = int(residual.size)
    dof = n - int(n_free)
    chi2 = float(np.sum(residual**2) / dof) if dof > 0 else None
    quality: Dict[str, Any] = {
        "residual": RESIDUAL_DESCRIPTION,
        "n": n,
        "n_free_parameters": int(n_free),
        "dof": dof,
        "rmse_deg": _number(np.sqrt(np.mean(residual**2))) if n else None,
        "residual_sum_deg": _number(np.sum(residual)),
        "chi2_per_dof_deg2": None if chi2 is None else _number(chi2),
        "scatter_deg": None if chi2 is None else _number(np.sqrt(chi2)),
        "by_reflection_and_run": [],
    }
    if labels is None:
        return quality
    if len(labels) != n or any(label is None for label in labels):
        missing = sum(1 for label in labels if label is None)
        quality["by_reflection_and_run_unavailable"] = (
            f"{n} residuals but {len(labels)} labels ({missing} missing): residuals and labels do not match"
        )
        return quality
    groups: Dict[ResidualLabel, List[float]] = defaultdict(list)
    for label, value in zip(labels, residual):
        groups[label].append(float(value))

    def order(item: Tuple[ResidualLabel, List[float]]) -> Tuple[int, str, str, str]:
        sub_run, powder, reflection, eta_mask = item[0]
        return int(sub_run), str(powder), str(reflection), str(eta_mask)

    for (sub_run, powder, reflection, eta_mask), values in sorted(groups.items(), key=order):
        entry: Dict[str, Any] = {"sub_run": int(sub_run), "powder": powder, "reflection": reflection}
        if eta_mask is not None:
            entry["eta_mask"] = eta_mask
        entry.update(
            {
                "n": len(values),
                "mean_deg": _number(np.mean(values)),
                "rms_deg": _number(np.sqrt(np.mean(np.square(values)))),
            }
        )
        quality["by_reflection_and_run"].append(entry)
    return quality


def undetermined(
    names: Sequence[str], errors: Union[Sequence[float], np.ndarray], reasons: Optional[Mapping[str, str]] = None
) -> List[Dict[str, str]]:
    """The parameters whose final error is -2 (undetermined), each with its reason if known."""
    reasons = reasons or {}
    return [
        {"parameter": str(name), "reason": reasons.get(str(name), "not recorded")}
        for name, error in zip(names, errors)
        if float(error) == _UNDETERMINED
    ]


def jsonable(value: Any) -> Any:
    """``value`` with every numpy scalar or array, tuple, path and enum turned into plain JSON types,
    and every non-finite number into None, so ``json.dumps`` of a provenance block cannot fail on them."""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _number(value)
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return value


def safe_build(build: Callable[[], Dict[str, Any]], what: str = "calibration provenance") -> Dict[str, Any]:
    """Build a provenance block that can always be written: the block never stops a calibration from
    being written (#1053).

    ``build()`` is called, its result made :func:`jsonable` and test-serialized. If any of that fails,
    a warning is issued and a minimal block saying why is returned instead.
    """
    try:
        block = jsonable(build())
        json.dumps(block, allow_nan=False)
        return block
    except Exception as build_err:  # noqa: BLE001 - recorded in the file, never blocks the write
        message = f"{what} could not be built: {type(build_err).__name__}: {build_err}"
        warnings.warn(message, stacklevel=2)
        return {"schema": SCHEMA_VERSION, "error": message}


def assemble(
    method: str,
    inputs: Mapping[str, Any],
    start_source: Any,
    start_values: Sequence[float],
    steps: Sequence[Mapping[str, Any]],
    quality: Optional[Dict[str, Any]],
    error_model: str,
    names: Sequence[str],
    errors: Union[Sequence[float], np.ndarray],
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """The provenance block of either calibration method, built one way.

    Args:
        method: One of :data:`METHODS`.
        inputs: The run, mask, settings and whatever the caller adds.
        start_source: Where the starting calibration came from (text, or a :func:`file_record`).
        start_values: The starting calibration vector, in ``names`` order.
        steps: :func:`step_record` dicts of the steps since the start; renumbered from 1 here.
        quality: :func:`fit_quality` at the final calibration, or None if unknown.
        error_model: How the errors were computed.
        names: Parameter names of the calibration vector.
        errors: Final errors, in ``names`` order (-2: undetermined).
        extra: Method-specific keys (e.g. ``held_parameters``).
    """
    reasons: Dict[str, str] = {}
    for step in steps:
        reasons.update(step.get("undetermined_reasons", {}))
    block = header(method)
    block["inputs"] = dict(inputs)
    block["start"] = {"source": start_source, "values": {str(n): _number(v) for n, v in zip(names, start_values)}}
    block["steps"] = [dict(step, step=number) for number, step in enumerate(steps, start=1)]
    block["fit_quality"] = quality
    block["error_model"] = error_model
    block.update(dict(extra or {}))
    block["undetermined"] = undetermined(names, errors, reasons)
    return block
