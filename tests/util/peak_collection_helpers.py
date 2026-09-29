from collections.abc import Callable, Generator, Sequence
import numpy as np

from pyrs.core.peak_profile_utility import get_parameter_dtype
from pyrs.core.workspaces import HidraWorkspace
from pyrs.peaks import FitEngineFactory as PeakFitEngineFactory  # type: ignore
from pyrs.dataobjects.constants import HidraConstants
from pyrs.peaks.peak_collection import PeakCollection
from pyrs.peaks.peak_fit_engine import FitResult

import pytest

# Fixed literal seed for full reproducibility:
# Each consuming RNG is re-instantiated fresh inside the fixture
# below (not held at module scope) so that no test's random draws depend on how many draws
# earlier tests happened to consume from a shared stream.
_SEED = 0x923F109B1D944AF5


def make_peak_collection(
    *,
    rng: np.random.Generator,
    peak_tag: str,
    peak_profile: str,
    background_type: str,
    wavelength: float,
    projectfilename: str,
    runnumber: int,
    N_subrun: int,
    sub_runs: Sequence[int] | None = None,
    mask: str = HidraConstants.DEFAULT_MASK,
    exclude_list: Sequence[int] | None = None,
    N_counts=1000,  # range for random counts
    N_span=10000.0,  # domain for random axes
    error_fraction_min=0.005,  # minimum fractional error (0.5%), as a fraction of the value drawn
    error_fraction_max=0.05,  # maximum fractional error (5%), as a fraction of the value drawn
) -> PeakCollection:
    """A `PeakCollection` filled with random but physically plausible fit values.

    The plain-function form of the `createPeakCollection` fixture, for callers
    outside pytest (e.g. scripts under `tests/scripts/`). Pass
    `np.random.default_rng(_SEED)` for the fixture's reproducible draws.
    """
    if not (0 < error_fraction_min <= error_fraction_max):
        raise ValueError(
            "make_peak_collection: invalid error_fraction bounds "
            f"(error_fraction_min={error_fraction_min}, error_fraction_max={error_fraction_max}); "
            "require 0 < error_fraction_min <= error_fraction_max"
        )

    peaks = PeakCollection(
        peak_tag,
        peak_profile,
        background_type,
        wavelength=wavelength,
        projectfilename=projectfilename,
        runnumber=runnumber,
        mask=mask,
    )

    # `sub_runs` overrides the default 1..N: a multi-workspace NXstress entry
    # needs its inputs to cover *disjoint* scan points, which 1..N cannot express.
    #
    # They must be strictly increasing, and that is not this helper's rule to
    # relax: `SubRuns.set` (pyrs/dataobjects/sample_logs.py) rejects anything
    # else outright -- duplicates included, since it tests `value[:-1] < value[1:]`.
    # The reason is that every per-scan-point array is paired with this axis
    # *positionally*, so a different order silently reattributes values rather
    # than failing. No check is repeated here: `SubRuns.set` names the real
    # problem, and the length check below is the one thing it cannot see.
    if sub_runs is not None:
        subruns = [int(n) for n in sub_runs]
        if len(subruns) != N_subrun:
            raise ValueError(f"make_peak_collection: sub_runs has {len(subruns)} entries but N_subrun is {N_subrun}")
    else:
        subruns = [n + 1 for n in range(N_subrun)]

    # Ensure that the parameter values are somewhat physically meaningful:
    #   for example, no negative peak widths or out-of-range mixing fractions.
    # Uncertainties are drawn as a bounded, non-degenerate *fraction of the value itself*
    #   (never as an absolute range independent of the value): this keeps the ratio between
    #   any two parameters' relative uncertainties bounded by error_fraction_max/error_fraction_min,
    #   which is what actually bounds the condition number of downstream error-propagation
    #   formulas that recombine several parameters' uncertainties (see e.g.
    #   pyrs/utilities/NXstress/_fit.py's PseudoVoigt Intensity-from-Height inversion, which
    #   subtracts comparable-magnitude terms and would otherwise amplify float32 rounding
    #   noise without bound whenever a draw happened to make one parameter's absolute
    #   uncertainty tiny relative to its value).
    params = peaks._peak_profile.native_parameters
    dtypes = dict(get_parameter_dtype(peaks._peak_profile, peaks._background_type))
    param_values = np.zeros(N_subrun, list(dtypes.items()))
    param_errors = np.zeros(N_subrun, list(dtypes.items()))
    for param in params:
        dtype = dtypes[param]
        match param:
            case "Height" | "Intensity":
                vs = rng.uniform(0.0, N_counts, size=(N_subrun,)).astype(dtype)
            case "PeakCentre":
                vs = rng.uniform(0.0, N_span, size=(N_subrun,)).astype(dtype)
            case "Sigma" | "FWHM":
                vs = rng.uniform(0.0, N_span / 10.0, size=(N_subrun,)).astype(dtype)
            case "Mixing":
                vs = rng.uniform(0.0, 1.0, size=(N_subrun,)).astype(dtype)
            case _:
                raise RuntimeError(f"`make_peak_collection`: unexpected param '{param}'")

        fraction = rng.uniform(error_fraction_min, error_fraction_max, size=(N_subrun,))
        es = (fraction * vs).astype(dtype)

        param_values[param] = vs
        param_errors[param] = es

    fit_costs = rng.uniform(0.0, 100.0, size=(N_subrun,)).astype(dtype)

    peaks.set_peak_fitting_values(subruns, param_values, param_errors, fit_costs, exclude_list)
    return peaks


@pytest.fixture
def createPeakCollection() -> Generator[Callable[..., PeakCollection]]:
    # This fixture generates a `PeakCollection` instance initialized using random values.
    # A fresh, identically-seeded RNG is created per test invocation (see `_SEED` above):
    # every test that requests this fixture gets the same deterministic sequence of draws,
    # regardless of what other tests ran before it in the same session.
    rng = np.random.default_rng(seed=_SEED)

    def _init(**kwargs) -> PeakCollection:
        return make_peak_collection(rng=rng, **kwargs)

    yield _init

    # teardown follows
    pass


def generate_FitResults_from_workspace(hidra_ws: HidraWorkspace, fit_dic: dict = {}) -> list[FitResult]:
    """
    You can use file tests/data/3393_PWHT-TD.h5  with fit_dic={"0": {"peak_range": [87.599, 91.569], "peak_label": "Peak0", "d0": 1.08}, "1": {"peak_range": [93.544, 95.89], "peak_label": "Peak1", "d0": 1.03}}
    """
    fit_results = []
    fit_engine = PeakFitEngineFactory.getInstance(
        hidraworkspace=hidra_ws,
        peak_function_name="PseudoVoigt",
        background_function_name="Linear",
        wavelength=hidra_ws.get_wavelength(True, True),
    )

    for peak in fit_dic.keys():
        print("Fitting data")
        print("peak_tag: {}".format(fit_dic[peak]["peak_label"]))
        print("x_min: {}".format(fit_dic[peak]["peak_range"][0]))
        print("x_max: {}".format(fit_dic[peak]["peak_range"][1]))
        print("")

        result = fit_engine.fit_peaks(
            peak_tag=fit_dic[peak]["peak_label"],
            x_min=fit_dic[peak]["peak_range"][0],
            x_max=fit_dic[peak]["peak_range"][1],
        )

        d0 = fit_dic[peak].get("d0")
        if d0 is not None:
            for pc in result.peakcollections:
                pc.set_d_reference(values=float(d0), errors=0.0)

        fit_results.append(result)

    return fit_results
