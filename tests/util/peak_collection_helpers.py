from collections.abc import Callable, Generator
import numpy as np

from pyrs.core.peak_profile_utility import get_parameter_dtype
from pyrs.core.workspaces import HidraWorkspace
from pyrs.peaks import FitEngineFactory as PeakFitEngineFactory  # type: ignore
from pyrs.peaks.peak_collection import PeakCollection
from pyrs.peaks.peak_fit_engine import FitResult

import pytest

RNG = np.random.default_rng(seed=0x923F109B1D944AF5)


@pytest.fixture
def createPeakCollection() -> Generator[Callable[..., PeakCollection]]:
    # This fixture generates a `PeakCollection` instance initialized using random values.

    def _init(
        *,
        peak_tag: str,
        peak_profile: str,
        background_type: str,
        wavelength: float,
        projectfilename: str,
        runnumber: int,
        N_subrun: int,
        exclude_list=None,
        N_counts=1000,  # range for random counts
        N_span=10000.0,  # domain for random axes
        error_fraction=0.01,  # fractional error for various initializations
    ) -> PeakCollection:
        peaks = PeakCollection(
            peak_tag,
            peak_profile,
            background_type,
            wavelength=wavelength,
            projectfilename=projectfilename,
            runnumber=runnumber,
        )

        """
        # Grab some random indices from somewhere in the middle of the permutations sequence.
        all_runs = [n for n in range(3 * N_subrun)]
        subruns = next(islice(permutations((n for n in range(3 * N_subrun)), N_subrun), 2 * N_subrun, 2 * N_subrun + 1))
        """
        # Assume subruns are supposed to be in order. Why would that be the case?
        subruns = [n + 1 for n in range(N_subrun)]

        # Ensure that the parameter values are somewhat physically meaningful:
        #   for example, no negative peak widths or out-of-range mixing fractions.
        params = peaks._peak_profile.native_parameters
        dtypes = dict(get_parameter_dtype(peaks._peak_profile, peaks._background_type))
        param_values = np.zeros(N_subrun, list(dtypes.items()))
        param_errors = np.zeros(N_subrun, list(dtypes.items()))
        for param in params:
            dtype = dtypes[param]
            match param:
                case "Height" | "Intensity":
                    vs = RNG.uniform(0.0, N_counts, size=(N_subrun,)).astype(dtype)
                    es = RNG.uniform(0.0, error_fraction * N_counts, size=(N_subrun,)).astype(dtype)
                case "PeakCentre":
                    vs = RNG.uniform(0.0, N_span, size=(N_subrun,)).astype(dtype)
                    es = RNG.uniform(0.0, error_fraction * N_span, size=(N_subrun,)).astype(dtype)
                case "Sigma" | "FWHM":
                    vs = RNG.uniform(0.0, N_span / 10.0, size=(N_subrun,)).astype(dtype)
                    es = RNG.uniform(0.0, error_fraction * N_span / 10.0, size=(N_subrun,)).astype(dtype)
                case "Mixing":
                    vs = RNG.uniform(0.0, 1.0, size=(N_subrun,)).astype(dtype)
                    es = RNG.uniform(0.0, error_fraction * 1.0, size=(N_subrun,)).astype(dtype)
                case _:
                    raise RuntimeError(f"`createPeakCollection`: unexpected param '{param}'")

            param_values[param] = vs
            param_errors[param] = es

        fit_costs = RNG.uniform(0.0, 100.0, size=(N_subrun,)).astype(dtype)

        peaks.set_peak_fitting_values(subruns, param_values, param_errors, fit_costs, exclude_list)
        return peaks

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
