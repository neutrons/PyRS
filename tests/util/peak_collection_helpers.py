from itertools import islice, permutations
import numpy as np

from pyrs.core.peak_profile_utility import EFFECTIVE_PEAK_PARAMETERS, get_parameter_dtype
from pyrs.peaks.peak_collection import PeakCollection

import pytest

RNG = np.random.default_rng(seed=0x923f109b1d944af5)

@pytest.fixture
def createPeakCollection(*, 
    peak_tag: str,
    peak_profile,
    background_type,
    wavelength: float,
    projectfilename: str,
    runnumber: int,
    N_subrun: int,
    exclude_list = None,
    N_counts = 1000, # range for random counts
    N_span = 10000.0, # domain for random axes
    error_fraction = 0.01 # fractional error for various initializations
    ):
    # This fixture generates a `PeakCollection` instance initialized using
    #   somewhat physical, but random, values.
    
    def _init() -> PeakCollection:
        peaks = PeakCollection(
            peak_tag, peak_profile, background_type,
            wavelength=wavelength,
            projectfilename=projectfilename,
            runnumber=runnumber
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
        param_values = np.zeros(N_subrun, dtypes.items())
        for param in params:
            match param:
                case 'Height' | 'Intensity':
                    vs = RNG.random(0.0, N_counts, size=(N_subrun,))
                    es = RNG.random(0.0, error_fraction * N_counts, size=(N_subrun,))
                case 'PeakCentre':
                    vs = RNG.random(0.0, N_span, size=(N_subrun,))
                    es = RNG.random(0.0, error_fraction * N_span, size=(N_subrun,))
                case 'Sigma' | 'FWHM':
                    vs = RNG.random(0.0, N_span / 10.0, size=(N_subrun,))
                    es = RNG.random(0.0, error_fraction * N_span / 10.0, size=(N_subrun,))
                case 'Mixing':
                    vs = RNG.random(0.0, 1.0, size=(N_subrun,))
                    es = RNG.random(0.0, error_fraction * 1.0, size=(N_subrun,))
                case _:
                    raise RuntimeError(f"`createPeakCollection`: unexpected param '{param}'")

            param_values[param] = vs
            param_errors[param] = es
        fit_costs = RNG.random(0.0, 100.0, size=(N_subrun,))
            
        peaks.set_peak_fitting_values(
            subruns,
            parameter_values,
            parameter_errors,
            fit_costs,
            exclude_list
        )
        return peaks
        
    yield _init
    
    # teardown follows
    pass

