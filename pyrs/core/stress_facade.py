from mantid.api import IMDHistoWorkspace
import numpy as np
from numpy.testing import assert_allclose
from typing import Dict, List, Optional, Union

from pyrs.core.peak_profile_utility import EFFECTIVE_PEAK_PARAMETERS
from pyrs.dataobjects.fields import ScalarFieldSample, StrainField, StrainFieldSingle, StressField, StressType


class StressFacade:

    def __init__(self, stress: StressField) -> None:
        r"""

        Parameters
        ----------
        stress: ~pyrs.dataobjects.fields.StressField
        """
        assert isinstance(stress, StressField)
        self._stress = stress
        self._selection: Optional[str] = None
        self._strain_cache: Dict[str, Union[StrainFieldSingle, StrainField]] = {}  # cache of strain references
        self._stress_cache: Dict[str, Optional[ScalarFieldSample]] = {}  # cache of stress references
        self._update_caches()
        self._d_reference: Optional[ScalarFieldSample] = None
        self._directions: Dict[str, str] = {}

    def _update_caches(self) -> None:
        r"""Update the strain and stress references for each direction and run number"""
        # Update stress cache
        self._stress_cache = {'11': self._stress.stress11, '22': self._stress.stress22, '33': self._stress.stress33}
        # Update strain cache
        self._strain_cache = {'11': self._stress.strain11, '22': self._stress.strain22, '33': self._stress.strain33}
        for direction in ('11', '22', '33'):
            strain = self._strain_cache[direction]
            for peak_collection, strain in zip(strain.peak_collections, strain.strains):
                self._strain_cache[str(peak_collection.runnumber)] = strain

    @property
    def selection(self) -> Optional[str]:
        r"""Pick a scanning direction or run number"""
        return self._selection

    @selection.setter
    def selection(self, choice: str) -> None:
        if len(choice) == 2:
            assert choice in ('11', '22', '33')
        else:
            assert choice in self._all_runs()
        self._selection = choice

    @property
    def direction(self) -> str:
        r"""Report the direction associated to the current selection"""
        # if the current selection is a direction, then return the selection
        assert self._selection is not None, 'A selection has not been made'
        if self._selection in ('11', '22', '33'):
            return self._selection
        # the current selection is a run number
        if bool(self._directions) is False:  # initialize empty dictionary
            for ii in ('11', '22', '33'):
                self._directions.update({run: ii for run in self.runs(ii)})
        return self._directions[self._selection]

    @property
    def x(self) -> np.ndarray:
        return self._stress.x

    @property
    def y(self) -> np.ndarray:
        return self._stress.y

    @property
    def z(self) -> np.ndarray:
        return self._stress.z

    @property
    def d_reference(self) -> ScalarFieldSample:
        r"""
        Consensus d_reference by probing  strains from different directions

        Example:
            vx                 :   0.0  1.0  2.0  3.0  4.0  5.0  6.0  7.0
            d_ref from strain11:   1.0  1.1  1.1  1.2  1.2  1.2  nan  nan
            d_ref from strain22:   nan  1.1  1.1  1.2  1.2  nan  nan  nan
            d_ref from strain33:   nan  nan  1.1  1.2  1.2  1.2  1.2  1.3
            consensus d_ref    :   1.0  1.1  1.1  1.2  1.2  1.2  1.2  1.3

        Returns
        -------
        ~pyrs.dataobjects.fields.ScalarFieldSample
        """
        if self._d_reference is None:  # initialize from the strains along each direction
            # Do we probe two or three directions?
            strains = [self._stress.strain11, self._stress.strain22]
            if self._stress.stress_type is not StressType.IN_PLANE_STRAIN:
                strains.append(self._stress.strain33)

            # Collect reference spacings from each valid direction
            d0s = [strain.get_d_reference() for strain in strains]

            # Ensure reference spacings along different directions coincide where is not nan
            d0s_values = [d0.values for d0 in d0s]
            for d0_ii in d0s_values[:-1]:
                for d0_jj in d0s_values[1:]:
                    mask = ~(np.isnan(d0_ii) | np.isnan(d0_ii))
                    assert_allclose(d0_ii[mask], d0_jj[mask],
                                    err_msg='reference spacings are different on different directions')

            # "merge" reference spacings along different directions where is not nan
            d0_values = np.sum([np.nan_to_num(d0.values) for d0 in d0s], axis=0) / 3.0
            d0_values[d0_values < 1e-9] = np.nan  # revert zeros to nan
            d0_errors = np.sum([np.nan_to_num(d0.errors) for d0 in d0s], axis=0) / 3.0  # they're the same

            # build the consensus scalar field
            self._d_reference = ScalarFieldSample('d-reference', d0_values, d0_errors, self.x, self.y, self.z)
        return self._d_reference

    @property
    def strain(self) -> ScalarFieldSample:
        r"""Scalar field sample with strain values for the selected direction or run number"""
        assert self._selection is not None, 'No selection has been entered'
        return self._strain_cache[self._selection].field

    @property
    def stress(self) -> ScalarFieldSample:
        r"""Scalar field sample with stress values for the selected direction or run number"""
        if self._selection not in ('11', '22', '33'):
            raise ValueError(f'Stress can only be computed for directions, not for run numbers')
        stress = self._stress_cache[self._selection]
        assert stress is not None, 'StressField has not been initialized'
        return stress

    def _all_runs(self) -> List[str]:
        r"""
        All run numbers contributing to the stress

        Returns
        -------
        list
        """
        run_lists = [self.runs(direction) for direction in ('11', '22', '33')]
        return [run for run_list in run_lists for run in run_list]

    def runs(self, direction: str) -> List[str]:
        r"""
        List of run numbers for a particular direction

        Parameters
        ----------
        direction: str
            One of '11', '22', '33'

        Returns
        -------
        list
        """
        # For in-plane measurements, we have no associated runs to direction '33'
        if self._stress.stress_type != StressType.DIAGONAL and direction == '33':
            return []
        self._stress.select(direction)
        assert self._stress.strain is not None, 'The StressField object has not been initialized'
        return [str(peak_collection.runnumber) for peak_collection in self._stress.strain.peak_collections]

    @property
    def youngs_modulus(self) -> float:
        return self._stress.youngs_modulus

    @property
    def poisson_ratio(self) -> float:
        return self._stress.poisson_ratio

    @property
    def peak_parameters(self) -> List[str]:
        r"""
        List of effective peak parameter names

        Returns
        -------
        list
        """
        return EFFECTIVE_PEAK_PARAMETERS

    def peak_parameter(self, query: str) -> ScalarFieldSample:
        r"""
        Peak parameter values for the selection direction or run number

        Parameters
        ----------
        query

        Returns
        -------
        ~pyrs.dataobjects.fields.ScalarFieldSample
        """
        if self._selection in ('11', '22', '33'):
            msg = 'Peak parameters can only be retrieved for run numbers, not directions. Update your selection'
            raise ValueError(msg)
        assert query in self.peak_parameters, f'Peak parameter must be one of {self.peak_parameters}'
        return self._strain_cache[self._selection].get_effective_peak_parameter(query)

    def workspace(self, query: str) -> IMDHistoWorkspace:
        r"""
        Create an MDHistoWorkspace for the selection strain, stress, or effective peak parameter

        Parameters
        ----------
        query: str
            One of 'strain', 'stress', or one of the effective peak parameter names

        Returns
        -------
        ~mantid.api.IMDHistoWorkspace
        """
        if query == 'strain':
            field = self.strain
        elif query == 'stress':
            field = self.stress
        else:
            field = self.peak_parameter(query)
        return field.to_md_histo_workspace()
