from mantid.api import IMDHistoWorkspace
import numpy as np
from numpy.testing import assert_allclose
from typing import Dict, List, Optional, Tuple, Union

from pyrs.core.peak_profile_utility import EFFECTIVE_PEAK_PARAMETERS
from pyrs.dataobjects.fields import ScalarFieldSample, StrainField, StrainFieldSingle, StressField, StressType
from pyrs.dataobjects.sample_logs import PointList


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

    def _update_cache_stress(self) -> None:
        # Update stress cache
        self._stress_cache = {'11': self._stress.stress11, '22': self._stress.stress22, '33': self._stress.stress33}

    def _update_caches(self) -> None:
        r"""Update the strain and stress references for each direction and run number"""
        self._update_cache_stress()
        # Update strain cache
        self._strain_cache = {'11': self._stress.strain11, '22': self._stress.strain22, '33': self._stress.strain33}
        for direction in ('11', '22', '33'):
            strain = self._strain_cache[direction]
            for peak_collection, strain in zip(strain.peak_collections, strain.strains):
                self._strain_cache[str(peak_collection.runnumber)] = strain

    @property
    def selection(self) -> Optional[str]:
        r"""
        Pick a scanning direction or run number

        Examples
        --------
        > facade.selection = '11'  # select along the first direction
        > facade.selection = '1234'  # select run number 1234
        """
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
    def size(self) -> int:
        return self._stress.size

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
    def point_list(self) -> PointList:
        return self._stress.point_list

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
            self._update_d_reference()
        assert self._d_reference is not None  # this line is brought to you by mypy
        return self._d_reference

    @d_reference.setter
    def d_reference(self, d_update: Union[float, Tuple[float, float], List[float],
                                          np.ndarray, ScalarFieldSample]) -> None:
        r"""
        Update the reference lattice spacing, and recalculate strains and stresses.

        Examples
        --------
        facade.d_reference = 1.01  # single value, and assumed a 0.0 error
        facade.d_reference = (1.01, 0.03)  # single value and error
        facade.d_reference = field  # an instance of ScalarFieldSample

        Parameters
        ----------
        d_update: float, tuple list, ~pyrs.dataobjects.fields.ScalarFieldSample
        """
        if isinstance(d_update, float):
            self._stress.set_d_reference((d_update, 0.0))
        elif isinstance(d_update, (list, tuple, np.ndarray)):
            self._stress.set_d_reference(d_update[0:2])  # type: ignore
        elif isinstance(d_update, ScalarFieldSample):
            self._stress.set_d_reference(d_update)
        self._update_d_reference()

    def _update_d_reference(self) -> None:
        r"""
        Initialize or update the cache `self._d_reference`
        """
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
                mask = ~(np.isnan(d0_ii) | np.isnan(d0_jj))  # indexes where d0_ii and d0_jj are not Nan
                assert_allclose(d0_ii[mask], d0_jj[mask], rtol=1.e-4,
                                err_msg='reference spacings are different on different directions')

        # "merge" reference spacings along different directions where is not nan
        d0_values = np.full(len(self.x), np.nan)
        d0_errors = np.full(len(self.x), 0.0)
        for d0 in d0s:
            not_nan_indices = ~np.isnan(d0.values)
            d0_values[not_nan_indices] = d0.values[not_nan_indices]
            d0_errors[not_nan_indices] = d0.errors[not_nan_indices]

        # build the consensus scalar field
        self._d_reference = ScalarFieldSample('d-reference', d0_values, d0_errors, self.x, self.y, self.z)

    @property
    def strain(self) -> ScalarFieldSample:
        r"""
        Scalar field sample with strain values for the selected direction or run number

        Returns
        -------
        ~pyrs.dataobjects.fields.ScalarFieldSample
        """
        assert self._selection is not None, 'No selection has been entered'
        return self._extend_to_stacked_point_list(self._strain_cache[self._selection].field)

    def _extend_to_stacked_point_list(self, field: ScalarFieldSample) -> ScalarFieldSample:
        r"""
        Extend a scalar field sample from a single run to the point list of the stacked strains

        Parameters
        ----------
        field: ~pyrs.dataobjects.fields.ScalarFieldSample
            A scalar field sample generated by one of the single runs

        Returns
        -------
        ~pyrs.dataobjects.fields.ScalarFieldSample
        """
        if self._selection in ('11', '22', '33'):
            return field  # this field was generated by one of the stacked strains
        else:
            return field.extend_to_point_list(self.point_list)

    @property
    def stress(self) -> ScalarFieldSample:
        r"""
        Scalar field sample with stress values for the selected direction or run number

        Raises
        ------
        ValueError
            When the the selection is a run number, instead of one of the directions

        Returns
        -------
        ~pyrs.dataobjects.fields.ScalarFieldSample
        """
        if self._selection not in ('11', '22', '33'):
            raise ValueError('Stress can only be computed for directions, not for run numbers')
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

    @youngs_modulus.setter
    def youngs_modulus(self, value: float) -> None:
        self._stress.youngs_modulus = value
        self._update_cache_stress()

    @property
    def poisson_ratio(self) -> float:
        return self._stress.poisson_ratio

    @poisson_ratio.setter
    def poisson_ratio(self, value: float) -> None:
        self._stress.poisson_ratio = value
        self._update_cache_stress()

    @property
    def stress_type(self) -> str:
        r"""
        Stress type, one of ('diagonal', 'in-plane-strain', 'in-plane-stress')

        Returns
        -------
        str
        """
        return self._stress.stress_type.value

    @property
    def peak_parameters(self) -> List[str]:
        r"""
        List of effective peak parameter names, plus 'd' for lattice-plane spacing.

        Returns
        -------
        list
        """
        return ['d'] + EFFECTIVE_PEAK_PARAMETERS

    def peak_parameter(self, query: str) -> ScalarFieldSample:
        r"""
        Peak parameter values (including d-spacing) for the selection direction or run number

        Parameters
        ----------
        query: str

        Returns
        -------
        ~pyrs.dataobjects.fields.ScalarFieldSample
        """
        assert self._selection is not None, 'Please make a direction or number number selection'
        assert query in self.peak_parameters, f'Peak parameter must be one of {self.peak_parameters}'

        if query == 'd':
            return self._d_spacing()

        if self._selection == '33' and self.stress_type in ('in-plane-strain', 'in-plane-stress'):
            msg = f'{query} not measured along 33 when in {self.stress_type}'
            raise ValueError(msg)

        peak_parameter_field = self._strain_cache[self._selection].get_effective_peak_parameter(query)

        return self._extend_to_stacked_point_list(peak_parameter_field)

    def _d_spacing(self) -> ScalarFieldSample:
        r"""
        Calculate d-spacing for a direction or single run

        Returns
        -------
        ~pyrs.dataobjects.fields.ScalarFieldSample
        """
        if self._selection == '33' and self.stress_type in ('in-plane-strain', 'in-plane-stress'):
            msg = f'd-spacing not measured along 33 when in {self.stress_type}'
            raise ValueError(msg)

        assert self._selection is not None
        d_spacing_field = self._strain_cache[self._selection].get_dspacing_center()

        return self._extend_to_stacked_point_list(d_spacing_field)

    def workspace(self, query: str) -> IMDHistoWorkspace:
        r"""
        Create an MDHistoWorkspace for the selection strain, stress, or effective peak parameter

        Parameters
        ----------
        query: str
            One of 'strain', 'stress', 'd_reference', 'd', 'Center', 'Height', 'FWHM', 'Mixing',
            'A0', 'A1', 'Intensity'

        Returns
        -------
        ~mantid.api.IMDHistoWorkspace
        """
        if query == 'strain':
            field = self.strain
        elif query == 'stress':
            field = self.stress
        elif query == 'd_reference':
            field = self.d_reference
        else:
            field = self.peak_parameter(query)
        return field.to_md_histo_workspace()
