from typing import List

from pyrs.dataobjects.fields import StressField, StressType


class StressFacade:

    def __init__(self, stress: StressField):
        r"""

        Parameters
        ----------
        stress: ~pyrs.dataobjects.fields.StressField
        """
        assert isinstance(stress, StressField)
        self._stress = stress
        self._selection = None
        self._strain_cache = {}  # cache of strain references
        self._stress_cache = {}  # cache of stress references
        self._update_caches()

    def _update_caches(self):
        r"""Update the strain and stress references for each direction and run number"""
        # Update stress cache
        self._stress_cache = {'11': self._stress.stress11, '22': self._stress.stress22, '33':self._stress.stress33}
        # Update strain cache
        self._strain_cache = {'11': self._stress.strain11, '22': self._stress.strain22, '33':self._stress.strain33}
        for direction in ('11', '22', '33'):
            strain = self._strain_cache[direction]
            for peak_collection, strain in zip(strain.peak_collections, strain.strains):
                self._strain_cache[peak_collection.runnumber] = strain
        # Update d_reference


    @property
    def selection(self):
        r"""Pick a scanning direction or run number"""
        return self._selection

    @selection.setter
    def selection(self, choice):
        if len(choice) == 2:
            assert choice in ('11', '22', '33')
        else:
            assert choice in self._all_runs()
        self._selection = choice

    @property
    def strain(self):
        self._strain[self._selection]

    @property
    def stress(self):
        if self._selection not in ('11', '22', '33'):
            raise ValueError(f'Selection {self._selection} must specify one direction')
        self._stress[self._selection]

    def _all_runs(self):
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
        return [str(peak_collection.runnumber) for peak_collection in self._stress.strain.peak_collections]

    @property
    def youngs_modulus(self):
        return self._stress.youngs_modulus

    @property
    def poisson_ratio(self):
        return self._stress.poisson_ratio

    @property
    def d_refernce(self):
        pass
