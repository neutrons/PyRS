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

    def select(self, selection: str) -> None:
        r"""
        Pick a scanning direction or run number

        Parameters
        ----------
        selection: str
            If a scanning direction, pick one of ('11', '22', '33'), or enter a run number
        """
        if len(selection) == 2:
            assert selection in ('11', '22', '33')
        else:
            assert selection in self._all_runs()
        self._selection = selection

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
    def d0(self):
        pass