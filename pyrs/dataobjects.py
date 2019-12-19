# extentable version of dict https://treyhunner.com/2019/04/why-you-shouldnt-inherit-from-list-and-dict-in-python/
from collections import MutableMapping
import numpy as np


def _coerce_to_ndarray(value):
    if isinstance(value, np.ndarray):
        return value
    else:
        return np.atleast_1d(value)


class SampleLogs(MutableMapping):
    SUBRUN_KEY = 'sub-runs'  # TODO should be pyrs.projectfile.HidraConstants.SUB_RUNS

    def __init__(self, **kwargs):
        self._data = dict(kwargs)
        self._subruns = np.ndarray((0))
        self._plottable = set([self.SUBRUN_KEY])

    def __del__(self):
        del self._data
        del self._subruns

    def __delitem__(self, key):
        if key == self.SUBRUN_KEY:
            self.subruns = np.ndarray((0))  # use full method
        else:
            del self._data[key]

    def __getitem__(self, key):
        if isinstance(key, tuple):
            key, subruns = key
        else:
            subruns = None

        if key == self.SUBRUN_KEY:
            if subruns:
                raise RuntimeError('Cannot use __getitem__ to get subset of subruns')
            return self._subruns
        else:
            if (subruns is None) or self.matching_subruns(subruns):
                return self._data[key]
            else:
                return self._data[key][self.get_subrun_indices(subruns)]

    def __iter__(self):
        # does not include subruns
        return iter(self._data)

    def __len__(self):
        '''The number of keys in the underlying dictionary'''
        # does not include subruns
        return len(self._data)

    def __setitem__(self, key, value):
        value = _coerce_to_ndarray(value)
        if key == self.SUBRUN_KEY:
            self.subruns = value  # use full method
        else:
            if self._subruns.size == 0:
                raise RuntimeError('Must set subruns first')
            elif value.size != self.subruns.size:
                raise ValueError('Number of values[{}] isn\'t the same as number of '
                                 'subruns[{}]'.format(value.size, self.subruns.size))
            self._data[key] = value
            # add this to the list of plottable parameters
            if value.dtype.kind in 'iuf':  # int, uint, float
                self._plottable.add(key)

    def plottable_logs(self):
        '''Return the name of all logs that are plottable

        This always includes :py:obj:`~pyrs.projectfile.HidraConstants.SUB_RUNS`
        in addition to all the other logs'''
        return list(self._plottable)

    def constant_logs(self, atol=0.):
        '''Return the name of all logs that have a constant value

        Parameters
        ----------
        atol: float
            Logs with a smaller stddev than the atol (inclusive) will be considered constant'''
        result = list()
        # plottable logs are the numeric ones
        for key in self._plottable:
            if key == self.SUBRUN_KEY:
                continue
            if self._data[key].std() <= atol:
                result.append(key)

        return result

    @property
    def subruns(self):
        '''This method must exist in order to customize the setter'''
        return self._subruns

    @subruns.setter
    def subruns(self, value):
        '''Set the subruns and build up associated values'''
        if self._subruns.size != 0:
            if not self.matching_subruns(value):
                raise RuntimeError('Cannot set subruns on non-empty SampleLog')
        value = _coerce_to_ndarray(value)
        if not np.all(value[:-1] < value[1:]):
            raise RuntimeError('subruns are not soryed in increasing order')
        self._subruns = value

    def matching_subruns(self, subruns):
        subruns = _coerce_to_ndarray(subruns)
        if subruns.size != self._subruns.size:
            return False
        else:
            return np.all(subruns == self._subruns)

    def get_subrun_indices(self, subruns):
        if self.matching_subruns(subruns):
            return np.arange(self._subruns.size)
        else:
            subruns = _coerce_to_ndarray(subruns)
            # look for the single value
            if subruns.size == 1:
                indices = np.nonzero(self._subruns == subruns[0])[0]
                if indices.size > 0:
                    return indices
            # check that the first and last values are in the array
            elif subruns[0] in self._subruns and subruns[-1] in self._subruns:
                return np.searchsorted(self._subruns, _coerce_to_ndarray(subruns))

        # fall-through is an error
        raise IndexError('Failed to find subruns={}'.format(subruns))
