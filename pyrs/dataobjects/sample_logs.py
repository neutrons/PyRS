# extentable version of dict https://treyhunner.com/2019/04/why-you-shouldnt-inherit-from-list-and-dict-in-python/
from collections import Iterable, MutableMapping
from typing import Any, List, NamedTuple, Tuple
import numpy as np
from .constants import HidraConstants  # type: ignore

__all__ = ['SampleLogs', 'SubRuns']


def _coerce_to_ndarray(value):
    if isinstance(value, np.ndarray):
        return value
    elif isinstance(value, SubRuns):
        return value._value  # pylint: disable=protected-access
    else:
        return np.atleast_1d(value)


class SubRuns(Iterable):
    '''SubRun class is a (mostly) immutable object that allows for getting the index of its arguments.'''
    def __init__(self, subruns=None):
        '''Default is to create zero-length subruns. This is the only version of
        subrun that can have its value updated'''
        self._value = np.ndarray((0))

        if subruns is not None:
            self.set(subruns)

    def __getitem__(self, key):
        return self._value[key]

    def __eq__(self, other):
        other = _coerce_to_ndarray(other)
        if other.size != self._value.size:
            return False
        else:
            return np.all(other == self._value)

    def __ne__(self, other):
        return not (self.__eq__(other))

    def __iter__(self):
        iterable = self._value.tolist()

        return iterable.__iter__()

    def __repr__(self):
        return repr(self._value)

    def __str__(self):
        return str(self._value)

    @property
    def size(self):
        return self._value.size

    @property
    def shape(self):
        return self._value.shape

    @property
    def ndim(self):
        return self._value.ndim

    def __len__(self):
        return self._value.size

    def empty(self):
        return self._value.size == 0

    def set(self, value):
        value = _coerce_to_ndarray(value)
        if not self.empty():
            if self.__ne__(value):
                raise RuntimeError('Cannot change subruns when non-empty '
                                   '(previous={}, new={})'.format(self._value, value))
        if not np.all(value[:-1] < value[1:]):
            raise RuntimeError('subruns are not soryed in increasing order')
        self._value = value.astype(int)

    def raw_copy(self):
        '''Raw copy of underlying values'''
        return np.copy(self._value)

    def get_indices(self, subruns):
        '''Convert the list of subruns into indices into the subrun array'''
        if self.__eq__(subruns):
            return np.arange(self._value.size)
        else:
            subruns = _coerce_to_ndarray(subruns)
            # look for the single value
            if subruns.size == 1:
                indices = np.nonzero(self._value == subruns[0])[0]
                if indices.size > 0:
                    return indices
            # check that the first and last values are in the array
            elif subruns[0] in self._value and subruns[-1] in self._value:
                return np.searchsorted(self._value, subruns)

        # fall-through is an error
        raise IndexError('Failed to find subruns={} in {}'.format(subruns, self._value))


class SampleLogs(MutableMapping):
    SUBRUN_KEY = HidraConstants.SUB_RUNS

    def __init__(self, **kwargs):
        self._data = dict(kwargs)
        self._subruns = SubRuns()
        self._plottable = set([self.SUBRUN_KEY])

    def __del__(self):
        del self._data
        del self._subruns

    def __delitem__(self, key):
        if key == self.SUBRUN_KEY:
            self.subruns = SubRuns()  # set to empty subruns
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
            self.subruns = SubRuns(value)  # use full method
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
        for key in sorted(self.keys()):
            if key == self.SUBRUN_KEY:
                continue
            elif key in self._plottable:  # plottable things are numbers
                if self._data[key].std() <= atol:
                    result.append(key)
            elif np.alltrue(self._data[key] == self._data[key][0]):  # all values are equal
                result.append(key)

        return result

    @property
    def subruns(self):
        '''This method must exist in order to customize the setter'''
        return self._subruns

    @subruns.setter
    def subruns(self, value):
        '''Set the subruns and build up associated values'''
        self._subruns.set(value)

    def matching_subruns(self, subruns):
        return self._subruns == subruns

    def get_subrun_indices(self, subruns):
        return self._subruns.get_indices(subruns)


class _DirectionExtents(NamedTuple):
    min: float  # minimum value of the sample coordinate along one particular direction
    max: float  # maximum value of the sample coordinate along one particular direction
    delta: float  #


class DirectionExtents(_DirectionExtents):
    r"""
    Spacing parameters for sample positions sampled along a particular direction.

    Two sample positions are deemed the same if they differ by less than some distance, here a
    class attribute termed 'precision'.

    Attributes:
        min: minimum sample position sampled
        max: maximum sample position sampled
        delta: average spacing between unique sample positions sampled
    """

    precision = 1.e-03  # two coordinates values differing by less that this amount are considered the same value

    def __new__(cls, coordinates: List[float]):
        min_coord = min(coordinates)
        max_coord = max(coordinates)
        # unique number of different coordinates using and assumed precision in the coordinate values
        coordinates_count_unique = len(set([int(x / cls.precision) for x in coordinates]))
        # delta is the spacing between unique coordinates
        delta = (max_coord - min_coord) / (coordinates_count_unique - 1)

        extents_tuple = super(DirectionExtents, cls).__new__(cls, min_coord, max_coord, delta)
        extents_tuple._numpoints = coordinates_count_unique
        return extents_tuple

    @property
    def numpoints(self):
        r"""
        Number of centerpoints where self.min and self.max are the first and last centerpoints
        """
        return self._numpoints

    @property
    def number_of_bins(self):
        r"""
        Number of spacings separating consecutive bin boundaries
        """
        return self._numpoints  # same as number of center points

    @property
    def to_createmd(self) -> str:
        r"""
        Minimum and maximum extents to be passed as argument Extent of Mantid algorithm
        `CreateMDWorkspace <https://docs.mantidproject.org/nightly/algorithms/CreateMDWorkspace-v1.html>`_.

        Input extents for CreateMDWorkspace become the first and last bin boundaries, but `self.min` and
        `self.max` are the first and last center-points

        Returns
        -------
        str
        """
        return f'{self.min - self.delta / 2},{self.max + self.delta / 2}'

    def to_binmd(self, label: str) -> str:
        r"""
        Binning parameters to be passed as one of the AlignedDimX arguments of Mantid algorithm
        `BinMD <>`_.

        Returns
        -------
        str
        """
        return f'{label},{self.to_createmd},{self.number_of_bins}'


class PointList:

    class _PointList(NamedTuple):
        r"""Data structure containing the list of coordinates"""
        vx: List[float]  # coordinates stored in log name HidraConstants.SAMPLE_COORDINATE_NAMES[0]
        vy: List[float]  # coordinates stored in log name HidraConstants.SAMPLE_COORDINATE_NAMES[1]
        vz: List[float]  # coordinates stored in log name HidraConstants.SAMPLE_COORDINATE_NAMES[2]

    def __init__(self, input_source: SampleLogs):
        r"""
        List of sample coordinates.

        point_list.vx returns the list of coordinates along the first axis
        point_list[42] return the (vx, vy, vz) coordinates of point 42
        Iteration is over the 3D points, not over the three directions

        Parameters
        ----------
        input_source: ~pyrs.dataobjects.sample_logs.SampleLogs
            data structure containing the values of the coordinates for each direction.
        """
        coordinates = [list(input_source[name]) for name in HidraConstants.SAMPLE_COORDINATE_NAMES]
        # Check the number of coordinate values on each direction is the same for all directions
        assert len(set([len(c) for c in coordinates])) == 1, 'Directions have different number of coordinates'
        self._points = self.__class__._PointList(*coordinates)

    def __len__(self) -> int:
        return len(self._points.vx)  # assumed all the three directions have the same number of coordinate values

    def __getattr__(self, item: str) -> Any:
        r"""Enable self.vx, self.vy, self.vz"""
        try:
            points = self.__dict__['_points']
            return getattr(points, item)
        except AttributeError:
            getattr(self, item)

    def __getitem__(self, item: int) -> List[float]:
        r"""Enable self[0],... self[N] as well as making this class iterable over the 3D points."""
        return [coordinate_list[item] for coordinate_list in self._points]

    @property
    def extents(self) -> Tuple[DirectionExtents, DirectionExtents, DirectionExtents]:
        r"""
        Extents along each direction. Each extent is composed of the minimum and maximum coordinates,
        as well a coordinate increment.

        Returns
        -------
        list
            three-item list, where each item is an object of type ~pyrs.dataobjects.sample_logs.DirectionExtents.
        """
        return tuple([DirectionExtents(axes_coords) for axes_coords in self._points])
