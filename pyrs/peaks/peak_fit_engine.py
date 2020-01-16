# This is the virtual base class as the fitting frame
from collections import namedtuple
import numpy as np
from pyrs.core import mantid_helper
from pyrs.core.peak_profile_utility import BackgroundFunction, PeakShape
from mantid.simpleapi import DeleteWorkspace, RenameWorkspace

__all__ = ['PeakFitEngine']


class FitResult(namedtuple('FitResult', 'peakcollections fitted difference')):
    '''(:py:obj:`(~pyrs.peaks.peak_collection)`, :py:obj:`~mantid.api.MatrixWorkspace`,
    :py:obj:`~mantid.api.MatrixWorkspace`)'''
    pass


class PeakFitEngine(object):
    def __init__(self, hidraworkspace, peak_function_name, background_function_name,
                 out_of_plane_angle):
        '''It is not expected that any subclass will need to implement this method. It is
        designed to unpack all of the information necessary for fitting.'''
        if out_of_plane_angle:
            # TODO variable should go into creating the mantid workspace
            raise NotImplementedError('Do not currently support out_of_plane_angle')
        # TODO generate_mantid_workspace should not use `mask_id`
        self._mtd_wksp = mantid_helper.generate_mantid_workspace(hidraworkspace, hidraworkspace.name, None)
        self._subruns = hidraworkspace.get_sub_runs()

        # create a
        self._peak_function = PeakShape.getShape(peak_function_name)
        self._background_function = BackgroundFunction.getFunction(background_function_name)

    @staticmethod
    def _check_fit_range(x_min, x_max):
        scalar = True
        try:
            x_min = [float(x_min)]
            x_max = [float(x_max)]
        except TypeError:
            scalar = False
            # coerce data types
            x_min = np.atleast_1d(x_min).astype(float)
            x_max = np.atleast_1d(x_max).astype(float)

        # validate that they are in the right order
        for left, right in zip(x_min, x_max):
            if left >= right:
                raise RuntimeError('Invalid fitting range {} >= {}'.format(left, right))

        # TODO? verify it is within the data range

        if scalar:
            x_min, x_max = x_min[0], x_max[0]

        return x_min, x_max

    def fit_peaks(self, peak_tag, x_min, x_max):
        '''Fit a single peak across subruns. This will return a :py:obj:`FitResult`'''
        raise NotImplementedError('This must be implemented by the concrete class')

    def fit_multiple_peaks(self, peak_tags, x_mins, x_maxs):
        '''Fit multiple peaks across subruns. This will return a :py:obj:`FitResult`
        Concrete instances may instantiate this as needed'''
        x_mins, x_maxs = self._check_fit_range(x_mins, x_maxs)
        assert len(peak_tags) == len(x_mins) == len(x_maxs), 'All inputs must have same number of values'

        # fit each peak separately
        peakcollections = []
        fitted = None
        for peak_tag, x_min, x_max in zip(peak_tags, x_mins, x_maxs):
            # fit an individual peak
            individual = self.fit_peaks(peak_tag, x_min, x_max)

            # collect information
            peakcollections.extend(individual.peakcollections)
            if fitted:
                fitted += individual.fitted
                DeleteWorkspace(individual.fitted)
            else:
                fitted = individual.fitted
                fitted = RenameWorkspace(InputWorkspace=fitted,
                                         OutputWorkspace='{}_fitted'.format(peak_tags[0]))

            # original difference isn't needed
            DeleteWorkspace(individual.difference)

        # calculate the difference
        difference = self._mtd_wksp - fitted
        difference = RenameWorkspace(InputWorkspace=difference, OutputWorkspace=peak_tags[0]+'_diff')

        return FitResult(peakcollections=peakcollections, fitted=fitted, difference=difference)
