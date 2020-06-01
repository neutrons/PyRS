from collections import namedtuple
from mantid.kernel import Logger
from mantid.simpleapi import DeleteWorkspace, RenameWorkspace
import numpy as np
from pyrs.core import mantid_helper
from pyrs.core.peak_profile_utility import BackgroundFunction, PeakShape

__all__ = ['PeakFitEngine']


class FitResult(namedtuple('FitResult', 'peakcollections fitted difference')):
    '''(:py:obj:`(~pyrs.peaks.peak_collection)`, :py:obj:`~mantid.api.MatrixWorkspace`,
    :py:obj:`~mantid.api.MatrixWorkspace`)'''
    pass


class PeakFitEngine(object):
    '''This is the virtual base class as the fitting frame'''
    def __init__(self, hidraworkspace, peak_function_name, background_function_name,
                 wavelength, out_of_plane_angle):
        '''It is not expected that any subclass will need to implement this method. It is
        designed to unpack all of the information necessary for fitting.'''
        # configure logging for this class
        self._log = Logger(__name__)

        if out_of_plane_angle:
            # TODO variable should go into creating the mantid workspace
            raise NotImplementedError('Do not currently support out_of_plane_angle')
        # TODO generate_mantid_workspace should not use `mask_id`
        self._mtd_wksp = mantid_helper.generate_mantid_workspace(hidraworkspace, hidraworkspace.name, None)
        self._subruns = hidraworkspace.get_sub_runs()

        # create a
        self._peak_function = PeakShape.getShape(peak_function_name)
        self._background_function = BackgroundFunction.getFunction(background_function_name)

        self._wavelength = wavelength

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

    def _guess_center(self, x_min, x_max):
        '''
        Try getting the peak center from each subrun by one of three methods (first one wins):
        1. First moment of the data within the window
        2. x-value of the maximum y-valuu in the window
        3. Center value of the window
        '''
        center = []

        for wksp_index in range(self._mtd_wksp.getNumberHistograms()):
            x_vals = self._mtd_wksp.readX(wksp_index)
            i_min, i_max = x_vals.searchsorted([x_min, x_max])
            if i_min >= i_max:
                msg = 'Failed to find requested x-range({} < {}) '.format(x_min, x_max)
                msg += 'in data with x-range ({} < {})'.format(x_vals[0], x_vals[-1])
                self._log.warning(msg)
                continue  # don't use this workspace index
            y_vals = self._mtd_wksp.readY(wksp_index)[i_min:i_max]
            y_offset = np.abs(y_vals.max())

            # add the first moment to the list of centers
            moment = np.sum(x_vals[i_min:i_max] * (y_vals + y_offset)) / np.sum(y_vals + y_offset)
            if (x_min < moment < x_max):
                center.append(moment)
            else:
                self._log.notice('Moment calculation failed to find peak center. Using maximum y-value')
                top = y_vals.argmax()
                if top > 0:
                    center.append(x_vals[i_min + top])
                else:
                    self._log.warning('Failed to find maximum y-value. Using center of fit window')
                    center.append(0.5 * (x_min + x_max))

        # calculate the average value across all the subruns
        if len(center) == 0:
            raise RuntimeError('Failed to find any peak centers')
        center = np.mean(center)  # mean value of everything

        # final error check
        if not (x_min < center < x_max):
            raise RuntimeError('Failed to guess peak center between {} < {} < {}'.format(x_min, center, x_max))
        return center
