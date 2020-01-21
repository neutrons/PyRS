# Peak fitting engine by calling mantid
from .peak_fit_engine import PeakFitEngine, FitResult
from pyrs.core.peak_profile_utility import PeakShape
from pyrs.peaks import PeakCollection
import numpy as np
from mantid.kernel import Logger
from mantid.simpleapi import DeleteWorkspace, FitPeaks, RenameWorkspace

__all__ = ['MantidPeakFitEngine']

DEBUG = False   # Flag for debugging mode


class MantidPeakFitEngine(PeakFitEngine):
    def __init__(self, hidraworkspace, peak_function_name, background_function_name, out_of_plane_angle):
        super(MantidPeakFitEngine, self).__init__(hidraworkspace, peak_function_name,
                                                  background_function_name, out_of_plane_angle)
        # configure logging for this class
        self._log = Logger(__name__)

    def __guess_center(self, x_min, x_max):
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

    def fit_peaks(self, peak_tag, x_min, x_max):
        x_min, x_max = self._check_fit_range(x_min, x_max)

        # Create output workspace names
        r_positions_ws_name = 'fitted_peak_positions_{0}'.format(self._mtd_wksp)
        r_param_table_name = 'param_m_{0}'.format(self._mtd_wksp)
        r_error_table_name = 'param_e_{0}'.format(self._mtd_wksp)
        r_model_ws_name = 'model_full_{0}'.format(self._mtd_wksp)

        # estimate the peak center
        peak_center = self.__guess_center(x_min, x_max)

        # add in extra parameters for starting values
        kwargs = {}
        if self._peak_function == PeakShape.PSEUDOVOIGT:
            # max_intensity = PseudoVoigt.cal_intensity(max_estimated_height, hidra_fwhm, default_mixing)
            intensity = self._mtd_wksp.extractY().max() - self._mtd_wksp.extractY().mean()  # TODO improve this

            kwargs['PeakParameterNames'] = 'Mixing, Intensity'  # FWHM also available
            kwargs['PeakParameterValues'] = '{}, {}'.format(0.6, intensity)  # mixing agreed upon default

        # Fit peak by Mantid.FitPeaks
        fit_return = FitPeaks(InputWorkspace=self._mtd_wksp,
                              PeakFunction=str(self._peak_function),
                              BackgroundType=str(self._background_function),
                              FindBackgroundSigma=1,
                              HighBackground=True,
                              PeakCenters=peak_center,
                              FitWindowBoundaryList=(x_min, x_max),
                              # ConstrainPeakPositions=True,  TODO should this be turned on?
                              RawPeakParameters=True,
                              OutputWorkspace=r_positions_ws_name,  # peak centers
                              OutputPeakParametersWorkspace=r_param_table_name,  # peak parameters
                              OutputParameterFitErrorsWorkspace=r_error_table_name,  # peak parameter uncertainties
                              FittedPeaksWorkspace=r_model_ws_name,  # peaks calculated from model
                              MaxFitIterations=50,  # TODO increase to 500
                              **kwargs
                              )
        # r is a class containing multiple outputs (workspaces)
        if fit_return is None:
            raise RuntimeError('return from FitPeaks cannot be None')

        # convert the output tables into a pyrs.peaks.PeakCollection
        peak_collection = self.__tables_to_peak_collection(peak_tag,
                                                           fit_return.OutputPeakParametersWorkspace,
                                                           fit_return.OutputParameterFitErrorsWorkspace)
        DeleteWorkspace(fit_return.OutputPeakParametersWorkspace)
        DeleteWorkspace(fit_return.OutputParameterFitErrorsWorkspace)

        # create the difference workspace
        difference = self._mtd_wksp - fit_return.FittedPeaksWorkspace
        difference = RenameWorkspace(InputWorkspace=difference, OutputWorkspace=peak_tag+'_diff')

        # return the final results
        return FitResult(peakcollections=(peak_collection,), fitted=fit_return.FittedPeaksWorkspace,
                         difference=difference)

    def __tables_to_peak_collection(self, peak_tag, table_params, table_errors):
        def convert_from_table_to_arrays(table_ws):  # TODO put this in mantid_helper
            # Table column names
            table_col_names = table_ws.getColumnNames()
            num_sub_runs = table_ws.rowCount()

            # Set the structured numpy array
            data_type_list = [(name, np.float32) for name in table_col_names]
            struct_array = np.zeros(num_sub_runs, dtype=data_type_list)

            # get fitted parameter value
            for col_index, param_name in enumerate(table_col_names):
                # get value from column in value table
                struct_array[param_name] = table_ws.column(col_index)

            return struct_array

        peak_params_value_array = convert_from_table_to_arrays(table_params)
        peak_params_error_array = convert_from_table_to_arrays(table_errors)
        fit_cost_array = peak_params_value_array['chi2']

        # Create PeakCollection instance
        peak_object = PeakCollection(peak_tag=peak_tag, peak_profile=self._peak_function,
                                     background_type=self._background_function)
        peak_object.set_peak_fitting_values(self._subruns, peak_params_value_array,
                                            peak_params_error_array, fit_cost_array)

        return peak_object
