from pyrs.core.peak_fit_engine import PeakFitEngine
from scipy.optimize import leastsq
import numpy as np
from pyrs.utilities import checkdatatypes


class ScipyPeakFitEngine(PeakFitEngine):
    """peak fitting engine class for mantid

    Peak fitting engine by calling mantid
    Set up the testing environment for PyVDrive commands

    """

    def __init__(self, data_set_list, ref_id):
        """
        initialization
        :param data_set_list:
        :param ref_id:
        :param
        """
        self.fitted_ws = None

        super(ScipyPeakFitEngine, self).__init__(data_set_list, ref_id)

        return

    @staticmethod
    def calculate_peak(X, Data, TTH, peak_function_name, background_function_name, ReturnModel=False):
        """ static method to calculate peak
        :param X:
        :param Data:
        :param TTH:
        :param peak_function_name:
        :param background_function_name:
        :param ReturnModel:
        :return:
        """
        checkdatatypes.check_string_variable('Peak profile function', peak_function_name)

        model_y = np.zeros_like(Data)

        if peak_function_name == 'Lorentzian':
            # Lorentzian
            x0 = X[0]
            N = X[1]
            f = X[2]
            model_y += N * 2. / np.pi * 1 / f * 1 / (1 + 4 * (TTH - x0) ** 2 / f ** 2)

        elif peak_function_name == 'Gaussian':
            # Gaussian
            x0 = X[0]
            N = X[1]
            f = X[2]
            model_y += N * 2. * np.sqrt(np.log(2) / np.pi) * 1 / f * np.exp(-np.log(2) * 4 * (TTH - x0) ** 2 / f ** 2)

        elif peak_function_name == 'PseudoVoigt':
            x0 = X[0]
            N = X[1]
            f = X[2]
            w = X[3]
            model_y += N * (1. - w) * 2. / np.pi * 1 / f * 1 / (1 + 4 * (TTH - x0) ** 2 / f ** 2) + N * (
                w) * 2. * np.sqrt(np.log(2) / np.pi) * 1 / f * np.exp(-np.log(2) * 4 * (TTH - x0) ** 2 / f ** 2)

        if background_function_name == 'Linear':
            model_y += X[-2:][0] * TTH + X[-2:][1]
        elif background_function_name == 'Quadratic':
            model_y += X[-3:][0] * TTH * TTH + X[-3:][1] * TTH + X[-3:][2]

        if ReturnModel:
            return [TTH, model_y]
        else:
            return Data - model_y

    def fit_peaks(self, peak_function_name, background_function_name, scan_index=None):
        """
        fit peaks
        :param peak_function_name:
        :param background_function_name:
        :param scan_index:
        :return:
        """
        checkdatatypes.check_string_variable('Peak function name', peak_function_name)
        checkdatatypes.check_string_variable('Background function name', background_function_name)

        M = []

        # for i in range(self._data_workspace[2]):
        for i in range(self._num_data_set):
            Data = self._data_set[i].vec_x  # self._data_workspace[1][i]
            TTH = self._data_set[i].vec_y   # self._data_workspace[0][i]

            MaxIndex = np.where(np.max(Data) == Data)[0][0]

            Pos = TTH[MaxIndex]
            LL = np.where((Pos - 1.) < TTH)[0][0]
            UL = np.where((Pos + 1.) > TTH)[0][-1:][0]
            IA = np.sum(Data[LL:UL]) * (TTH[1] - TTH[0])
            f = IA / Data[MaxIndex]
            if peak_function_name == 'PseudoVoigt':
                x0 = [Pos, IA, f, 0.99]
            else:
                x0 = [Pos, IA, f]

            if background_function_name.split(' ')[0] == 'Linear':
                x0.append(0)
                x0.append(Data[0])
            else:
                x0.append(0)
                x0.append(0)
                x0.append(Data[0])

            result = leastsq(self.calculate_peak, x0,
                             args=(Data, TTH, peak_function_name, background_function_name.split(' ')[0]),
                             full_output=True, ftol=1.e-15, xtol=1.e-15)

            M.append(result[0])

        M = np.array(M)
        print('M is of shape {}'.format(M.shape))

        # process output
        # TODO FIXME - pandas is disabled due to deployment conflict with numpy for Python2.7
        #              Develop another data structure to replace pandas.DataFrame!
        # create pandas data frame
        # self.peak_pos_ws = M[:, 0]
        # if peak_function_name == 'PseudoVoigt':
        #     self.func_param_ws = pd.DataFrame(
        #         data={'PeakCentre': M[:, 0], 'Height': M[:, 1], 'FWHM': M[:, 2], 'Mixing': M[:, 3]})
        # elif peak_function_name == 'Lorentzian':
        #     self.func_param_ws = pd.DataFrame(
        #         data={'PeakCentre': M[:, 0], 'Amplitude': M[:, 1], 'FWHM': M[:, 2], 'chi2': 0})
        # else:
        #     self.func_param_ws = pd.DataFrame(
        #         data={'PeakCentre': M[:, 0].T, 'Height': M[:, 1], 'Sigma': M[:, 2] / 2.3548, 'chi2': 0})

        # calculate patterns
        CalcPatts = []
        for log_index in range(self._data_workspace[2]):
            CalcPatts.append(self.calculate_peak(M[log_index, :], self._data_workspace[1][log_index],
                                                 self._data_workspace[0]
                                                 [log_index],
                                                 peak_function_name, background_function_name.split(' ')[0],
                                                 ReturnModel=True))

        self.fitted_ws = np.array(CalcPatts)

        print('[DB...BAT] function parameters keys: {}'.format(self.func_param_ws.keys))

        return

    def get_calculated_peak(self, scan_index):
        """
        get the calculated peak's value
        :param scan_index:
        :return:
        """
        # TODO
        vec_x = self.fitted_ws[scan_index][0].T
        vec_y = self.fitted_ws[scan_index][1].T

        return vec_x, vec_y

    def get_function_parameter_names(self):
        """
        get function parameters' names
        :return:
        """
        return self.func_param_ws.keys()

    def get_number_scans(self):
        """
        get number of scans in input data to fit
        :return:
        """
#        if self._data_workspace is None:
#            raise RuntimeError('No data is set up!')
#
#        return self._data_workspace.getNumberHistograms()

        return self._data_workspace[2]

    def get_fitted_params(self, param_name):
        """

        :return:
        """
#        col_names = self.func_param_ws.getColumnNames()
#        if param_name in col_names:
#            col_index = col_names.index(param_name)
#        else:
#            raise RuntimeError('Function parameter {0} does not exist.'.format(param_name))
#
#        param_vec = np.ndarray(shape=(self._data_workspace[2]), dtype='float')
#        for row_index in range(self._data_workspace[2]):
#            print self.func_param_ws[param_name]
#            print self.func_param_ws[param_name][row_index][0]

#            param_vec[row_index] = self.func_param_ws[param_name][row_index][0]

        return self.func_param_ws[param_name].values
