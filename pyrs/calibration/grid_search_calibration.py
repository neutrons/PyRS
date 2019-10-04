# A simple approach to calibrate the instrument geometry by brute-force grid search
# It is time consuming and inefficient.  But it can give some insight how parameters affect geometry
import math
from pyrs.utilities import checkdatatypes
from pyrs.core import mantid_fit_peak


class GridSearchCalibration(object):
    """
    Calibrate by grid searching algorithm using Brute Force or Monte Carlo random walk
    """

    def __init__(self, hb2b_instrument):
        """
        Initialization
        """

        self._instrument_reducer = hb2b_instrument

        return

    def _calculate_cost(self):
        """
        calculate the quantitative
        :return:
        """
        peaks_pos_dict = dict()

        for roi_index in range(7):
            peaks_pos_dict[roi_index] = mantid_fit_peak.fit_peaks(self._reduced_data[roi_index])
        # END-FOR

        for peak_index in range(self._peak_range):
            # calculate the standard deviation
            std_dev_i = blabla

    def _estimate_calculation_time(self, num_rot_x_inter, num_rot_y_inter, num_rot_z_inter, num_shift_x_iter,
                                   num_shift_y_iter, num_shift_z_iter):
        """
        estimate the reduction time for these loops
        :param num_rot_x_inter:
        :param num_rot_y_inter:
        :param num_rot_z_inter:
        :param num_shift_x_iter:
        :param num_shift_y_iter:
        :param num_shift_z_iter:
        :return:
        """
        est_time = num_shift_z_iter * num_shift_y_iter * num_shift_x_iter * num_rot_z_inter * \
            num_rot_y_inter * num_rot_x_inter * self._instrument_reducer.get_estimated_reduction_time

        return est_time

    @staticmethod
    def _set_up_grid(grid_params, note):
        """
        set up a grid
        :param grid_params:
        :param note:
        :return:
        """
        # check and get values
        checkdatatypes.check_series(note, grid_params, [float, None], size=3)
        val_min, val_step, val_max = grid_params

        if val_min is None:
            raise RuntimeError('Minimum value of {} cannot be None'.format(note))

        if val_step is None and val_max is None:
            val_step = 0.
            num_iter = 1
        elif val_step is None:
            val_step = val_max - val_min
            num_iter = 2
        elif val_max is None:
            raise RuntimeError('Maximum value cannot be None when grid step ({}) is not None'.format(val_step))
        elif abs(val_step) > 1.E-20:
            # step cannot be close to zero
            num_steps = (val_max - val_min) / val_step
            if num_steps < -1.E-20:
                raise RuntimeError('Value min {} and max {} are specified in a wrong order with grid step {}'
                                   ''.format(val_min, val_max, val_step))
            num_iter = int(math.ceil(num_steps)) + 1
        else:
            # only 1 situation left
            raise RuntimeError('Grid setup {} has a infinitesimal grid step {}'
                               ''.format(grid_params, val_step))

        return val_min, val_step, num_iter

    def grid_search(self, rot_x, rot_y, rot_z, shift_x, shift_y, shift_z, force_exec):
        """
        do grid search
        :param rot_x:
        :param rot_y:
        :param rot_z:
        :param shift_x:
        :param shift_y:
        :param shift_z:
        :param force_exec: if True, do the search no matter how long it will take.  Otherwise, do an estimation,
        :return:
        """
        rot_x_start, rot_x_step, num_rot_x_inter = self._set_up_grid(rot_x, 'X direction flip')
        rot_y_start, rot_y_step, num_rot_y_inter = self._set_up_grid(rot_y, 'Y direction rotate')
        rot_z_start, rot_z_step, num_rot_z_inter = self._set_up_grid(rot_z, 'Z direction spin')

        shift_x_start, shift_x_step, num_shift_x_iter = self._set_up_grid(shift_x, 'X direction shift')
        shift_y_start, shift_y_step, num_shift_y_iter = self._set_up_grid(shift_y, 'Y direction shift')
        shift_z_start, shift_z_step, num_shift_z_iter = self._set_up_grid(shift_z, 'Z direction shift')

        if not force_exec:
            estimated_computation_time = self._estimate_calculation_time(num_rot_x_inter, num_rot_y_inter,
                                                                         num_rot_z_inter, num_shift_x_iter,
                                                                         num_shift_y_iter, num_shift_z_iter)
            if estimated_computation_time > GRIDSEARCHTIME1:
                return False, 'With current setup, computation time will exceed {} minute' \
                              'If you want to proceed, use option force'.format(estimated_computation_time / 60)

        for i_rot_x in range(num_rot_x_inter):
            rot_x_i = rot_x_start + i_rot_x * rot_x_step
            for i_rot_y in range(num_rot_y_inter):
                rot_y_i = rot_y_start + i_rot_y * rot_y_step
                for i_rot_z in range(num_rot_z_inter):
                    rot_z_i = rot_z_start + i_rot_z * rot_z_step
                    for i_shift_x in range(num_shift_x_iter):
                        x_shift_i = shift_x_start + i_shift_x * shift_x_step
                        for i_shift_y in range(num_shift_y_iter):
                            y_shift_i = shift_y_start + i_shift_y * shift_y_step
                            for i_shift_z in range(num_shift_z_iter):
                                z_shift_i = shift_z_start + i_shift_z * shift_z_step

                                # build the instrument
                                self._instrument_reducer.build_instrument(rot_x_i, rot_y_i,
                                                                          rot_z_i, x_shift_i,
                                                                          y_shift_i, z_shift_i)
                                # reduce
                                self._instrument_reducer.reduce_to_2theta_histogram()

                                # calculate the cost
                                cost_i = self._calculate_cost(record=True)

        return

    def monte_carlo_grid_search(self, rot_x, rot_y, rot_z, shift_x, shift_y, shift_z):
        """
        do grid search with Monte carlo random walk on grid
        :param rot_x:
        :param rot_y:
        :param rot_z:
        :param shift_x:
        :param shift_z:
        :return:
        """

        return
