import numpy as np
from pyrs.interface.peak_fitting.config import LIST_AXIS_TO_PLOT
from pyrs.interface.peak_fitting.config import fit_dict as FIT_DICT
from typing import Optional, Tuple


class DataRetriever:

    def __init__(self, parent=None):
        self.parent = parent
        self.hidra_workspace = self.parent.hidra_workspace

    def get_data(self, name: str = 'Sub-runs', peak_index: int = 0) -> Tuple[np.ndarray, Optional[np.ndarray]]:

        try:
            keep_list = np.array(self.parent.fit_result.peakcollections[peak_index].get_exclude_list()) is False
        except AttributeError:
            keep_list = np.ones_like(self.hidra_workspace.get_sub_runs()) == 1

        if name == 'Sub-runs':
            return (np.array(self.hidra_workspace.get_sub_runs())[keep_list], None)

        # do not have typing information for object
        if name in LIST_AXIS_TO_PLOT['raw'].keys():  # type: ignore
            return (self.hidra_workspace._sample_logs[name][keep_list], None)

        if name == 'd-spacing':
            peak_collection = self.parent.fit_result.peakcollections[peak_index]
            _d_reference = np.float32(str(self.parent.ui.peak_range_table.item(peak_index, 3).text()))
            peak_collection.set_d_reference(values=_d_reference)
            values, error = peak_collection.get_dspacing_center()
            return (values[keep_list], error[keep_list])

        if name == 'microstrain':
            peak_collection = self.parent.fit_result.peakcollections[peak_index]
            values, error = peak_collection.get_strain(units='microstrain')
            return (values[keep_list], error[keep_list])

        # do not have typing information for object
        if name in LIST_AXIS_TO_PLOT['fit'].keys():  # type: ignore
            return self.get_fitted_value(peak=self.parent.fit_result.peakcollections[peak_index],
                                         value_to_display=name)

        raise RuntimeError('Do not know how to get values for "{}"'.format(name))

    def get_fitted_value(self, peak=None, value_to_display='Center'):
        """
        return the values and errors of the fitted parameters of the given peak
        :param peak:
        :param value_to_display:
        :return:
        """

        keep_list = np.array(peak.get_exclude_list()) is False

        value, error = peak.get_effective_params()

        mantid_value_to_display = FIT_DICT[value_to_display]
        value_selected = value[mantid_value_to_display]
        error_selected = error[mantid_value_to_display]
        return value_selected[keep_list], error_selected[keep_list]
