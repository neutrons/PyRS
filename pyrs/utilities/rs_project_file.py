# This is rs_scan_io.DiffractionFile's 2.0 version


class HydraProjectFile(object):
    """ Read and/or write an HB2B project to an HDF5 with entries for detector counts, sample logs, reduced data,
    fitted peaks and etc.
    All the import/export information will be buffered in order to avoid exception during operation
    """
    def __init__(self):
        """

        """
        return

    def init_file(self):
        """
        initialize a project file from scratch by opening it
        :return:
        """

        return

    def add_scan_counts(self):
        """ add raw detector counts collected in a single scan/Pt
        :return:
        """
        return

    def add_diffraction_data(self, unit):
        """ add reduced and corrected diffraction data in specified unit
        :param unit:
        :return:
        """
        return

    def add_experiment_infomation(self):
        """ add information about the experiment including scan indexes, sample logs, 2theta and etc
        :return:
        """
        return

    def get_scan_counts(self):
        """
        get the raw detector counts
        :return:
        """
        return

    def set_geometry(self):
        """
        set the instrument geometry information with calibration
        :return:
        """
        return

    def save_hydra_project(self):
        """
        convert all the information about project to HDF file
        :return:
        """
        return
