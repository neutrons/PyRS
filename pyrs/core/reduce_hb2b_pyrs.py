# This is a prototype reduction engine for HB2B living independently from Mantid
import numpy as np
import uncertainties.unumpy as unp
from pyrs.core import instrument_geometry
from pyrs.core.detector_model import DetectorModel
from pyrs.utilities import checkdatatypes
from pyrs.utilities.convertdatatypes import to_float
from typing import Optional


class ResidualStressInstrument:
    """
    This is a class to define HB2B instrument geometry and related calculation
    """

    def __init__(self, instrument_setup):
        """
        initialization
        :param instrument_setup: DENEXDetectorGeometry
        """
        # check input
        checkdatatypes.check_type("Instrument setup", instrument_setup, instrument_geometry.DENEXDetectorGeometry)

        # Instrument geometry parameters
        self._instrument_geom_params = instrument_setup

        self._pixel_matrix = None  # used by external after build_instrument: matrix for pixel positions
        self._pixel_2theta_matrix = None  # matrix for pixel's 2theta value
        self._pixel_eta_matrix = None  # matrix for pixel's eta value

        self._wave_length = None

        return

    def build_instrument(self, two_theta: float, l2: Optional[float] = None, instrument_calibration=None):
        """
        Build instrument considering calibration: pixel positions come from
        :meth:`pyrs.core.detector_model.DetectorModel.lab_positions` (see it for the order of operations,
        #1011), then each pixel's 2theta and eta are computed.

        :param float two_theta: 2theta position of the detector arm
        :param float l2: distance of detector from the center of rotation; None for the arm length
        :param DENEXDetectorShift instrument_calibration: DENEXDetectorShift or None (no calibration)
        :return: numpy.ndarray of pixel positions; shape = (num_columns) x (num_rows) x 3
        """
        two_theta = to_float("2theta", two_theta)
        if l2 is not None:
            l2 = to_float("L2", l2, 1e-2)

        # kept although lab_positions re-validates: a wrong type has always raised AssertionError here
        if instrument_calibration is not None:
            checkdatatypes.check_type(
                "Instrument calibration", instrument_calibration, instrument_geometry.DENEXDetectorShift
            )

        num_rows, num_columns = self._instrument_geom_params.detector_size
        self._pixel_matrix = DetectorModel(self._instrument_geom_params).lab_positions(
            np.arange(num_columns)[:, np.newaxis],
            np.arange(num_rows)[np.newaxis, :],
            two_theta,
            l2,
            instrument_calibration,
        )

        self._calculate_pixel_2theta()
        self._calculate_pixel_eta()

        return self._pixel_matrix

    def _calculate_pixel_2theta(self):
        """
        convert the pixel position matrix to 2theta. Result is recorded to self._pixel_2theta_matrix
        :return: array for pixel 2theta positions
        """
        # check whether instrument is well built
        if self._pixel_matrix is None:
            raise RuntimeError("Instrument has not been built yet. Pixel matrix is missing")

        det_pos_array = self._pixel_matrix.copy()

        if len(self._pixel_matrix[:].shape) == 3:
            # N x M x 3 array
            # convert detector position matrix to 2theta

            # normalize the detector position 2D array
            det_pos_norm_matrix = np.sqrt(
                self._pixel_matrix[:][:, :, 0] ** 2
                + self._pixel_matrix[:][:, :, 1] ** 2
                + self._pixel_matrix[:][:, :, 2] ** 2
            )
            twotheta_matrix = np.rad2deg(np.arccos(det_pos_array[:, :, 2] / det_pos_norm_matrix))
            return_value = twotheta_matrix
        else:
            # (N x M) x 3 array
            # convert detector positions array to 2theta array
            # normalize the detector position 2D array
            des_pos_array = self._pixel_matrix
            det_pos_norm_array = np.sqrt(
                des_pos_array[:, 0] ** 2 + des_pos_array[:, 1] ** 2 + des_pos_array[:, 2] ** 2
            )
            twotheta_array = np.rad2deg(np.arccos(det_pos_array[:, 2] / det_pos_norm_array))
            return_value = twotheta_array
        # END-IF-ELSE

        self._pixel_2theta_matrix = return_value

        return

    def _calculate_pixel_eta(self):
        """
        convert the pixel position matrix to 2theta. Result is recorded to self._pixel_eta_matrix
        :return:
        """
        # check whether instrument is well built
        if self._pixel_matrix is None:
            raise RuntimeError("Instrument has not been built yet. Pixel matrix is missing")

        # define

        det_pos_array = self._pixel_matrix.copy()

        if len(self._pixel_matrix[:].shape) == 3:
            # N x M x 3 array
            # convert detector position matrix to 2theta

            eta_matrix = 180.0 - np.rad2deg(np.arctan2(det_pos_array[:, :, 1], det_pos_array[:, :, 0]))
            eta_temp = eta_matrix.reshape(-1)
            index = np.where(eta_temp > 180.0)[0]
            eta_temp[index] -= 360
            eta_matrix = eta_temp.reshape(eta_matrix.shape)

            return_value = eta_matrix
        else:
            # (N x M) x 3 array
            # convert detector positions array to 2theta array
            # normalize the detector position 2D array

            eta_array = 180.0 - np.rad2deg(np.arctan2(det_pos_array[:, 1], det_pos_array[:, 0]))
            index = np.where(eta_array > 180.0)[0]
            eta_array[index] -= 360

            return_value = eta_array
        # END-IF-ELSE

        self._pixel_eta_matrix = return_value

        return

    def get_pixel_matrix(self):
        """
        return the 2D matrix of pixels' coordination

        :return: 3D array (2D of 1-D array) (N x M x 3)
        """
        if self._pixel_matrix is None:
            raise RuntimeError("Instrument has not been built yet")

        return self._pixel_matrix

    def get_pixels_2theta(self, dimension):
        """
        get the 2theta values for all the pixels

        :param int dimension: 1 for array, 2 for matrix
        :return:
        """
        if self._pixel_2theta_matrix is None:
            raise RuntimeError("2theta values for all the pixels are not calculated yet. (instrument not built")

        if dimension == 1:
            m, n = self._pixel_2theta_matrix.shape
            two_theta_values = self._pixel_2theta_matrix.reshape((m * n,))
        else:
            two_theta_values = self._pixel_2theta_matrix[:, :]

        return two_theta_values

    def get_eta_values(self, dimension):
        """
        get the 2theta values for all the pixels

        :param int dimension: 1 for array, 2 for matrix
        :return:
        """
        if self._pixel_eta_matrix is None:
            raise RuntimeError("2theta values for all the pixels are not calculated yet. (instrument not built")

        if dimension == 1:
            m, n = self._pixel_eta_matrix.shape
            eta_values = self._pixel_eta_matrix.reshape((m * n,))
        else:
            eta_values = self._pixel_eta_matrix[:, :]

        return eta_values

    def get_dspacing_value(self, dimension=1):
        """
        get the dspacing value for all pixels

        :param int dimension: 1 for array, 2 for matrix
        :return:
        """
        two_theta_array = self.get_pixels_2theta(dimension)
        print("[DB...BAT] 2theta range: ({}, {})".format(two_theta_array.min(), two_theta_array.max()))
        assert isinstance(two_theta_array, np.ndarray), "check"

        # convert to d-spacing
        d_spacing_array = 0.5 * self._wave_length / np.sin(np.deg2rad(0.5 * two_theta_array))
        assert isinstance(d_spacing_array, np.ndarray)

        print("[DB...BAT] Converted d-spacing range: ({}, {})".format(d_spacing_array.min(), d_spacing_array.max()))

        return d_spacing_array

    def get_pixel_array(self):
        """
        return the 1D array of pixels' coordination in the order of pixel IDs
        :return: 2D array ((N x M) x 3)
        """
        if self._pixel_matrix is None:
            raise RuntimeError("Instrument has not been built yet")

        num_x, num_y, num_z = self._pixel_matrix.shape
        if num_z != 3:
            raise RuntimeError("Pixel matrix shall have (x, y, 3) shape but not {}".format(self._pixel_matrix.shape))

        # reshape to 1D
        pixel_pos_array = self._pixel_matrix.reshape((num_x * num_y), 3)

        return pixel_pos_array

    def set_wavelength(self, w_l):
        """
        :param w_l: Neutron wavelength
        """
        self._wave_length = w_l


class PyHB2BReduction:
    """A class to reduce HB2B data in pure Python and numpy"""

    def __init__(self, instrument, wave_length=None):
        """
        initialize the instrument
        :param instrument
        """
        self._instrument = ResidualStressInstrument(instrument)

        if wave_length is not None:
            self._instrument.set_wavelength(wave_length)

        self._detector_2theta = None
        self._detector_l2 = None
        self._detector_counts = None

        # buffer for the last reduced data set
        # supposed to be 2 tuple for vector of 2theta and vector of intensity
        self._reduced_diffraction_data = None

        return

    @property
    def instrument(self):
        """
        Return instrument geometry/calculation instance
        :return:
        """
        return self._instrument

    def build_instrument(self, calibration):
        """Build an instrument for each pixel's position in cartesian coordinate

        :param DENEXDetectorShift calibration: DENEXDetectorShift from geometry calibration
        :return: 2D numpy array
        """
        if calibration is not None:
            checkdatatypes.check_type(
                "Instrument geometry calibrated shift", calibration, instrument_geometry.DENEXDetectorShift
            )

        self._instrument.build_instrument(self._detector_2theta, self._detector_l2, instrument_calibration=calibration)

        return

    def get_pixel_positions(self, is_matrix=False, corner_center=False):
        """Get pixels' positions

        Return the pixel matrix of the instrument

        :param boolean is_matrix: flag to output pixels in matrix. Otherwise, 1D array in order of pixel IDs
        :param boolean corner_center: only return 5 positions

        :return pixel_array: array of pixel positions
        :rtype: numpy.ndarray
        """

        if is_matrix:
            pixel_array = self._instrument.get_pixel_matrix()
        else:
            # 1D array
            pixel_array = self._instrument.get_pixel_array()

            if corner_center:
                # only return 5 positions: 4 corners and center
                pos_array = np.ndarray(shape=(5, 3), dtype="float")

                # num detectors
                num_dets = pixel_array.shape[0]

                linear_size = int(np.sqrt(num_dets))
                for i_pos, pos_tuple in enumerate(
                    [
                        (0, 0),
                        (0, linear_size - 1),
                        (linear_size - 1, 0),
                        (linear_size - 1, linear_size - 1),
                        (linear_size / 2, linear_size / 2),
                    ]
                ):
                    i_ws = pos_tuple[0] * linear_size + pos_tuple[1]
                    pos_array[i_pos] = pixel_array[i_ws]
                # END-FOR

                # re-assign output
                pixel_array = pos_array

            # END-IF

        return pixel_array

    def get_eta_value(self):
        """Get vector of solid angles for each pixel

        :rtype: numpy.ndarray

        """
        return self._instrument.get_eta_values(dimension=1)

    def reduce_to_2theta_histogram(self, two_theta_bins, mask_array, is_point_data=True, vanadium_counts_array=None):
        """Reduce the previously added detector raw counts to 2theta histogram (i.e., diffraction pattern)

        :param numpy.ndarray two_theta_bins: 2theta bin boundaries
        :param numpy.ndarray mask_array: mask: 1 to keep, 0 to mask (exclude)
        :param bool is_point_data: Flag whether the output is point data (numbers of X and Y are same)
        :param vanadium_counts_array: Vanadium counts array for normalization and efficiency calibration
        :type vanadium_counts_array: numpy.ndarray, optional
        :return: two_theta_bins, intensity_vector, variances_vector
        :rtype: numpy.ndarray
        """

        # Get two-theta-histogram vector
        checkdatatypes.check_numpy_arrays("2theta array", [two_theta_bins], 1, False)

        # Get the data (each pixel's 2theta and counts): the 2theta value is the absolute diffraction angle
        # that disregards the real 2theta value in the instrument coordinate system
        pixel_2theta_array = self._instrument.get_pixels_2theta(1)
        checkdatatypes.check_numpy_arrays(
            "Two theta and detector counts array",
            [pixel_2theta_array, self._detector_counts],
            1,
            check_same_shape=True,
        )  # optional check

        # Convert vector counts array's dtype to float
        counts_array = self._detector_counts.astype("float64")

        # Apply mask: act on local variable vec_counts and thus won't affect raw data
        if mask_array is not None:
            # mask detector counts, assuming detector mask and counts are in same order of pixel
            checkdatatypes.check_numpy_arrays("Counts vector and mask vector", [counts_array, mask_array], 1, True)
            # exclude mask from histogramming
            counts_array = counts_array[np.where(mask_array == 1)]
            pixel_2theta_array = pixel_2theta_array[np.where(mask_array == 1)]
            if vanadium_counts_array is not None:
                vanadium_counts_array = vanadium_counts_array[np.where(mask_array == 1)]
        else:
            # no mask: do nothing
            pass
        # END-IF-ELSE

        # use numpy.histogram
        two_theta_bins, intensity_vector, variances_vector = self.histogram_by_numpy(
            pixel_2theta_array, counts_array, two_theta_bins, is_point_data, vanadium_counts_array
        )

        # Record
        self._reduced_diffraction_data = two_theta_bins, intensity_vector, variances_vector

        return two_theta_bins, intensity_vector, variances_vector

    def set_experimental_data(self, two_theta: float, l2: Optional[float], raw_count_vec):
        """Set experimental data (for a sub-run)

        :param float two_theta: detector position
        :param float l2: detector distance from center of rotation
        :param numpy.ndarray raw_count_vec: detector raw counts
        :return:
        """
        self._detector_2theta = to_float("2-theta", two_theta, -180, 180)

        if l2 is not None:
            l2 = to_float("L2", l2, 1.0e-2)
        self._detector_l2 = l2

        checkdatatypes.check_numpy_arrays("Detector (raw) counts", [raw_count_vec], None, False)
        self._detector_counts = raw_count_vec

    def set_raw_counts(self, raw_count_vec):
        """Set experimental data (for a sub-run)

        :param numpy.ndarray raw_count_vec: detector raw counts
        :return:
        """
        checkdatatypes.check_numpy_arrays("Detector (raw) counts", [raw_count_vec], None, False)

        self._detector_counts = raw_count_vec

        return

    @staticmethod
    def histogram_by_numpy(pixel_2theta_array, pixel_count_array, two_theta_bins, is_point_data, vanadium_counts):
        """Histogram a data set (X, Y) by numpy histogram algorithm

        Assumption:
        1. pixel_2theta_array[i] and vec_counts[i] correspond to the same detector pixel

        :param numpy.ndarray pixel_2theta_array: 2theta (1D) array for each pixel
        :param numpy.ndarray pixel_count_array: count array (1D) for each pixel and paired to pixel_2theta_array
        :param numpy.ndarray pixel_2theta_array: 2-theta bin boundaries
        :param bool is_point_data: Output shall be point data; otherwise, histogram data
        :param None or numpy.ndarray vanadium_counts: Vanadium counts for normalization and efficiency calibration.
            It is allowed to be None
        :return: bins, data_hist, data_var
        :rtype: numpy.ndarray
        """

        checkdatatypes.check_numpy_arrays(
            "Pixel 2theta array, pixel counts array", [pixel_2theta_array, pixel_count_array], 1, True
        )

        # Exclude pixels with no vanadium counts
        if vanadium_counts is not None:
            vanadium_mask = vanadium_counts < 0.9
            pixel_2theta_array = np.ma.masked_where(vanadium_mask, pixel_2theta_array)
            pixel_count_array = np.ma.masked_where(vanadium_mask, pixel_count_array)
            vanadium_counts = np.ma.masked_where(vanadium_mask, vanadium_counts)

        # Exclude NaN and infinity regions
        masked_pixels = (np.isnan(pixel_count_array)) | (np.isinf(pixel_count_array))
        pixel_2theta_array = np.ma.masked_where(masked_pixels, pixel_2theta_array).compressed()
        pixel_count_array = np.ma.masked_where(masked_pixels, pixel_count_array).compressed()

        # construct data variance array
        pixel_var_array = np.sqrt(pixel_count_array)
        pixel_var_array[pixel_var_array == 0.0] = 1.0

        # Call numpy to histogram raw counts and variance
        data_hist, bin_edges = np.histogram(pixel_2theta_array, bins=two_theta_bins, weights=pixel_count_array)
        data_var, var_edges = np.histogram(pixel_2theta_array, bins=two_theta_bins, weights=pixel_var_array**2)
        data_var = np.sqrt(data_var)

        # get indexs in histograms that do not have neutron counts
        zero_count_mask = data_hist == 0.0

        # Optionally to normalize by number of pixels (sampling points) in the 2theta bin
        if vanadium_counts is not None:
            # Normalize by vanadium including efficiency calibration
            checkdatatypes.check_numpy_arrays("Vanadium counts", [vanadium_counts], 1, False)

            # Exclude NaN and infinity regions
            vanadium_counts = np.ma.masked_where(masked_pixels, vanadium_counts).compressed()

            # construct vanadium variance array
            vanadium_var = np.sqrt(vanadium_counts)
            vanadium_var[vanadium_var == 0.0] = 1.0

            # Call numpy to histogram vanadium counts and variance
            van_hist, be_temp = np.histogram(pixel_2theta_array, bins=two_theta_bins, weights=vanadium_counts)
            van_var, van_var_temp = np.histogram(pixel_2theta_array, bins=two_theta_bins, weights=vanadium_var**2)
            van_var = np.sqrt(van_var)

            # set indexs in histograms with no counts to 1
            data_hist[zero_count_mask] = 1.0
            van_hist[zero_count_mask] = 1.0
            data_var[zero_count_mask] = 1.0
            van_var[zero_count_mask] = 1.0

            # setup uncertainties arrays
            data_array = unp.uarray(data_hist, data_var)
            van_array = unp.uarray(van_hist, van_var)
            van_array_max = unp.uarray(
                van_hist[van_hist == van_hist.max()].max(), van_var[van_hist == van_hist.max()].max()
            )

            # normalize data
            normalized_data = data_array * (van_array_max / van_array)

            data_hist = unp.nominal_values(normalized_data)
            data_var = unp.std_devs(normalized_data)

        # END-IF-ELSE

        # set indexs in histograms that do not have any neutron counts to 0 with var = 1
        data_hist[zero_count_mask] = 0.0
        data_var[zero_count_mask] = 1.0

        # convert to point data as an option.  Use the center of the 2theta bin as new theta
        if is_point_data:
            # calculate bin centers
            bins = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        else:
            # return bin edges
            bins = bin_edges

        return bins, data_hist, data_var


# END-CLASS
