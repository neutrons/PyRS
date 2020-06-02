# This is a prototype reduction engine for HB2B living independently from Mantid
import numpy as np
import numpy
from pyrs.core import instrument_geometry
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
        :param instrument_setup:
        """
        # check input
        checkdatatypes.check_type('Instrument setup', instrument_setup,
                                  instrument_geometry.AnglerCameraDetectorGeometry)

        # Instrument geometry parameters
        self._instrument_geom_params = instrument_setup

        # Pixels' positions without calibration. It is kept stable upon calibration values (shifts) and arm (000 plane)
        self._raw_pixel_matrix = self._set_uncalibrated_pixels()  # never been used for external client: len(shape) = 3

        self._pixel_matrix = None  # used by external after build_instrument: matrix for pixel positions
        self._pixel_2theta_matrix = None  # matrix for pixel's 2theta value
        self._pixel_eta_matrix = None  # matrix for pixel's eta value

        self._wave_length = None

        return

    @staticmethod
    def _rotate_detector(detector_matrix, rotation_matrix):
        """
        rotate instrument
        :param detector_matrix:
        :param rotation_matrix:
        :return:
        """
        rotate_det = np.zeros(detector_matrix.shape, dtype='float')

        for i in range(3):
            for j in range(3):
                temp_det = rotation_matrix[i, j] * detector_matrix[:, :, j]
                rotate_det[:, :, i] += temp_det[:, :]

        return rotate_det

    def _set_uncalibrated_pixels(self):
        """
        set up a matrix of pixels instrument on XY plane (Z=0)
        Note:
          1. the pixel matrix is set up such that with a simple reshape to 1D, the order of the pixel ID is ordered
             from lower left corner, going up and then going right.
          2. this is not a useful geometry because arm length is not set.
        :return: numpy.ndarray; shape = (num_rows) x (num_cols) x 3
        """
        assert self._instrument_geom_params is not None, 'Initial instrument setup is not set yet'

        # build raw instrument/pixel matrix
        num_rows, num_columns = self._instrument_geom_params.detector_size
        pixel_size_x, pixel_size_y = self._instrument_geom_params.pixel_dimension
        # arm_length = self._instrument_geom_params.arm_length

        # instrument is a N x M matrix, each element has
        pixel_matrix = np.ndarray(shape=(num_rows, num_columns, 3), dtype='float')

        # set Y as different from each row
        start_y_pos = -(num_rows * 0.5 - 0.5) * pixel_size_y
        start_x_pos = (num_columns * 0.5 - 0.5) * pixel_size_x

        for row_index in range(num_rows):
            pixel_matrix[row_index, :, 1] = start_y_pos + float(row_index) * pixel_size_y
        # set X as different from each column
        for col_index in range(num_columns):
            pixel_matrix[:, col_index, 0] = start_x_pos - float(col_index) * pixel_size_x
        # set Z: zero at origin
        pixel_matrix[:, :, 2] = 0.

        # Transpose is required to match instrument pixel ID arrangement:
        # This is the only pixel positions to be transposed in the instrument setup
        # Causing Error?  FIXME -
        pixel_matrix = pixel_matrix.transpose((1, 0, 2))

        return pixel_matrix

    def build_instrument(self, two_theta: float, l2: Optional[float] = None, instrument_calibration=None):
        """
        build instrument considering calibration
        step 1: rotate instrument according to the calibration
        step 2: rotate instrument about 2theta
        :param two_theta
        :param instrument_calibration: AnglerCameraDetectorShift or None (no calibration)
        :return:
        """
        # Check input
        two_theta = to_float('2theta', two_theta)
        # Check or set L2
        if l2 is None:
            l2 = self._instrument_geom_params.arm_length
        else:
            l2 = to_float('L2', l2, 1E-2)

        # print('[DB...L101] Build instrument: 2theta = {}, arm = {} (diff to default = {})'
        #       ''.format(two_theta, l2, l2 - self._instrument_geom_params.arm_length))

        # make a copy from raw (constant position)
        self._pixel_matrix = self._raw_pixel_matrix.copy()

        # Check and set instrument calibration
        if instrument_calibration is not None:
            # check type
            checkdatatypes.check_type('Instrument calibration', instrument_calibration,
                                      instrument_geometry.AnglerCameraDetectorShift)

            # shift center
            self._pixel_matrix[:, :, 0] += instrument_calibration.center_shift_x
            self._pixel_matrix[:, :, 1] += instrument_calibration.center_shift_y

            # rotation around instrument center
            # get rotation matrix at origin (for flip, spin and vertical): all data from calibration value
            rot_x_flip = instrument_calibration.rotation_x * np.pi / 180.
            rot_y_flip = instrument_calibration.rotation_y * np.pi / 180.
            rot_z_spin = instrument_calibration.rotation_z * np.pi / 180.
            calib_matrix = self.generate_rotation_matrix(rot_x_flip, rot_y_flip, rot_z_spin)
            # print ('[DB...BAT] Calibration rotation matrix:\n{}'.format(calib_matrix))
            # and rotate at origin
            self._pixel_matrix = self._rotate_detector(self._pixel_matrix, calib_matrix)
        # END-IF-ELSE

        # push to +Z at length of detector arm
        arm_l2 = l2
        if instrument_calibration is not None:
            # Apply the shift on Z (arm length)
            arm_l2 += instrument_calibration.center_shift_z
        # END-IF
        self._pixel_matrix[:, :, 2] += arm_l2

        # rotate detector (2theta) if it is not zero
        self.rotate_detector_2theta(two_theta)

        return self._pixel_matrix

    def rotate_detector_2theta(self, det_2theta):
        """Rotate detector, i.e., change 2theta value of the detector

        Parameters
        ----------
        det_2theta : float
            detector's 2theta (motor) position in degree

        Returns
        -------
        numpy.ndarray
            multiple dimensional array for detector positions
        """
        det_2theta = float(det_2theta)
        if abs(det_2theta) > 1.E-7:
            two_theta_rad = np.deg2rad(det_2theta)
            two_theta_rot_matrix = self._cal_rotation_matrix_y(two_theta_rad)
            self._pixel_matrix = self._rotate_detector(self._pixel_matrix, two_theta_rot_matrix)

        # get 2theta and eta
        self._calculate_pixel_2theta()
        self._calculate_pixel_eta()

        return self._pixel_matrix

    def _calculate_pixel_2theta(self):
        """
        convert the pixel position matrix to 2theta. Result is recorded to self._pixel_2theta_matrix
        :return:
        """
        # check whether instrument is well built
        if self._pixel_matrix is None:
            raise RuntimeError('Instrument has not been built yet. Pixel matrix is missing')

        # define
        # k_in_vec = [0, 0, 1]

        det_pos_array = self._pixel_matrix.copy()

        if len(self._pixel_matrix[:].shape) == 3:
            # N x M x 3 array
            # convert detector position matrix to 2theta

            # normalize the detector position 2D array
            det_pos_norm_matrix = np.sqrt(self._pixel_matrix[:][:, :, 0] ** 2 +
                                          self._pixel_matrix[:][:, :, 1] ** 2 +
                                          self._pixel_matrix[:][:, :, 2] ** 2)
            twotheta_matrix = np.arccos(det_pos_array[:, :, 2] / det_pos_norm_matrix) * 180 / np.pi
            return_value = twotheta_matrix
        else:
            # (N x M) x 3 array
            # convert detector positions array to 2theta array
            # normalize the detector position 2D array
            des_pos_array = self._pixel_matrix
            det_pos_norm_array = np.sqrt(des_pos_array[:, 0] ** 2 +
                                         des_pos_array[:, 1] ** 2 + des_pos_array[:, 2] ** 2)
            twotheta_array = np.arccos(det_pos_array[:, 2] / det_pos_norm_array) * 180 / np.pi
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
            raise RuntimeError('Instrument has not been built yet. Pixel matrix is missing')

        # define

        det_pos_array = self._pixel_matrix.copy()

        if len(self._pixel_matrix[:].shape) == 3:
            # N x M x 3 array
            # convert detector position matrix to 2theta

            eta_matrix = 180. - np.arctan2(det_pos_array[:, :, 1], det_pos_array[:, :, 0]) * 180 / np.pi
            eta_temp = eta_matrix.reshape(-1)
            index = np.where(eta_temp > 180.)[0]
            eta_temp[index] -= 360
            eta_matrix = eta_temp.reshape(eta_matrix.shape)

            return_value = eta_matrix
        else:
            # (N x M) x 3 array
            # convert detector positions array to 2theta array
            # normalize the detector position 2D array

            eta_array = 180. - np.arctan2(det_pos_array[:, 1], det_pos_array[:, 0]) * 180 / np.pi
            index = np.where(eta_array > 180.)[0]
            eta_array[index] -= 360

            return_value = eta_array
        # END-IF-ELSE

        self._pixel_eta_matrix = return_value

        return

    def generate_rotation_matrix(self, rot_x_rad, rot_y_rad, rot_z_rad):
        """
        build a ration matrix with 3 orthognol directions
        :param rot_x_rad: rotation about X-axis in rad (flip forward/backward)
        :param rot_y_rad: rotation about Y-axis in rad (vertical rotation)
        :param rot_z_rad: rotation about Z-axis in rad (spin)
        :return:
        """
        rot_x_matrix = self._cal_rotation_matrix_x(rot_x_rad)
        rot_y_matrix = self._cal_rotation_matrix_y(rot_y_rad)
        rot_z_matrix = self._cal_rotation_matrix_z(rot_z_rad)

        rotation_matrix = rot_x_matrix * rot_y_matrix * rot_z_matrix

        return rotation_matrix

    @staticmethod
    def _cal_rotation_matrix_x(angle_rad):
        """
        build rotation matrix around X-axis
        :param angle_rad:
        :return:
        """
        rotate_matrix = numpy.matrix([[1., 0., 0.],
                                      [0., numpy.cos(angle_rad), -numpy.sin(angle_rad)],
                                      [0., numpy.sin(angle_rad), numpy.cos(angle_rad)]],
                                     'float')

        return rotate_matrix

    @staticmethod
    def _cal_rotation_matrix_y(angle_rad):
        """
        build rotation matrix around Y-axis
        :param angle_rad:
        :return:
        """
        rotate_matrix = numpy.matrix([[numpy.cos(angle_rad), 0., numpy.sin(angle_rad)],
                                      [0., 1., 0.],
                                      [-numpy.sin(angle_rad), 0., numpy.cos(angle_rad)]],
                                     'float')

        return rotate_matrix

    @staticmethod
    def _cal_rotation_matrix_z(angle_rad):
        """
        build rotation matrix around Z-axis
        :param angle_rad:
        :return:
        """
        rotate_matrix = numpy.matrix([[numpy.cos(angle_rad), -numpy.sin(angle_rad), 0.],
                                      [numpy.sin(angle_rad), numpy.cos(angle_rad), 0.],
                                      [0., 0., 1.]],
                                     'float')

        return rotate_matrix

    def get_pixel_matrix(self):
        """
        return the 2D matrix of pixels' coordination
        :return: 3D array (2D of 1-D array) (N x M x 3)
        """
        if self._pixel_matrix is None:
            raise RuntimeError('Instrument has not been built yet')

        return self._pixel_matrix

    def get_pixels_2theta(self, dimension):
        """
        get the 2theta values for all the pixels
        :param dimension: 1 for array, 2 for matrix
        :return:
        """
        if self._pixel_2theta_matrix is None:
            raise RuntimeError('2theta values for all the pixels are not calculated yet. (instrument not built')

        if dimension == 1:
            m, n = self._pixel_2theta_matrix.shape
            two_theta_values = self._pixel_2theta_matrix.reshape((m * n,))
        else:
            two_theta_values = self._pixel_2theta_matrix[:, :]

        return two_theta_values

    def get_eta_values(self, dimension):
        """
                get the 2theta values for all the pixels
                :param dimension: 1 for array, 2 for matrix
                :return:
        """
        if self._pixel_eta_matrix is None:
            raise RuntimeError('2theta values for all the pixels are not calculated yet. (instrument not built')

        if dimension == 1:
            m, n = self._pixel_eta_matrix.shape
            eta_values = self._pixel_eta_matrix.reshape((m * n,))
        else:
            eta_values = self._pixel_eta_matrix[:, :]

        return eta_values

    def get_dspacing_value(self, dimension=1):
        """
        get the dspacing value for all pixels
        :param dimension:
        :return:
        """
        two_theta_array = self.get_pixels_2theta(dimension)
        print('[DB...BAT] 2theta range: ({}, {})'
              ''.format(two_theta_array.min(), two_theta_array.max()))
        assert isinstance(two_theta_array, numpy.ndarray), 'check'

        # convert to d-spacing
        d_spacing_array = 0.5 * self._wave_length / numpy.sin(0.5 * two_theta_array * numpy.pi / 180.)
        assert isinstance(d_spacing_array, numpy.ndarray)

        print('[DB...BAT] Converted d-spacing range: ({}, {})'
              ''.format(d_spacing_array.min(), d_spacing_array.max()))

        return d_spacing_array

    def get_pixel_array(self):
        """
        return the 1D array of pixels' coordination in the order of pixel IDs
        :return: 2D array ((N x M) x 3)
        """
        if self._pixel_matrix is None:
            raise RuntimeError('Instrument has not been built yet')

        num_x, num_y, num_z = self._pixel_matrix.shape
        if num_z != 3:
            raise RuntimeError('Pixel matrix shall have (x, y, 3) shape but not {}'
                               ''.format(self._pixel_matrix.shape))

        # reshape to 1D
        pixel_pos_array = self._pixel_matrix.reshape((num_x * num_y), 3)

        return pixel_pos_array

    def set_wavelength(self, w_l):
        self._wave_length = w_l


class PyHB2BReduction:
    """ A class to reduce HB2B data in pure Python and numpy
    """

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
        """ Build an instrument for each pixel's position in cartesian coordinate
        :param calibration: AnglerCameraDetectorShift from geometry calibration
        :return: 2D numpy array
        """
        if calibration is not None:
            checkdatatypes.check_type('Instrument geometry calibrated shift', calibration,
                                      instrument_geometry.AnglerCameraDetectorShift)

        self._instrument.build_instrument(self._detector_2theta, self._detector_l2,
                                          instrument_calibration=calibration)

        return

    def build_instrument_prototype(self, two_theta, arm_length, arm_length_shift, center_shift_x, center_shift_y,
                                   rot_x_flip, rot_y_flip, rot_z_spin):
        """
        build an instrument
        :param two_theta: 2theta position of the detector panel.  It shall be negative to sample log value
        :param arm_length_shift: shift along Z-direction (arm length)
        :param center_shift_x:
        :param center_shift_y:
        :param rot_x_flip:
        :param rot_y_flip:
        :param rot_z_spin:
        :return: 2D numpy array
        """
        print('[INFO] Building instrument: 2theta @ {}'.format(two_theta))

        calibration = instrument_geometry.AnglerCameraDetectorShift(
            arm_length_shift, center_shift_x, center_shift_y, rot_x_flip, rot_y_flip, rot_z_spin)

        self._instrument.build_instrument(two_theta=two_theta, l2=arm_length,
                                          instrument_calibration=calibration)

        return

    def rotate_two_theta(self, two_theta_0, two_theta_1):
        """
        build an instrument
        :param two_theta_0: inital 2theta position of the detector panel.
        :param two_theta_1: final 2theta position of the detector panel.
        """
        print('[INFO] Rotating: 2theta from {} to {}'.format(two_theta_0, two_theta_1))
        self._instrument.rotate_detector(two_theta_1 - two_theta_0)

        return

    def get_pixel_positions(self, is_matrix=False, corner_center=False):
        """Get pixels' positions

        Return the pixel matrix of the instrument built

        Parameters
        ----------
        is_matrix: boolean
            flag to output pixels in matrix. Otherwise, 1D array in order of pixel IDs
        corner_center

        Returns
        -------

        """
        if is_matrix:
            pixel_array = self._instrument.get_pixel_matrix()
        else:
            # 1D array
            pixel_array = self._instrument.get_pixel_array()

            if corner_center:
                # only return 5 positions: 4 corners and center
                pos_array = numpy.ndarray(shape=(5, 3), dtype='float')

                # num detectors
                num_dets = pixel_array.shape[0]

                linear_size = int(np.sqrt(num_dets))
                for i_pos, pos_tuple in enumerate([(0, 0), (0, linear_size - 1),
                                                   (linear_size - 1, 0), (linear_size - 1, linear_size - 1),
                                                   (linear_size / 2, linear_size / 2)]):
                    i_ws = pos_tuple[0] * linear_size + pos_tuple[1]
                    pos_array[i_pos] = pixel_array[i_ws]
                # END-FOR

                # re-assign output
                pixel_array = pos_array

            # END-IF

        return pixel_array

    def get_eta_value(self):
        """Get solid angle values for each pixel

        Returns
        -------
        numpy.ndarray

        """
        return self._instrument.get_eta_values(dimension=1)

    def reduce_to_2theta_histogram(self, two_theta_bins, mask_array,
                                   is_point_data=True, vanadium_counts_array=None):
        """Reduce the previously added detector raw counts to 2theta histogram (i.e., diffraction pattern)

        Parameters
        ----------
        two_theta_bins : numpy.ndarray
            2theta bin boundaries to binned to
        mask_array : numpy.ndarray or None
            mask: 1 to keep, 0 to mask (exclude)
        is_point_data : bool
            Flag whether the output is point data (numbers of X and Y are same)
        vanadium_counts_array : None or numpy.ndarray
            Vanadium counts array for normalization and efficiency calibration

        Returns
        -------
        numpy.ndarray, numpy.ndarray, numpy.ndarray
            2theta vector, intensity vector, and variances_vector

        """
        # Get two-theta-histogram vector
        checkdatatypes.check_numpy_arrays('2theta array', [two_theta_bins], 1, False)

        # Get the data (each pixel's 2theta and counts): the 2theta value is the absolute diffraction angle
        # that disregards the real 2theta value in the instrument coordinate system
        pixel_2theta_array = self._instrument.get_pixels_2theta(1)
        checkdatatypes.check_numpy_arrays('Two theta and detector counts array',
                                          [pixel_2theta_array, self._detector_counts], 1,
                                          check_same_shape=True)  # optional check

        # Convert vector counts array's dtype to float
        counts_array = self._detector_counts.astype('float64')

        # print('[INFO] PyRS.Instrument: pixels 2theta range: ({}, {}) vs 2theta histogram range: ({}, {})'
        #       ''.format(pixel_2theta_array.min(), pixel_2theta_array.max(), two_theta_bins.min(),
        #                 two_theta_bins.max()))

        # Apply mask: act on local variable vec_counts and thus won't affect raw data
        if mask_array is not None:
            # mask detector counts, assuming detector mask and counts are in same order of pixel
            checkdatatypes.check_numpy_arrays('Counts vector and mask vector',
                                              [counts_array, mask_array], 1, True)
            # exclude mask from histogramming
            counts_array = counts_array[np.where(mask_array == 1)]
            pixel_2theta_array = pixel_2theta_array[np.where(mask_array == 1)]
            if vanadium_counts_array is not None:
                vanadium_counts_array = vanadium_counts_array[np.where(mask_array == 1)]
        else:
            # no mask: do nothing
            pass
        # END-IF-ELSE

        # Histogram:
        # NOTE: input 2theta_range may not be accurate because 2theta max may not be on the full 2-theta tick
        # TODO - If use vanadium for normalization, then (1) flag to normalize by pixel count and (2) efficiency
        #        are not required anymore but both of them will be replaced by integrated vanadium counts

        # use numpy.histogram
        two_theta_bins, intensity_vector, variances_vector = self.histogram_by_numpy(pixel_2theta_array,
                                                                                     counts_array,
                                                                                     two_theta_bins,
                                                                                     is_point_data,
                                                                                     vanadium_counts_array)

        # Record
        self._reduced_diffraction_data = two_theta_bins, intensity_vector, variances_vector

        return two_theta_bins, intensity_vector, variances_vector

    def set_experimental_data(self, two_theta: float, l2: Optional[float], raw_count_vec):
        """ Set experimental data (for a sub-run)
        :param two_theta: detector position
        :param raw_count_vec: detector raw counts
        :return:
        """
        self._detector_2theta = to_float('2-theta', two_theta, -180, 180)

        if l2 is not None:
            l2 = to_float('L2', l2, 1.E-2)
        self._detector_l2 = l2

        checkdatatypes.check_numpy_arrays('Detector (raw) counts', [raw_count_vec], None, False)
        self._detector_counts = raw_count_vec

    def set_raw_counts(self, raw_count_vec):
        """ Set experimental data (for a sub-run)
        :param raw_count_vec: detector raw counts
        :return:
        """
        checkdatatypes.check_numpy_arrays('Detector (raw) counts', [raw_count_vec], None, False)

        self._detector_counts = raw_count_vec

        return

    @staticmethod
    def histogram_by_numpy(pixel_2theta_array, pixel_count_array, two_theta_bins, is_point_data, vanadium_counts):
        """Histogram a data set (X, Y) by numpy histogram algorithm

        Assumption:
        1. pixel_2theta_array[i] and vec_counts[i] correspond to the same detector pixel

        Parameters
        ----------
        pixel_2theta_array : ~numpy.ndarray
            2theta (1D) array for each pixel
        pixel_count_array : numpy.ndarray
            count array (1D) for each pixel and paired to pixel_2theta_array
        two_theta_bins : numpy.ndarray
            2-theta bin boundaries
        is_point_data : bool
            Output shall be point data; otherwise, histogram data
        vanadium_counts : None or numpy.ndarray
            Vanadium counts for normalization and efficiency calibration.  It is allowed to be None

        Returns
        -------

        """
        # Check inputs
        checkdatatypes.check_numpy_arrays('Pixel 2theta array, pixel counts array',
                                          [pixel_2theta_array, pixel_count_array],
                                          1, True)

        # Exclude pixels with no vanadium counts
        if vanadium_counts is not None:
            vandium_mask = vanadium_counts < 0.9
            pixel_2theta_array = np.ma.masked_where(vandium_mask, pixel_2theta_array)
            pixel_count_array = np.ma.masked_where(vandium_mask, pixel_count_array)
            vanadium_counts = np.ma.masked_where(vandium_mask, vanadium_counts)

        # Exclude NaN and infinity regions
        masked_pixels = (np.isnan(pixel_count_array)) | (np.isinf(pixel_count_array))
        pixel_2theta_array = np.ma.masked_where(masked_pixels, pixel_2theta_array).compressed()
        pixel_count_array = np.ma.masked_where(masked_pixels, pixel_count_array).compressed()

        # construct variance array
        pixel_var_array = np.sqrt(pixel_count_array)
        pixel_var_array[pixel_var_array == 0.0] = 1.

        # Call numpy to histogram raw counts and variance
        hist, bin_edges = np.histogram(pixel_2theta_array, bins=two_theta_bins, weights=pixel_count_array)
        var, var_edges = np.histogram(pixel_2theta_array, bins=two_theta_bins, weights=pixel_var_array**2)
        var = np.sqrt(var)

        # Optionally to normalize by number of pixels (sampling points) in the 2theta bin
        if vanadium_counts is not None:
            # Normalize by vanadium including efficiency calibration
            checkdatatypes.check_numpy_arrays('Vanadium counts', [vanadium_counts], 1, False)

            # Exclude NaN and infinity regions
            vanadium_counts = np.ma.masked_where(masked_pixels, vanadium_counts).compressed()

            # construct variance array
            vanadium_var = np.sqrt(vanadium_counts)
            vanadium_var[vanadium_var == 0.0] = 1.

            # Call numpy to histogram vanadium counts and variance
            hist_bin, be_temp = np.histogram(pixel_2theta_array, bins=two_theta_bins, weights=vanadium_counts)
            van_var, van_var_temp = np.histogram(pixel_2theta_array, bins=two_theta_bins, weights=vanadium_var**2)
            van_var = np.sqrt(van_var)

            # Find out the bin where there is either no vanadium count or no pixel's located
            # Mask these bins by NaN
            # make sure it is float
            hist_bin = hist_bin.astype(float)
            hist_bin[np.where(hist_bin < 1E-10)] = np.nan

            # propogation of error
            var = np.sqrt((var / hist)**2 + (van_var / hist_bin)**2)

            # Normalize diffraction data
            hist /= hist_bin  # normalize
            var *= hist

        # END-IF-ELSE

        # convert to point data as an option.  Use the center of the 2theta bin as new theta
        if is_point_data:
            # calculate bin centers
            bins = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        else:
            # return bin edges
            bins = bin_edges

        return bins, hist, var
# END-CLASS
