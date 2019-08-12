# This is a prototype reduction engine for HB2B living independently from Mantid
import numpy as np
import numpy
import calibration_file_io
from pyrs.utilities import checkdatatypes


class ResidualStressInstrument(object):
    """
    This is a class to define HB2B instrument geometry and related calculation
    """
    def __init__(self, instrument_setup):
        """
        initialization
        :param instrument_setup:
        """
        # check input
        checkdatatypes.check_type('Instrument setup', instrument_setup, calibration_file_io.InstrumentSetup)

        self._init_setup = instrument_setup

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
        assert self._init_setup is not None, 'Initial instrument setup is not set yet'

        # build raw instrument/pixel matrix
        num_rows = self._init_setup.detector_rows
        num_columns = self._init_setup.detector_columns
        pixel_size_x = self._init_setup.pixel_size_x
        pixel_size_y = self._init_setup.pixel_size_y
        # arm_length = self._init_setup.arm_length

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

    def build_instrument(self, two_theta, instrument_calibration):
        """
        build instrument considering calibration
        step 1: rotate instrument according to the calibration
        step 2: rotate instrument about 2theta
        :param two_theta
        :param instrument_calibration:
        :return:
        """
        # check input
        checkdatatypes.check_type('Instrument calibration', instrument_calibration,
                                  calibration_file_io.ResidualStressInstrumentCalibration)
        checkdatatypes.check_float_variable('2theta', two_theta, (None, None))

        # make a copy
        self._pixel_matrix = self._raw_pixel_matrix.copy()

        # shift center
        self._pixel_matrix[:, :, 0] += instrument_calibration.center_shift_x
        self._pixel_matrix[:, :, 1] += instrument_calibration.center_shift_y

        # get rotation matrix at origin (for flip, spin and vertical): all data from calibration value
        rot_x_flip = instrument_calibration.rotation_x * np.pi / 180.
        rot_y_flip = instrument_calibration.rotation_y * np.pi / 180.
        rot_z_spin = instrument_calibration.rotation_z * np.pi / 180.
        calib_matrix = self.generate_rotation_matrix(rot_x_flip, rot_y_flip, rot_z_spin)
        # print ('[DB...BAT] Calibration rotation matrix:\n{}'.format(calib_matrix))
        # and rotate at origin
        self._pixel_matrix = self._rotate_detector(self._pixel_matrix, calib_matrix)

        # push to +Z at length of detector arm
        print ('[DEBUG...DELETE LATER] arm length = {}'.format(self._init_setup.arm_length))
        self._pixel_matrix[:, :, 2] += self._init_setup.arm_length + instrument_calibration.center_shift_z

        # rotate 2theta
        print ('[INFO] Build instrument with 2theta = {}'.format(two_theta))
        two_theta_rad = two_theta * np.pi / 180.
        two_theta_rot_matrix = self._cal_rotation_matrix_y(two_theta_rad)
        # print ('[DB...BAT] 2-theta rotation matrix:\n{}'.format(two_theta_rot_matrix))
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
        k_in_vec = [0, 0, 1]

        det_pos_array = self._pixel_matrix.copy()

        if len(self._pixel_matrix[:].shape) == 3:
            # N x M x 3 array
            # convert detector position matrix to 2theta

            # normalize the detector position 2D array
            det_pos_norm_matrix = np.sqrt(self._pixel_matrix[:][:, :, 0] ** 2 +
                                          self._pixel_matrix[:][:, :, 1] ** 2 +
                                          self._pixel_matrix[:][:, :, 2] ** 2)
            #### Instrument K_in is fixed to [0,0,1]
            # normalize pixel position for diffraction angle
            #for i_dir in range(3):
            #    det_pos_array[:, :, i_dir] /= det_pos_norm_matrix

            # convert to  2theta in degree
            #diff_angle_cos_matrix = det_pos_array[:, :, 0] * k_in_vec[0] + det_pos_array[:, :, 1] * k_in_vec[1] + \
            #                        det_pos_array[:, :, 2] * k_in_vec[2]
            #twotheta_matrix = np.arccos(diff_angle_cos_matrix) * 180 / np.pi
            twotheta_matrix = np.arccos(det_pos_array[:, :, 2] / det_pos_norm_matrix ) * 180 / np.pi

            return_value = twotheta_matrix
        else:
            # (N x M) x 3 array
            # convert detector positions array to 2theta array
            # normalize the detector position 2D array
            det_pos_norm_array = np.sqrt(det_pos_array[:, 0] ** 2 + det_pos_array[:, 1] ** 2 + det_pos_array[:, 2] ** 2)

            #### Instrument K_in is fixed to [0,0,1]            
            # normalize pixel position for diffraction angle
            #for i_dir in range(3):
            #    des_pos_array[:, i_dir] /= det_pos_norm_array

            # convert to  2theta in degree
            #diff_angle_cos_array = det_pos_array[:, 0] * k_in_vec[0] + det_pos_array[:, 1] * k_in_vec[1] + \
            #                       det_pos_array[:, 2] * k_in_vec[2]

            #twotheta_array = np.arccos(diff_angle_cos_array) * 180 / np.pi
            twotheta_array = np.arccos( det_pos_array[:, 2] / det_pos_norm_array ) * 180 / np.pi
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

            eta_matrix = 180. - np.arctan2(det_pos_array[:, :, 1],  det_pos_array[:, :,0] ) * 180 / np.pi
            eta_temp = eta_matrix.reshape(-1)
            index = np.where( eta_temp > 180. ) [0]
            eta_temp[index] -= 360
            eta_matrix = eta_temp.reshape(eta_matrix.shape)

            return_value = eta_matrix
        else:
            # (N x M) x 3 array
            # convert detector positions array to 2theta array
            # normalize the detector position 2D array

            eta_array = 180. - np.arctan2(det_pos_array[:, 1],  det_pos_array[:,0] ) * 180 / np.pi
            index = np.where( eta_array > 180. ) [0]
            eta_array[index] -= 360

            return_value = eta_array
        # END-IF-ELSE

        self._pixel_eta_matrix = return_value

        return

    def generate_rotation_matrix(self, rot_x_rad, rot_y_rad, rot_z_rad):
        """
        build a ration matrix with 3 orthoganol directions
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

    def get_2theta_values(self, dimension):
        """
        get the 2theta values for all the pixels
        :param dimension: 1 for array, 2 for matrix
        :return:
        """
        if self._pixel_2theta_matrix is None:
            raise RuntimeError('2theta values for all the pixels are not calculated yet. (instrument not built')

        if dimension == 1:
            m, n = self._pixel_2theta_matrix.shape
            two_theta_values = self._pixel_2theta_matrix.reshape((m*n,))
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
            eta_values = self._pixel_eta_matrix.reshape((m*n,))
        else:
            eta_values = self._pixel_eta_matrix[:, :]

        return eta_values

    def get_dspacing_value(self, dimension=1):
        """
        get the dspacing value for all pixels
        :param dimension:
        :return:
        """
        two_theta_array = self.get_2theta_values(dimension)
        print ('[DB...BAT] 2theta range: ({}, {})'
               ''.format(two_theta_array.min(), two_theta_array.max()))
        assert isinstance(two_theta_array, numpy.ndarray), 'check'

        # convert to d-spacing
        d_spacing_array = 0.5 * self._wave_length / numpy.sin(0.5 * two_theta_array * numpy.pi / 180.)
        assert isinstance(d_spacing_array, numpy.ndarray)

        print ('[DB...BAT] Converted d-spacing range: ({}, {})'
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

    def set_wave_length(self, w_l):
        self._wave_length = w_l


class PyHB2BReduction(object):
    """ A class to reduce HB2B data in pure Python and numpy
    """
    def __init__(self, instrument, wave_length=None):
        """
        initialize the instrument
        :param instrument
        """
        self._instrument = ResidualStressInstrument(instrument)

        if wave_length is not None:
            self._instrument.set_wave_length(wave_length)

        return

    def build_instrument(self, two_theta, arm_length_shift, center_shift_x, center_shift_y,
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
        print ('[INFO] Building instrument: 2theta @ {}'.format(two_theta))

        calibration = calibration_file_io.ResidualStressInstrumentCalibration()
        calibration.center_shift_z = arm_length_shift
        calibration.center_shift_x = center_shift_x
        calibration.center_shift_y = center_shift_y
        calibration.rotation_x = rot_x_flip
        calibration.rotation_y = rot_y_flip
        calibration.rotation_z = rot_z_spin

        self._instrument.build_instrument(two_theta, instrument_calibration=calibration)

        return

    def get_pixel_positions(self, is_matrix=False):
        """
        return the pixel matrix of the instrument built
        :param is_matrix: flag to output pixels in matrix. Otherwise, 1D array in order of pixel IDs
        :return:
        """
        if is_matrix:
            pixel_array = self._instrument.get_pixel_matrix()
        else:
            pixel_array = self._instrument.get_pixel_array()

        return pixel_array

    # TODO - TONIGHT 1 - Doc and clean
    # TODO - TONIGHT 0 - Normalize by num bins (aka vanadium as old saying)

    def get_eta_Values( self ):
        return self._instrument.get_eta_values(dimension=1)

    def reduce_to_2theta_histogram(self, counts_array, mask, num_bins, x_range=None,
                                   is_point_data=True, use_mantid_histogram=False):
        """ convert the inputs (detector matrix and counts to 2theta histogram)
        :param counts_array:
        :param mask: vector of masks
        :param num_bins:
        :param x_range: range of X value
        :param use_mantid_histogram: use Mantid ResampleX to compare numpy histogram
        :return: 2-tuple (bin edges, counts in histogram)
        """
        # get vector of X: 2theta
        pixel_array = self._instrument.get_pixel_array()

        # ... broken?
        # two_theta_array = self.convert_to_2theta(pixel_array)
        two_theta_array = self._instrument.get_2theta_values(dimension=1)
        checkdatatypes.check_numpy_arrays('Two theta array', [two_theta_array], 1, check_same_shape=False)

        # check with counts
        if two_theta_array.shape != counts_array.shape:
            raise RuntimeError('Counts (array) has a different ... blabla')

        # convert count type
        vec_counts = counts_array.astype('float64')
        print ('[INFO] PyRS.Instrument: 2theta range: {}, {}'.format(two_theta_array.min(),
                                                                     two_theta_array.max()))

        # check inputs of x range
        if x_range:
            checkdatatypes.check_tuple('X range', x_range, 2)
            if x_range[0] >= x_range[1]:
                raise RuntimeError('X range {} is not allowed'.format(x_range))

        # apply mask
        raw_counts = vec_counts.sum()
        if mask is not None:
            checkdatatypes.check_numpy_arrays('Counts vector and mask vector', [counts_array, mask], 1, True)
            vec_counts *= mask
            masked_counts = vec_counts.sum()
            num_masked = mask.shape[0] - mask.sum()
        else:
            masked_counts = raw_counts
            num_masked = 0
        # END-IF-ELSE
        print ('[INFO] Raw counts = {}, # Masked Pixels = {}, Counts in ROI = {}'
               ''.format(raw_counts, num_masked, masked_counts))

        # this is histogram data
        use_mantid_histogram = False   # TODO FIXME - Turned on for debugging!
        if not use_mantid_histogram:
            norm_bins = True
            bin_edges, hist = self.histogram_by_numpy(two_theta_array, vec_counts, x_range,
                                                      num_bins, is_point_data, norm_bins)

        else:
            # this is a branch used for testing against Mantid method
            bin_edges, hist = self.histogram_by_mantid(two_theta_array, vec_counts)
        # END-IF

        return bin_edges, hist

    # TODO - TEST 0 -
    def reduce_to_dspacing_histogram(self, counts_array, mask, num_bins, x_range=None,
                                     export_point_data=True, use_mantid_histogram=False):
        """
        w_l = 2d sin (0.5 * two_theta)
        :param counts_array:
        :param mask:
        :param num_bins:
        :param x_range:
        :param export_point_data:
        :param use_mantid_histogram:
        :return:
        """
        # get d-spacing
        d_space_vec = self._instrument.get_dspacing_value(dimension=1)
        checkdatatypes.check_numpy_arrays('dSpacing array', [d_space_vec], 1, check_same_shape=False)

        # check with counts
        if d_space_vec.shape != counts_array.shape:
            raise RuntimeError('Counts (array) has a different ... blabla')

        # convert count type
        vec_counts = counts_array.astype('float64')
        print ('[INFO] PyRS.Instrument: 2theta range: {}, {}'.format(d_space_vec.min(),
                                                                     d_space_vec.max()))

        # check inputs of x range
        if x_range:
            checkdatatypes.check_tuple('X range', x_range, 2)
            if x_range[0] >= x_range[1]:
                raise RuntimeError('X range {} is not allowed'.format(x_range))

        # apply mask
        raw_counts = vec_counts.sum()
        if mask is not None:
            checkdatatypes.check_numpy_arrays('Counts vector and mask vector', [counts_array, mask], 1, True)
            vec_counts *= mask
            masked_counts = vec_counts.sum()
            num_masked = mask.shape[0] - mask.sum()
        else:
            masked_counts = raw_counts
            num_masked = 0
        # END-IF-ELSE
        print ('[INFO] Raw counts = {}, # Masked Pixels = {}, Counts in ROI = {}'
               ''.format(raw_counts, num_masked, masked_counts))

        # this is histogram data
        norm_bins = True
        is_point_data = True
        bin_edges, hist = self.histogram_by_numpy(d_space_vec, vec_counts, x_range, num_bins,
                                                  is_point_data, norm_bins)

        return bin_edges, hist

    @staticmethod
    def histogram_by_mantid(two_theta_array, vec_counts, x_range, num_bins):
        from mantid.simpleapi import CreateWorkspace, SortXAxis, ResampleX
        import time
        # create a 1-spec workspace
        t0 = time.time()

        pixel_ids = numpy.arange(two_theta_array.shape[0])
        CreateWorkspace(DataX=two_theta_array, DataY=vec_counts, DataE=pixel_ids, NSpec=1,
                        OutputWorkspace='prototype')

        t1 = time.time()

        # Sort X-axis
        temp_ws = SortXAxis(InputWorkspace='prototype', OutputWorkspace='prot_sorted', Ordering='Ascending',
                            IgnoreHistogramValidation=True)
        temp_vec_y = temp_ws.readY(0)
        print ('[DEBUG] After SortXAxis: Y-range = ({}, {})'.format(temp_vec_y.min(), temp_vec_y.max()))
        t2 = time.time()

        # Resample
        binned = ResampleX(InputWorkspace='prot_sorted', OutputWorkspace='mantid_binned',
                           XMin=x_range[0], XMax=x_range[1],
                           NumberBins=num_bins, EnableLogging=False)

        t3 = time.time()

        print ('[STAT] Create workspace: {}\n\tSort: {}\n\tResampleX: {}'
               ''.format(t1 - t0, t2 - t0, t3 - t0))

        bin_edges = binned.readX(0)
        hist = binned.readY(0)
        pixel_ids = binned.readE(0)

        print ('[DB...BAT] Workspace Size: {}, {}, {}'.format(len(bin_edges), len(hist), len(pixel_ids)))

        return bin_edges, hist

    @staticmethod
    def histogram_by_numpy(two_theta_array, vec_counts, x_range, num_bins, is_point_data, norm_bins):
        hist, bin_edges = np.histogram(two_theta_array, bins=num_bins, range=x_range, weights=vec_counts)

        if norm_bins:
            vec_one = numpy.zeros(shape=vec_counts.shape) + 1
            hist_bin, bin_edges2 = np.histogram(two_theta_array, bins=num_bins, range=x_range, weights=vec_one)

            # avoid zero number of bins on any X
            for ibin in range(hist_bin.shape[0]):
                if hist_bin[ibin] < 1.E-4:  # zero
                    hist_bin[ibin] = 1.E10  # doesn't matter how big it is
            # END-FOR

            hist /= hist_bin
        # END-IF

        # bins information output
        bin_size_vec = (bin_edges[1:] - bin_edges[:-1])
        print ('[DB...BAT] Histograms Bins: X = [{}, {}]'.format(bin_edges[0], bin_edges[-1]))
        print ('[DB...BAT] Bin size = {}, Std = {}'.format(numpy.average(bin_size_vec), numpy.std(bin_size_vec)))

        # convert to point data
        if is_point_data:
            delta_bin = bin_edges[1] - bin_edges[0]
            bin_edges += delta_bin * 0.5
            bin_edges = bin_edges[:-1]

        return bin_edges, hist

# END-CLASS



