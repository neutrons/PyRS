"""
Containing classes serving for
1. instrument geometry
2. instrument geometry calibration
"""
from pyrs.utilities import checkdatatypes


class HydraSetup(object):
    """
    A class to handle anything to do with HB2B (HYDRA) including geometry, wavelength and calibration
    """
    def __init__(self, detector_setup, wavelength):
        """ Initialization
        :param detector_setup: (AnglerCameraDetectorGeometry) detector geometry setup
        :param wavelength: (float) wavelength
        """
        # check inputs
        checkdatatypes.check_type('Detector geometry setup', detector_setup, AnglerCameraDetectorGeometry)
        checkdatatypes.check_float_variable('Monochromator wavelength (A)', wavelength, (1.E-5, None))

        # set for original instrument setup defined by engineering
        self._geometry_setup = detector_setup
        self._wave_length = wavelength

        self._geometry_shift = None
        self._wave_length_shift = 0.

        # calibration state
        self._calibration_applied = False

        return

    def get_instrument_geometry(self, calibrated):
        """

        :param calibrated:
        :return:
        """
        if calibrated and self._geometry_shift is not None:
            return self._geometry_setup.apply_shift(self._geometry_shift)

        return self._geometry_setup

    def get_wavelength(self, calibrated):
        """
        Get wave length
        :param calibrated:
        :return:
        """
        if calibrated:
            return self._wave_length_shift + self._wave_length

        return self._wave_length

    def get_wavelength_shift(self):
        """

        :return:
        """
        return self._wave_length_shift

    def set_wavelength_calibration(self, wave_length_shift):
        """
        set the wave length shift
        :param wave_length_shift:
        :return:
        """
        checkdatatypes.check_float_variable('Wavelength shift from original value', wave_length_shift, (None, None))

        if self._wave_length + wave_length_shift < 0.1:
            raise RuntimeError('Wavelength shift {} to {} results in an unphysical value'.format(wave_length_shift))

        return

    def set_geometry_calibration(self, calibration):
        """
        Apply instrument geometry calibration to this instrument
        1. without changing
        :param calibration:
        :return:
        """

        return


class AnglerCameraDetectorGeometry(object):
    """
    A class to handle and save instrument geometry setup
    """
    def __init__(self, num_rows, num_columns, pixel_size_x, pixel_size_y, arm_length, calibrated):
        """
        Initialization of instrument geometry setup for 1 angler camera
        :param num_rows: number of rows of pixels in detector (number of pixels per column)
        :param num_columns: number of columns of pixels in detector (number of pixels per row)
        :param pixel_size_x: pixel size at X direction (along a row)
        :param pixel_size_y: pixel size at Y direction (along a column)
        :param arm_length: arm length
        :param calibrated: flag whether these values are calibrated
        """
        # check inputs
        checkdatatypes.check_float_variable('Arm length', arm_length, (1E-5, None))
        checkdatatypes.check_float_variable('Pixel size (x)', pixel_size_x, (1E-7, None))
        checkdatatypes.check_float_variable('Pixel size (y)', pixel_size_y, (1E-7, None))
        checkdatatypes.check_int_variable('Number of rows in detector', num_rows, (1, None))
        checkdatatypes.check_int_variable('Number of columns in detector', num_columns, (1, None))
        checkdatatypes.check_bool_variable('Flag indicating instrument setup been calibrated', calibrated)

        # geometry parameters for raw parameters
        self._arm_length = arm_length

        self._detector_rows = num_rows
        self._detector_columns = num_columns
        self._pixel_size_x = pixel_size_x
        self._pixel_size_y = pixel_size_y

        return

    def apply_shift(self, geometry_shift):
        """

        :param geometry_shift:
        :return:
        """
        checkdatatypes.check_type('Detector geometry shift', geometry_shift, AnglerCameraDetectorShift)

        self._arm_length += geometry_shift.center_shift_z

        return


class AnglerCameraDetectorShift(object):
    """
    A class to handle and save instrument geometry calibration information
    """
    def __init__(self, shift_x, shift_y, shift_z, rotation_x, rotation_y, rotation_z):
        """
        initialize
        """
        self._center_shift_x = 0.
        self._center_shift_y = 0.
        self._center_shift_z = 0.  # center shift Z along detector arm

        self._rotation_x = 0.  # in Y-Z plane (horizontal), i.e, flip
        self._rotation_y = 0.  # in X-Z plane along Y axis (vertical), i.e., rotate
        self._rotation_z = 0.  # in X-Y plane along Z axis, i.e., spin at detector center

        # Need data from client to finish this
        self.calibrated_wave_length = {'Si001': 1.00}

        return

    def __str__(self):
        nice = '[Calibration]\nShift:    {},  {},  {}\nRotation: {}, {}, {}' \
               ''.format(self.center_shift_x, self.center_shift_y, self.center_shift_z,
                         self.rotation_x, self.rotation_y, self.rotation_z)

        return nice

    @property
    def center_shift_x(self):
        return self._center_shift_x

    @center_shift_x.setter
    def center_shift_x(self, value):
        checkdatatypes.check_float_variable('Center shift along X direction', value, (None, None))
        self._center_shift_x = value

    @property
    def center_shift_y(self):
        return self._center_shift_y

    @center_shift_y.setter
    def center_shift_y(self, value):
        checkdatatypes.check_float_variable('Center shift along Y direction', value, (None, None))
        self._center_shift_y = value

    @property
    def center_shift_z(self):
        return self._center_shift_z

    @center_shift_z.setter
    def center_shift_z(self, value):
        checkdatatypes.check_float_variable('Center shift along Z direction', value, (None, None))
        self._center_shift_z = value

    @property
    def rotation_x(self):
        return self._rotation_x

    @rotation_x.setter
    def rotation_x(self, value):
        checkdatatypes.check_float_variable('Rotation along X direction', value, (-360, 360))
        self._rotation_x = value

    @property
    def rotation_y(self):
        return self._rotation_y

    @rotation_y.setter
    def rotation_y(self, value):
        checkdatatypes.check_float_variable('Rotation along Y direction', value, (-360, 360))
        self._rotation_y = value

    @property
    def rotation_z(self):
        return self._rotation_z

    @rotation_z.setter
    def rotation_z(self, value):
        checkdatatypes.check_float_variable('Rotation along Z direction', value, (-360, 360))
        self._rotation_z = value