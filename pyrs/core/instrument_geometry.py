"""
Containing classes serving for
1. instrument geometry
2. instrument geometry calibration
"""
from pyrs.utilities import checkdatatypes
from pyrs.utilities.convertdatatypes import to_float, to_int


class HidraSetup:
    """A class to work with instrument geometry calculation

    Handle anything to do with HB2B (HiDRA) including geometry, wavelength and calibration

    """

    def __init__(self, detector_setup):
        """Initialization

        Initialization HB2B instrument setup

        Parameters
        ----------
        l1
        detector_setup
        """
        # check inputs
        checkdatatypes.check_type('Detector geometry setup', detector_setup, AnglerCameraDetectorGeometry)

        # set for original instrument setup defined by engineering
        self._geometry_setup = detector_setup
        self._single_wave_length = None

        self._geometry_shift = None
        self._calibrated_wave_length = None

        # calibration state
        self._calibration_applied = False

    def get_instrument_geometry(self, calibrated):
        """Get instrument geometry parameters

        Get HB2B geometry setup, raw or calibrated optionally

        Parameters
        ----------
        calibrated

        Returns
        -------
        GeometrySetup
            Geometry setup parameters
        """
        if calibrated and self._geometry_shift is not None:
            return self._geometry_setup.apply_shift(self._geometry_shift)

        return self._geometry_setup

    def get_wavelength(self, wave_length_tag):
        """Get wave length

        Get wave length for only calibrated

        Parameters
        ----------
        wave_length_tag: str
            user tag (as 111, 222) for wave length. None for single wave length

        Returns
        -------
        float
            wave length in A
        """
        if wave_length_tag is not None:
            raise NotImplementedError('Need use case to re-define the method')

        return self._calibrated_wave_length

    def get_wavelength_shift(self):
        return self._calibrated_wave_length

    @property
    def name(self):
        return 'HB2B'


class AnglerCameraDetectorGeometry:
    """
    A class to handle and save instrument geometry setup
    """

    def __init__(self, num_rows: int, num_columns: int,
                 pixel_size_x: float, pixel_size_y: float,
                 arm_length: float, calibrated: bool) -> None:
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
        self._arm_length = to_float('Arm length', arm_length, min_value=1E-5)
        self._pixel_size_x = to_float('Pixel size (x)', pixel_size_x, min_value=1E-7)
        self._pixel_size_y = to_float('Pixel size (y)', pixel_size_y, min_value=1E-7)
        self._detector_rows = to_int('Number of rows in detector', num_rows, min_value=1)
        self._detector_columns = to_int('Number of columns in detector', num_columns, min_value=1)
        # TODO this isn't used - BUG?
        checkdatatypes.check_bool_variable('Flag indicating instrument setup been calibrated', calibrated)

    def apply_shift(self, geometry_shift):
        checkdatatypes.check_type('Detector geometry shift', geometry_shift, AnglerCameraDetectorShift)

        self._arm_length += geometry_shift.center_shift_z

    @property
    def arm_length(self):
        """L2/arm length
        """
        return self._arm_length

    @property
    def detector_size(self):
        """Detector size (number of pixels)
        :return: number of rows, number of columns
        """
        return self._detector_rows, self._detector_columns

    @property
    def pixel_dimension(self):
        """ Each pixel's linear size at X (across columns) and Y (across rows) direction
        :return: size along X-axis, size along Y-axis in unit of meter
        """
        return self._pixel_size_x, self._pixel_size_y


class AnglerCameraDetectorShift:
    """
    A class to handle and save instrument geometry calibration information
    """

    def __init__(self, shift_x, shift_y, shift_z, rotation_x, rotation_y, rotation_z):
        """
        initialize
        """
        self._center_shift_x = shift_x
        self._center_shift_y = shift_y
        self._center_shift_z = shift_z  # center shift Z along detector arm

        self._rotation_x = rotation_x  # in Y-Z plane (horizontal), i.e, flip
        self._rotation_y = rotation_y  # in X-Z plane along Y axis (vertical), i.e., rotate
        self._rotation_z = rotation_z  # in X-Y plane along Z axis, i.e., spin at detector center

        # Need data from client to finish this
        self.calibrated_wave_length = {'Si001': 1.00}

    def __str__(self):
        nice = '[Calibration]\nShift:    {},  {},  {}\nRotation: {}, {}, {}' \
               ''.format(self.center_shift_x, self.center_shift_y, self.center_shift_z,
                         self.rotation_x, self.rotation_y, self.rotation_z)

        return nice

    @property
    def center_shift_x(self):
        return self._center_shift_x

    @center_shift_x.setter
    def center_shift_x(self, value: float) -> None:
        self._center_shift_x = to_float('Center shift along X direction', value)

    @property
    def center_shift_y(self):
        return self._center_shift_y

    @center_shift_y.setter
    def center_shift_y(self, value: float) -> None:
        self._center_shift_y = to_float('Center shift along Y direction', value)

    @property
    def center_shift_z(self):
        return self._center_shift_z

    @center_shift_z.setter
    def center_shift_z(self, value: float) -> None:
        self._center_shift_z = to_float('Center shift along Z direction', value)

    @property
    def rotation_x(self):
        return self._rotation_x

    @rotation_x.setter
    def rotation_x(self, value: float) -> None:
        self._rotation_x = to_float('Rotation along X direction', value, -360, 360)

    @property
    def rotation_y(self):
        return self._rotation_y

    @rotation_y.setter
    def rotation_y(self, value: float) -> None:
        self._rotation_y = to_float('Rotation along Y direction', value, -360, 360)

    @property
    def rotation_z(self):
        return self._rotation_z

    @rotation_z.setter
    def rotation_z(self, value: float) -> None:
        self._rotation_z = to_float('Rotation along Z direction', value, -360, 360)

    def convert_to_dict(self):
        """
        Convert instrument geometry calibration to a dictionary
        :return:
        """
        geometry_shift_dict = dict()

        geometry_shift_dict['Shift_x'] = self._center_shift_x
        geometry_shift_dict['Shift_y'] = self._center_shift_y
        geometry_shift_dict['Shift_z'] = self._center_shift_z
        geometry_shift_dict['Rot_x'] = self._rotation_x
        geometry_shift_dict['Rot_y'] = self._rotation_y
        geometry_shift_dict['Rot_z'] = self._rotation_z

        return geometry_shift_dict

    def convert_error_to_dict(self):
        """
        Convert instrument geometry calibration to a dictionary if this shift is shift error
        :return:
        """
        geometry_shift_dict = dict()

        geometry_shift_dict['error_Shift_x'] = self._center_shift_x
        geometry_shift_dict['error_Shift_y'] = self._center_shift_y
        geometry_shift_dict['error_Shift_z'] = self._center_shift_z
        geometry_shift_dict['error_Rot_x'] = self._rotation_x
        geometry_shift_dict['error_Rot_y'] = self._rotation_y
        geometry_shift_dict['error_Rot_z'] = self._rotation_z

        return geometry_shift_dict
