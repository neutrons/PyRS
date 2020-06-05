# Reduction engine including slicing
import numpy as np
from uncertainties import unumpy
import pyrs.utilities.checkdatatypes
# TODO:
#   1) check that vx, vy, vz positions of peakcollections align
#   2) Check for duplicate entries (can happen if two runs are combined)



class StrainDiagonalsCollection:
    def __init__(self, peak_collections, in_plane=None, poisson_ratio=None):
        r"""
        A sequence of strain tensor diagonals, the order is as that of the lattice spacings provided by the peak
        collections. It is assumed the peak collections have the same ordering.

        Parameters
        ----------
        peak_collections: list
            A list of ~pyrs.peaks.peak_collection.PeakCollection. Usually three but maybe
            two if measurements are done in-plane.
        in_plane: str
            If measurements in-plane, then the third component is zero
        poisson_ratio: float

        """
        strain_uniaxial_collections = [peak_coll.get_strain(units='strain') for peak_coll in peak_collections]
        subruns_count = len(strain_uniaxial_collections[0][0])

        # Verify the collections have the same number of subruns
        if len(set([len(collection) for collection in strain_uniaxial_collections])) > 1:
            raise RuntimeError('strain collections have different lengths for different orientations')

        # Verify we have three collections or two collections plus one in-plane condition
        in_plane_categories = ('strain', 'stress')
        if len(strain_uniaxial_collections) == 2 and in_plane not in in_plane_categories:
            raise RuntimeError('A valid in-plane condition is required with two collections')

        if in_plane == 'strain':
            # Include the third, zero component
            strain_uniaxial_collections.append((np.zeros(subruns_count, dtype=float),  # values
                                                np.zeros(subruns_count, dtype=float))  # errors
                                               )

        # shape = (peak_collections_count, subruns_count)
        strain_uniaxial_values  = np.array( collection[0] for collection in strain_uniaxial_collections)
        strain_uniaxial_errors = np.array( collection[1] for collection in strain_uniaxial_collections)
        strain_uniaxials = unumpy.uarray(strain_uniaxial_values, strain_uniaxial_errors)

        if in_plane == 'stress':
            # Calculate the third stress tensor diagonal with the in-plane stress condition
            if poisson_ratio is None:
                raise RuntimeError('Cannot calculate the last strain component without the Poisson Ratio')
            factor = poisson_ratio / (poisson_ratio - 1)
            strain_third_component = factor * np.sum(strain_uniaxials, axis=0)  # add the two collections
            strain_uniaxials = unumpy.append(strain_uniaxials, strain_third_component[np.newaxis, :], axis=0)

        self._strains_diagonal = unumpy.transpose(strain_uniaxials)  # shape = (subruns_count, 3)

    @property
    def strains(self):
        r"""A sequence of strain tensor diagonals"""
        return unumpy.copy(self._strains_diagonal)

    @@property
    def traces(self):
        r"""A sequence of traces of the strain tensors"""
        return unumpy.sum(self._strains_diagonal, axis=1)


class StressDiagonalsCollection:
    r"""
    A list of stress tensor diagonals

    Parameters
    ----------
    macro_strain: ~pyrs.core_strain_stress_calculator.MacroStrainCollection
    poisson_ratio: float, ~uncertainties.ufloat
    young_modulus: float, ~uncertainties.ufloat
    """

    def __init__(self, strains_diagonals, young_modulus, poisson_ratio):
        shear_modulus = young_modulus / (1 + poisson_ratio)
        trace_factor = poisson_ratio / (1 - 2 * poisson_ratio)
        self._stress_diagonals = shear_modulus * (strains_diagonals + trace_factor * strains_diagonals.traces)

    @property
    def stresses(self):
        r"""List of stress tensor diagonals"""
        return unumpy.copy(self._stress_diagonal)


class StrainStress:
    """
    Object to contain stress parameters (names and values) of a collection of peaks for sub runs
    """

    def __init__(self,  peak_1, peak_2, peak_3, young_modulus, poisson_ratio, is_plane_strain, is_plane_stress):
        """Initialization
        Parameters
        ----------
        peak_1 : PeakCollection
            Peakcollection for strain in the 1 direction
        peak_2 : PeakCollection
            Peakcollection for strain in the 2 direction
        peak_3 : PeakCollection
            Peakcollection for strain in the 3 direction
        young_modulus : float
            Youngs modulus of the material of interest
        poisson_ratio : float
            Poisson ratio of the material of interest
        is_plane_strain : Bool
            Use plane strain simplification
        is_plane_strain : Bool
            Use plane stress simplification
        """

        # check input
        pyrs.utilities.checkdatatypes.check_float_variable('Young modulus E', young_modulus, (None, None))
        pyrs.utilities.checkdatatypes.check_float_variable('Poisson ratio Nu', poisson_ratio, (None, None))

        # store peak fit results
        self._peak_1 = peak_1
        self._peak_2 = peak_2
        self._peak_3 = peak_3

        self._E = young_modulus
        self._nu = poisson_ratio

        # Initalize strain and stress matrices
        self._setup_strain_stress_matrix(is_plane_strain, is_plane_stress)

        # Calculate stress
        self._calculate_stress_matrix()

        return

    def _setup_strain_stress_matrix(self, is_plane_strain, is_plane_stress):
        """ Initalize epsilon vector for stress calculation

        Parameters
        ----------
        is_plane_strain : Bool
            Use plane strain simplification
        is_plane_strain : Bool
            Use plane stress simplification
        Returns
        -------
        """

        eps_1, delta_eps_1 = self._peak_1.get_microstrain()
        eps_1 *= 1.e-6  # convert to strain
        delta_eps_1 *= 1.e-6  # convert to strain
        eps_2, delta_eps_2 = self._peak_2.get_microstrain()
        eps_2 *= 1.e-6  # convert to strain
        delta_eps_2 *= 1.e-6  # convert to strain

        if is_plane_strain:
            eps_3 = np.zeros(shape=eps_1.shape[0], dtype=np.float)
            delta_eps_3 = np.zeros(shape=eps_1.shape[0], dtype=np.float)
        elif is_plane_stress:
            eps_3 = self._nu / (self._nu - 1.) * (eps_1 + eps_2)
        else:
            eps_3, delta_eps_3 = self._peak_3.get_microstrain()
            eps_3 *= 1.e-6  # convert to strain
            delta_eps_3 *= 1.e-6  # convert to strain

        self._epsilon = np.zeros(shape=(3, eps_1.shape[0]), dtype=np.float)
        self._epsilon_error = np.zeros(shape=(3, eps_1.shape[0]), dtype=np.float)

        self._sigma = np.zeros(shape=(3, eps_1.shape[0]), dtype=np.float)
        self._sigma_error = np.zeros(shape=(3, eps_1.shape[0]), dtype=np.float)

        self._epsilon[0] = eps_1
        self._epsilon[1] = eps_2
        self._epsilon[2] = eps_3
        self._epsilon_error[0] = delta_eps_1
        self._epsilon_error[1] = delta_eps_2
        self._epsilon_error[2] = delta_eps_3

    def _calculate_stress_component_error(self, ax_1=0, ax_2=1, ax_3=2):
        """
        Calculate uncertainty in stress using propagation of error
        Parameters
        ----------
        ax_1 : int
            strain direction to use for axis 1 in error propogation
        ax_2 : int
            strain direction to use for axis 2 in error propagation
        ax_3 : int
            strain direction to use for axis 3 in error propagation
        Returns
        -------
        """

        # calculate partial dirivates of stress for each strain component
        d_sig_ij_d_ax_1 = (1 + self._nu / (1. - 2. * self._nu) * (self._epsilon[ax_2] + self._epsilon[ax_3]))
        d_sig_ij_d_ax_2 = self._nu / (1. - 2. * self._nu) * (self._epsilon[ax_1] + self._epsilon[ax_3])
        d_sig_ij_d_ax_3 = self._nu / (1. - 2. * self._nu) * (self._epsilon[ax_1] + self._epsilon[ax_2])

        self._sigma_error[ax_1] = self._E / (1. + self._nu) *\
            np.sqrt(np.square(d_sig_ij_d_ax_1) * np.square(self._epsilon_error[ax_1]) +
                    np.square(d_sig_ij_d_ax_2) * np.square(self._epsilon_error[ax_2]) +
                    np.square(d_sig_ij_d_ax_3) * np.square(self._epsilon_error[ax_3]))

        return

    def _calculate_stress_component(self, direction=0):
        """
        Calculate stress component for specific direction
        Parameters
        ----------
        direction : int
            stress direction to calculate
        Returns
        -------
        """

        # calculate stress
        self._sigma[direction] = self._E / (1 + self._nu)(self._epsilon[direction] + self._nu / (1 - 2 * self._nu) *
                                                          (self._epsilon.sum(axis=0)))

        return

    def _calculate_stress_matrix(self):
        """
        Calculate stress and uncertainty
        Parameters
        ----------
        Returns
        -------
        """
        self._calculate_stress_component(direction=0)
        self._calculate_stress_component(direction=1)
        self._calculate_stress_component(direction=2)
        self._calculate_stress_component_error(ax_1=0, ax_2=1, ax_3=2)
        self._calculate_stress_component_error(ax_1=1, ax_2=0, ax_3=2)
        self._calculate_stress_component_error(ax_1=2, ax_2=0, ax_3=1)

        return

    def get_strain(self):
        """get strain values and uncertainties
        Parameters
        ----------
        values :
            1D numpy array or floats
        Returns
        -------
          tuple
              A two-item tuple containing the strain and its uncertainty.
        """

        return self._epsilon

    def get_stress(self):
        """get stress values and uncertainties
        Parameters
        ----------
        values :
            1D numpy array or floats
        Returns
        -------
          tuple
              A two-item tuple containing the strain and its uncertainty.
        """

        return self._sigma, self._sigma_error
