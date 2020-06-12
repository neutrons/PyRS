# Reduction engine including slicing
import numpy as np
import pyrs.utilities.checkdatatypes
from typing import Tuple

from mantid.simpleapi import mtd, CreateMDWorkspace, BinMD
from mantid.api import IMDHistoWorkspace
from pyrs.dataobjects.sample_logs import DirectionExtents

# TODO:
#   1) check that vx, vy, vz positions of peakcollections align
#   2) Check for duplicate entries (can happen if two runs are combined)


def _to_md(name: str,
           extents: Tuple[DirectionExtents, DirectionExtents, DirectionExtents],
           signal, errors, units: str) -> IMDHistoWorkspace:
    for extent in extents:
        assert extent[0] < extent[1], f'min value of {extent} is not smaller than max value'
    extents_str = ','.join([extent.to_createmd for extent in extents])
    # create an empty event workspace of the correct dimensions
    axis_labels = ('x', 'y', 'z')
    CreateMDWorkspace(OutputWorkspace=name, Dimensions=3, Extents=extents_str,
                      Names=','.join(axis_labels), Units=units)
    # set the bins for the workspace correctly
    aligned_dimensions = [f'{label},{extent.to_binmd}' for label, extent in zip(axis_labels, extents)]  # type: ignore
    aligned_kwargs = {f'AlignedDim{i}': aligned_dimensions[i] for i in range(len(aligned_dimensions))}
    BinMD(InputWorkspace=name, OutputWorkspace=name, **aligned_kwargs)

    # get a handle to the workspace
    wksp = mtd[name]
    # set the signal and errors
    dims = [extent.number_of_bins for extent in extents]
    wksp.setSignalArray(signal.reshape(dims))
    wksp.setErrorSquaredArray(np.square(errors.reshape(dims)))

    return wksp


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

        eps_1, delta_eps_1 = self._peak_1.get_strain()
        eps_1 *= 1.e-6  # convert to strain
        delta_eps_1 *= 1.e-6  # convert to strain
        eps_2, delta_eps_2 = self._peak_2.get_strain()
        eps_2 *= 1.e-6  # convert to strain
        delta_eps_2 *= 1.e-6  # convert to strain

        if is_plane_strain:
            eps_3 = np.zeros(shape=eps_1.shape[0], dtype=np.float)
            delta_eps_3 = np.zeros(shape=eps_1.shape[0], dtype=np.float)
        elif is_plane_stress:
            eps_3 = self._nu / (self._nu - 1.) * (eps_1 + eps_2)
        else:
            eps_3, delta_eps_3 = self._peak_3.get_strain()
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
