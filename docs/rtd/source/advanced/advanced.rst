Definitions
###########

HB2B Project File
-----------------

Definition
==========

HB2B's project file shall contain reduced data in a project scope, which may contain multiple runs.
There are several levels reduced data for an engineering diffraction experiment project, such as diffraction pattern, fitted peaks and calculated strain and stress.
This project file shall record all these regular reduced data.

Here is a list of reduced data that will be stored in project file
 - Raw experimental data presented as neutron counts on each pixel
 - Histogram data as :math:`2\theta` vs counts
 - Histogram data as :math:`d-Spacing` vs counts
 - Instrument geometry parameters
 - Calibrated instrument geometry parameters
 - Fitted peaks' parameters
 - Calculated strain and stress

To be noticed that neutron event data will not be saved to this file.


File Structure
==============

Project file will be saved to HDF file format.
Therefore all the reduced data will be organized in tree structure.

Here is the proposed structure of the project file

- Experiment information
  - IPTS number: single value
  - Run numbers: vector of integers

- Instrument  
   - Geometry parameters: table of parameters
   - Geometry calibration:
      - Calibration file
      - Calibration run
      - Calibration date
      - Table of calibrated parameters

- Raw Data (i.e., counts per pixel)
   - RunNumber[0]: 1D integer array in order of pixel number
   - RunNumber[1]:
   - ... ...

- Histogram
   - RunNumber[0]
      - :math:`2\theta` vs counts
      - :math:`d-Spacing` vs counts
   - RunNumber[1] 
      - :math:`2\theta` vs counts
      - :math:`d-Spacing` vs counts

- Peaks
   - HKL_0
      - RunNumber[0]
         - Parameter table: peak profile type, fitted parameter value and error, cost, etc.
   - HKL_1
   - ...
  

- Strain



Python Scripts
##############

#TODO




Math Models
###########

PyRS models Strain and Stress tensor :math:`_{ij}` fields using the following formulation:

*Strain*, aka  unconstrained strain, :math:`\epsilon_{ij}` is measured as the fraction change from a reference state :math:`d_0`.

.. math::
   
   \epsilon_{ij} = \frac{d_{ij} - d_0}{d_0}
   

*Residual stress* is determined by measuring stress along 3 orthogonal directions

.. math::

   \sigma_{ij} = \frac{E}{(1 + \nu)}\left[\epsilon_{ij} + \frac{\nu}{1-2\nu}(\epsilon_{11} + \epsilon_{22} + \epsilon_{33})\right]

where:

:math:`\nu`
   Poisson's ratio

:math:`E`
   Young's modulus


Notice that :math:`\epsilon_{ij}` with :math:`i = j` are principal strains.
But not all all three orthogonal strains are equivalent to principal strains.
The off-diagonal strain component, i.e., :math:`\epsilon_{ij}` with :math:`i \neq j` are all set to *zero*.  
It is very hard to measure these values in HB2B's setup.

As a result, the stress in PyRS is calculated as follows:

.. math::

   \sigma_{ii} = \frac{E}{(1 + \nu)}\left[\epsilon_{ii} + \frac{\nu}{1-2\nu}(\epsilon_{11} + \epsilon_{22} + \epsilon_{33})\right]

where the second term in the sum is common to all 3 principle directions, :math:`\sigma_{11}`, :math:`\sigma_{22}`, and :math:`\sigma_{33}`.

**in-plane strain**: this is a special case in which the stress in one of the principal directions is zero, :math:`\sigma_{33}=0`. Therefore, the corresponding strain can be determined from the other principal components using Poisson's ratio:

.. math::

   \epsilon_{33} = \frac{\nu}{\nu-1}(\epsilon_{11} + \epsilon_{22}) 


Resulting in a simplified set of equations for the *in-plane stress* case:

.. math::

   \sigma_{11} &=& \frac{E}{(1 + \nu)}\left[\epsilon_{11} + \frac{\nu (\epsilon_{11} + \epsilon_{22})}{1-\nu}\right] \\[1cm]
   \sigma_{22} &=& \frac{E}{(1 + \nu)}\left[\epsilon_{22} + \frac{\nu (\epsilon_{11} + \epsilon_{22})}{1-\nu}\right] \\[1cm]
   \sigma_{33} &=& 0
  

