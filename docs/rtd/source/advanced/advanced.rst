Definitions
###########

HIDRA Project File
-----------------

Definition
==========

The HIDRA project file contains reduced data, which may contain multiple runs.
There are several levels reduced data for an engineering diffraction experiment project, such as
diffraction patterns, fitted peaks and calculated strains/stresses.
This project file shall store information about measured diffraction patterns (intensity vs :math:`2\theta`) and results from single peak fitting.
Strain/stress results are not stored in individual project files because these calculations rely on information from addition project files (d0 and additional strain components).

Here is a list of reduced data that will be stored in project file

- Raw experimental data presented as neutron intensity on each pixel (not by default)
- Histogram data vectors
  - :math:`2\theta`
  - intensity
  - estimated intensity uncertainty
- Instrument geometry parameters
- Calibrated instrument geometry parameters
- Fitted peaks' parameters

File Structure
==============

HIDRA project file utilizes an HDF file format with the following organizational tree structure.

- Instrument
  - Geometry parameters: table of parameters
  - Geometry calibration:
      - Calibration file
      - Calibration run
      - Calibration date
      - Table of calibrated parameters
- Mask
  - detector
      - default detector masked used to exclude pixels near the edge of the detector

- Peaks
  - HKL_0
    - chi2
    - d reference
    - d reference error
    - fitting error
    - parameters
    - sub-runs

  - HKL_1
      - ... ...

- Raw Data
  - logs
    - logs for motor positions and metadata information for each sub-run
  - sub-runs
    - intensity count vs pixel vectors (by default not saved)

- reduced diffraction data
  - 2theta
    - :math:`2\theta` vectors for each sub-run
  - main
    - count vectors for each sub-run
  - main_var
    - estimated error in counts for each sub-run
  - main_XANG
    - count vectors for each sub-run for a given out-of-plane angle (X)
  - main_XANG_var
      - estimated error in counts for each sub-run for a given out-of-plane angle (X)



Python Scripts
##############

#TODO
