HB2B Project File
-----------------

Definition
==========

HB2B's project file shall contain reduced data in a project scope, which may contain multiple runs.
There are several levels reduced data for an engineering diffraction experiment project, such as
diffraction pattern, fitted peaks and calculated strain and stress.
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
  - ... ...

- Peaks
  - HKL_0
    - RunNumber[0]
      - Parameter table: peak profile type, fitted parameter value and error, cost, and etc
    - ...
  - HKL_1
    ... ...
  - ... ...

- Strain