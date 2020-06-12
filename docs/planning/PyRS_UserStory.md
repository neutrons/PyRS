User Story Title: Strain-Stress
=============================


| **ID** | **Complexity** |  **Est.** |  **Author** |
| ------------- | ---------------|-----------|-------------|
| Strain-Stress | Medium | ? | Jeff Bunn |

As a neutron beamline user, I want to create a set of 1D, 2D or 3D data of strain, stress, or peak parameters from measured neutron data such that this set can be visualized, and output to be compared to, or used as, an input for material behavior modeling. 

Acceptance Criteria:
--------------------
I will define the HidraProjectFile to be used via a dialog box, each of these project files will contain PeakCollections object and spatial location information (vx, vy, vz, or sx, sy, sz). Additionally I will need to define which Peak_Label (peak object) I wish to utilize. I will then define a case of either 1) all three orthogonal directions (11, 22 and 33) are defined or 2) only two directions. In the case of only two directions I will define whether a ‘plane strain’ or ‘plane stress’ assumption will be utilized. For each direction, a user will select N>1 projects files. PyRS will extract the spatial information (vx,vy,vz by default) and determine the grid that is common across all selected project files. The strain-stress UI will provide an option to redefine the motor positions that are used as the x, y, and z grid. I will be presented with an error if there is a mismatch in grids across the files stating if there are missing points, and what are those points, and their directions. If a different peak_label is selected for multiple directions, I should be given a warning that a label mismatch is detected. If there are multiple points sub runs for the same vx,vy,vz for a direction I should be given an option to define the provenance of the points, whereby I keep a specific set of points, or the points will be 1) averaged, 2) point with lowest strain error, or 3) user selects the specific HidraProjectFile to use. Once the grid is complete and ‘aligned’ PyRS will then calculate the Strain/Stress for each spatial point (vx,vy,vz) and saved into a user defined named stress/strain object. This object should be able to be loaded later for visualization via a separate tab on the same page. I will be given feedback that the process what successful and that the file was saved. Additionally, the option to output a .csv file containing a table of the vx,vy,vz as well as the strain, stress and peak parameters. These parameters will exist for each PeakCollections (peaklabel).

Description, Additional Detail, Context
---------------------------------------
The Strain/Stress formulas can be found at: https://github.com/neutrons/PyRS/blob/master/docs/stress_strain.pdf

