pyRS User Interface
###################

The python Residual Stress (pyRS) software package was developed to analyze diffraction data measured at the high-intensity diffractometer for residual stress analysis. PyRS provides methods to determine the position, width, and height of diffraction peaks in a measured dataset.

Launch the Graphical Interface
==============================

pyRS graphical interface can launched using the pyrsplot executable

.. code-block::

  PYTHONPATH=$PWD:$PYTHONPATH python scripts/pyrsplot

you should be able to see pyRS's MainWindow:

.. image:: ../figures/startup.png
  :width: 400
  :alt: pyrsplot startup window

As listed in the MainWindow, pyrsplot can run 3 different usage modes as described in the next sections.

Data Manual Reduction
=====================

Manual data reduction requires access to raw neutron events stored on ORNL data mounts. The best mechanism to access these data is through analysis.sns.gov. Consult your local contact or the HB2B instrument team for further information about re-reducing data.

Peak Fitting
============

Overview of the pyRS peak fitting UI
------------------------------------

.. image:: ../figures/peak_overview.svg
  :width: 800
  :alt: pyrsplot startup window

Fitting measured data using pyRS
--------------------------------

pyRS was designed to allow the software user to define the peak fitting of data stored using the hidraprojectfile definition. Users interact with the peak fitting UI by defining N ranges for individual peak fitting, where N is the number of windows of interest. Data are loaded by either selecting a specific project file or on the analysis cluster by loading a specific run number. A user defines a peak window interactively in the UI by clicking (and holding) on one side of the peak dragging over the specific peak of interest. The user can tweak the graphically defined window by double clicking on the x_left, x_right, or Label entry in Peak Ranges. Users can export these inputs as a json file for use in later sessions.

.. image:: ../figures/define_range.png
  :width: 800
  :alt: define range

Below are examples of peak fits that use a single and multiple fit windows.

.. image:: ../figures/single_fit.png
  :width: 800
  :alt: single fit example

.. image:: ../figures/multi_fit.png
  :width: 800
  :alt: multifit example

After defining the range click "Fit Peak(s)" to launch the analysis.

.. image:: ../figures/fit_data.png
  :width: 800
  :alt: peak fitting

Results from the peak fitting are visualized on the right using 1D or 2D scatter plots. Users can define what parameters are visualized by changing the 1D or 3D scatter parameters

.. image:: ../figures/visualize_res.png
  :width: 800
  :alt: vis peak fit

Stress Strain Analysis
======================

Select the Stress/Strain Calculation option to launch the stress analysis UI. Reminder overview of the UI interface:

.. image:: ../figures/stress_overview.svg
  :width: 800
  :alt: Stress Analysis overivew

Define the stress condition that pyRS will use to calculate the stresses. A 2D condition allows the selection of either a plane strain or plane stress condition. A 2D condition only prompts loading 2 hidraprojectfiles with peak fit results. Otherwise, a 3D stress condition requires 3 hidraprojectfiles.

  .. image:: ../figures/Stress_Load.png
    :width: 800
    :alt: load project files

After loading 2, or 3, hidraprojectfiles the users need to enter the Mechanical constants (or Materials parameters) Youngs Modulus and Poisson's ratio.

  .. image:: ../figures/Stress_Define_Material.png
    :width: 800
    :alt: define materials

By default the stress/strain framework will use the d0 defined in the peak fitting to determine strains. A constant or spatially varying d0 can be entered at any time after selecting the hidraprojectfiles and entering the materials parameters. Changing the d0 will force the framework to recalculate all strains and stresses.

  .. image:: ../figures/Stress_Define_d0.png
    :width: 800
    :alt: define d0

After defining the Materials properties and d0, the user can now visualize the determine stresses by changing what is plotted (Define Visualization in overview). The user can switch between the 11, 22, and 33 components of the stress/strain.

  .. image:: ../figures/Stress_Final.png
    :width: 800
    :alt: visualize stress strain

Detector Calibration
====================

The detector calibration refines the HB2B detector's position and orientation (shifts x, y, z and rotations
x, y, z), the virtual arm offset 2theta_0 and the wavelength from a powder calibration run (Ni, Fe and Mo
powders). Reduction applies the resulting calibration file (JSON) to every run measured afterwards.

Open the calibration window from the MainWindow menu **Advanced > Calibration**. Two methods are offered:

* **Ring-based** (the default): fits every Debye ring once, in strips of each sub-run's detector image, and
  refines the geometry from where the rings lie on the detector. It then finds the rings again where the
  refined geometry predicts them and refines again, until a round moves the result by less than half its
  statistical precision (at most **Rounds** rounds, 10 by default). On the calibration runs tested it is
  more accurate than FitCalibration on both methods' own measures, and takes about half a minute.
* **FitCalibration**: re-reduces the run and re-fits its peaks in eta slices for every trial geometry. It is
  kept for comparison and for older recipes; started from the nominal geometry it can stop short of the
  best geometry.

A ring-based calibration in the window:

1. Load the calibration run.
2. Leave **Method** at *Ring-based*; **Strips** (5) sets how many strips each sub-run's image is cut into, and
   **Rounds** (10) the most ring-finding rounds.
3. Click **Fit Peaks** to find the rings, then **Calibrate**: the recipe starts with *iterate* (refine, then
   find the rings again until converged).
4. Check the predicted rings on the detector view, then export the calibration.

Recipes saved from the window keep the method, strips and rounds. A recipe saved before the method choice
existed opens with FitCalibration, its method then.

The same calibrations run from the command line:

.. code-block:: sh

  pixi run pyrs-pixel-calibration HB2B_3510.nxs.h5 --output HB2B_CAL.json     # ring-based (default)
  pixi run pyrs-calibration recipe.json                                       # FitCalibration

Every calibration file records how it was made (its *provenance*): the method, the run and mask files, each
refinement step and, for the ring-based method, each round of ring finding. The evaluation behind the
default is recorded in ``docs/ground_truths.md`` (GitHub issue #1038).
