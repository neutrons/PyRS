pyRS User Interface
###################

The python Residual Stress (pyRS) software package was developed to analyze diffraction data measured at the high-intensity diffractometer for residual stress analysis. PyRS provides methods to determine the position, width, and height of diffraction peaks in a measured dataset.

Launch the Graphical Interface
==============================

pyRS graphical interface can launched using the pyrsplot executable

.. code-block::

  PYTHONPATH=$PWD:$PYTHONPATH python scripts/pyrsplot

you should be able to see pyRS's MainWindow:

.. image:: startup.png
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

.. image:: peak_overview.svg
  :width: 800
  :alt: pyrsplot startup window

Fitting measured data using pyRS
--------------------------------

pyRS was designed to allow the software user to define the peak fitting of data stored using the hidraprojectfile definition. Users interact with the peak fitting UI by defining N ranges for individual peak fitting, where N is the number of windows of interest. Data are loaded by either selecting a specific project file or on the anlaysis cluster by loading a specific run number. A user defines a peak window interactively in the UI by clicking (and holding) on one side of the peak dragging over the specific peak of interest. The user can tweek the graphically defined window by double clicking on the x_left, x_right, or Label entry intry in Peak Ranges. Users can export these inputs as a json file for use in later sessions.

.. image:: define_range.png
  :width: 800
  :alt: pyrsplot startup window

Below are examples of peak fits that use a single an multiple fit windows.

.. image:: single_fit.png
  :width: 800
  :alt: pyrsplot startup window

.. image:: multi_fit.png
  :width: 800
  :alt: pyrsplot startup window

After defining the range click "Fit Peak(s)" to launch the anlaysis.

.. image:: fit_data.png
  :width: 800
  :alt: pyrsplot startup window

Results from the peak fitting are visualized on the right using 1D or 2D scatter plots. Users can define what paramters are visulized by changing the 1D or 3D scatter paramters

.. image:: visualize_res.png
  :width: 800
  :alt: pyrsplot startup window

Stress Strain Analysis
======================

Select the Stress/Strain Calculation option to launch the stress analysis UI. Reminder overivew of the UI interface:

.. image:: ../basics/stress_overview.svg
  :width: 800
  :alt: Stress Analysis overivew
