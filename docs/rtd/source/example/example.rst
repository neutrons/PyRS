pyRS Example Use
################

Launch the pyRS graphical interface to load the peak fitting UI.

.. code-block::

  PYTHONPATH=$PWD:$PYTHONPATH python scripts/pyrsplot

.. image:: ../figures/startup.png
  :width: 400
  :alt: pyrsplot startup window

Peak Fitting
============

Select the peak fitting option to launch the peak fitting UI. Reminder overivew of the UI interface:

.. image:: ../figures/peak_overview.svg
  :width: 800
  :alt: Peak Fitting overivew

Load measured data
------------------
After the UI loads, select the "Browse Exp. Data File" to select one of the three hidraprojectfiles in the examples folder "HB2B_2246.h5, HB2B_2247.h5, and HB2B_2251.h5". These three files represents data for the LD, TD, and ST directions.

Fitting measured data using pyRS
--------------------------------

Graphically define a fit window over the peak of interest. Double click in the x_left (87.0), x_right (92.5), and Label (Al_311). Save the fit range for later use. See below for an example of what the UI should look like before Fitting. After saving the range, click the "Fit Peak(s)" button to start the peak fitting process. After fitting, save the results using either "File/Save" or "File/Save As" to save a new hidraprojectfile. Note that the hidraprojectfiles are not write protected and you can overwrite prior peak fitting results.

.. image:: ../figures/Example_Fit.png
  :width: 800
  :alt: pyrsplot startup window

Below are examples of peak fits that use a single an multiple fit windows.

.. image:: ../figures/Fit_2246.png
  :width: 800
  :alt: Fit of run 2646

.. image:: ../figures/Fit_2247.png
  :width: 800
  :alt: Fit of run 2647

.. image:: ../figures/Fit_2251.png
  :width: 800
  :alt: Fit of run 2651

Stress Analysis
===============

Select the Stress/Strain Calculation option to launch the stress analysis UI. Reminder overivew of the UI interface:

.. image:: ../figures/stress_overview.svg
  :width: 800
  :alt: Stress Analysis overivew

Define the stress condition that pyRS will use to calculate the stresses.

  .. image:: ../figures/Stress_Load.png
    :width: 800
    :alt: pyrsplot startup window

  Below are examples of peak fits that use a single an multiple fit windows.

  .. image:: ../figures/Stress_Define_Material.png
    :width: 800
    :alt: pyrsplot startup window

  .. image:: ../figures/Stress_Define_d0.png
    :width: 800
    :alt: pyrsplot startup window

  After defining the range click "Fit Peak(s)" to launch the anlaysis.

  .. image:: ../figures/Stress_Final.png
    :width: 800
    :alt: pyrsplot startup window
