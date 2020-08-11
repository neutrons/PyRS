Installation
############

Using a Conda Environment
=========================

PyRS preferred method is to create a conda environment with the required Python dependencies.
Follow these steps:

1. Install basic dependencies: `Conda <https://docs.anaconda.com/anaconda/install/>`_, Python 3, and PyQt 
2. Create a new Conda environment with additional dependencies:

.. code-block::

   $ conda create -n pyrs -c mantid -c mantid/label/nightly mantid-workbench -c conda-forge  --file requirements.txt --file requirements_dev.txt

3. Activate the conda environment

.. code-block::

   $ conda activate pyrs


.. caution::
   
   Do not update this newly created environment as some dependencies might not be backwards compatible.


Quick Start
###########

Launch the Graphical Interface
==============================

PyRS graphical interface can launched using the pyrsplot executable

.. code-block::

   $ pyrsplot

you should be able to see PyRS's MainWindow:

.. image:: startup.png
  :width: 400
  :alt: pyrsplot startup window

As listed in the MainWindow, pyrsplot can run 3 different usage modes as described in the next sections.

Data Manual Reduction
=====================

#TODO

Peak Fitting
============

#TODO

Strain/Stress Fitting
=====================

#TODO

