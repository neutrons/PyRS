############
Introduction
############


#TODO High-level explanation of the product.

Installation
############

Using a Conda Environment
=========================

pyRS preferred method is to create a conda environment with the required Python dependencies.
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
