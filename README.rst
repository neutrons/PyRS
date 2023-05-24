pyRS
=========

.. image:: https://github.com/neutrons/PyRS/actions/workflows/ci.yml/badge.svg?branch=next
  :target: https://github.com/neutrons/PyRS/actions?query=branch:next

.. image:: https://codecov.io/gh/neutrons/PyRS/branch/next/graph/badge.svg
  :target: https://codecov.io/gh/neutrons/PyRS

Data reduction workflow
-----------------------

  The workflow is as follows.
  The word "script" is used to denote a distict step in processing the data rather than actual script.

  1. Start with a nexus file. As in, ``/HFIR/HB2B/IPTS-22731/nexus/HB2B_439.nxs.h5``
  2. This is read in by something that will split into sub-scans and create a project file [script 1 - currently convert_nexus_to_hidra.py]
  3. This is read in by something that will create powder patterns and add them to the project file [script 2 - currently reduce_HB2B.py]
  4. This is read in by the gui to do the peak fitting and add the results to the project file [script 3 - currently somewhere in the gui]
  5. This is read in by the gui and used to create the summary .csv file [script 4 - currently being written]

  Some other things to note
  * scripts 1-2 will be refactored/combined to create a new reduce_HB2B.py that can be run by autoreduction
  * scripts 2-4 (more like functionality 2-4) will all live in the gui in some way
  * going back any number of steps will delete the following steps from the project file. This is to prevent users from having a project file with powder patterns that are not associated with the peak fitting from the project file.


Developer Quick Start
-----------------------

If you've never used PyRS before, you can get started quickly by doing the following.

1. Install basic dependencies: Conda, Python, and PyQt. Conda installation requires linux.
2. Create a new Conda environment with additional dependencies:

Installation using an anaconda environment
------------------------------------------
Anaconda environments are only supported on OSx (x86) and Linux using python 3.8

1. Configure anaconda environment:

.. code-block::

  conda install -c conda-forge mamba

.. code-block::

  mamba env create --name pyrs --file environment.yml

2. Activate the conda environment

.. code-block::

  conda activate pyrs

3. From the PyRS directory, run the setup script in developer mode

.. code-block::

  python setup.py build

4. From the PyRS directory, start the user interface

.. code-block::

  PYTHONPATH=$PWD:$PYTHONPATH python scripts/pyrsplot

Running and developing PyRS
---------------------------

To start main window from analysis machine

.. code-block::

  PYTHONPATH=$PWD:$PYTHONPATH python scripts/pyrsplot

To develop

To run all of the tests

.. code-block::

  python -m pytest

Running specific tests can be done `through standard ways`
<https://docs.pytest.org/en/stable/usage.html>`_. For example

.. code-block::

   python -m pytest tests/unit

will only run the unit tests


Related packages
----------------
* `Mantid <https://github.com/mantidproject/mantid>`_ - The Mantid project provides a framework that supports high-performance computing and visualisation of scientific data.
* `Steca2 <https://gitlab-public.fz-juelich.de/mlz/steca/-/tree/main>`_ - The stress and texture calculator used by the Heinz Maier-Leibnitz Zentrum Garching facility.


Contributing to pyRS
--------------------
If you want to suggest changes of a feature or the inclusion of new features, you can either 1) [fork](https://github.com/neutrons/PyRS/fork) the repository to work on it and create an [issue](https://github.com/neutrons/PyRS/issues/new) to discuss it before proceeding with a pull request, or 2) create an [issue](https://github.com/neutrons/PyRS/issues/new) with your suggestion for others to discuss and potentially work on it.

Reporting bugs or asking for help
---------------------------------

Please report any bugs or ask for help by creating a new [issue](https://github.com/neutrons/PyRS/issues/new).

Funding
-------
This material is based upon work supported by the U.S. Department of Energy, Office of Science, Office of Basic Energy Sciences.
