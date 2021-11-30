.. image:: https://github.com/neutrons/PyRS/actions/workflows/ci.yml/badge.svg?branch=next
  :target: https://github.com/neutrons/PyRS/actions?query=branch:next

.. image:: https://codecov.io/gh/neutrons/PyRS/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/neutrons/PyRS

-----------------------
Developer Quick Start
-----------------------

If you've never used PyRS before, you can get started quickly by doing the following.

1. Install basic dependencies: Conda, Python, and PyQt
2. Create a new Conda environment with additional dependencies:

.. code-block::

   $ conda create -n pyrs -c mantid -c mantid/label/nightly mantid-workbench -c conda-forge  --file requirements.txt --file requirements_dev.txt

3. Activate the conda environment

.. code-block::

   $ conda activate pyrs

4. From the PyRS directory, run the setup script in developer mode

.. code-block::

   $ python setup.py develop

5. From the PyRS directory, start the user interface (``develop`` mode puts it in your ``PATH``)

.. code-block::

   $ pyrsplot

For those setting up in a virtual environment, the dependencies are
listed in ``requirements.txt`` and ``requirements_dev.txt``.  They can
be supplide to ``conda create`` using the `--file`` argument, or to
``pip install`` using the ``-R`` argument.

-----------------------
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

---------------------------
Running and developing PyRS
---------------------------

To start main window from analysis machine

.. code-block::

   $ PYTHONPATH=$PWD:$PYTHONPATH QT_API=pyqt python scripts/pyrsplot

To develop

.. code-block::

   $ pyrsdev.sh

To run all of the tests


.. code-block::

   $ python -m pytest

Running specific tests can be done `through standard ways`
<https://docs.pytest.org/en/stable/usage.html>`_. For example

.. code-block::

   $ python -m pytest tests/unit

will only run the unit tests
