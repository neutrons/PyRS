.. image:: https://travis-ci.org/neutrons/PyRS.svg?branch=master
    :target: https://travis-ci.org/neutrons/PyRS

-----------------------
Data reduction workflow
-----------------------

This is a test 

The workflow is as follows.
The word "script" is used to denote a distict step in processing the data rather than actual script.

1. Start with a nexus file. As in, /HFIR/HB2B/IPTS-22731/nexus/HB2B_439.nxs.h5
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

To run the tests


.. code-block::

   $ pyrstest.sh
