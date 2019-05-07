#!/bin/sh
python setup.py pyuic
python setup.py build

echo
MANTIDLOCALPATH=/home/wzz/Mantid_Project/debug/bin/
MANTIDMACPATH=/Users/wzz/MantidBuild/debug/bin/
MANTIDSNSPATH=/opt/mantidnightly/bin/
MANTIDPATH=$MANTIDMACPATH:$MANTIDLOCALPATH:$MANTIDSNSPATH
PYTHONPATH=$MANTIDPATH:$PYTHONPATH


# PYTHONPATH=build/lib:$PYTHONPATH $CMD build/scripts-2.7/pyrscalibration.py

echo "User option: $1"

if [ "$1" = "111" ] || [ "$1" = "prototype" ] ; then
    echo "Protyping calibration"
    PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH python ./prototypes/calibration/Quick_Calibration_v2.py
fi

if [ "$1" = "1" ] ; then
    echo "NOT DEFINED YET"
    PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH
fi
