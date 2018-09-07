#!/bin/sh
python setup.py pyuic
python setup.py build
if [ $1 ]; then
    CMD=$1
else
    CMD=''
fi
MANTIDLOCALPATH=/home/wzz/Mantid_Project/builds/debug-master/bin/
MANTIDMACPATH=/Users/wzz/MantidBuild/debug-stable/bin/
# MANTIDSNSDEBUGPATH=/SNS/users/wzz/Mantid_Project/builds/debug/bin/
MANTIDSNSDEBUGPATH=/opt/mantidnightly/bin/
MANTIDPATH=$MANTIDMACPATH:$MANTIDLOCALPATH:$MANTIDSNSDEBUGPATH
PYTHONPATH=$MANTIDPATH:$PYTHONPATH
echo $PYTHONPATH

# PYTHONPATH=build/lib:$PYTHONPATH $CMD build/scripts-2.7/pyrs_core_test.py
# PYTHONPATH=build/lib:$PYTHONPATH $CMD build/scripts-2.7/utilities_test.py
# PYTHONPATH=build/lib:$PYTHONPATH $CMD build/scripts-2.7/peakfitgui_test.py
# PYTHONPATH=build/lib:$PYTHONPATH $CMD build/scripts-2.7/polefigurecal_test.py
PYTHONPATH=build/lib:$PYTHONPATH $CMD build/scripts-2.7/straincalculationtest.py


