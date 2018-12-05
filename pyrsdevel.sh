#!/bin/sh
python setup.py pyuic
python setup.py build
if [ $1 ]; then
    CMD=$1
else
    CMD=''
fi
MANTIDLOCALPATH=/home/wzz/Mantid_Project/debug/bin/
MANTIDMACPATH=/Users/wzz/MantidBuild/debug-stable/bin/
MANTIDSNSDEBUGPATH=/SNS/users/wzz/Mantid_Project/builds/debug/bin/
MANTIDPATH=$MANTIDMACPATH:$MANTIDLOCALPATH:$MANTIDSNSDEBUGPATH
PYTHONPATH=$MANTIDPATH:$PYTHONPATH
echo $PYTHONPATH

PYTHONPATH=build/lib:$PYTHONPATH $CMD build/scripts-2.7/pyrsplot
