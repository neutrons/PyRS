#!/bin/sh
python setup.py pyuic
python setup.py build
if [ $1 ]; then
    CMD=$1
else
    CMD=''
fi
PYTHONPATH=$PYTHONPATH:/Users/wzz/MantidBuild/debug-stable/bin
echo $PYTHONPATH
PYTHONPATH=build/lib:$PYTHONPATH $CMD build/scripts-2.7/pyrsplot
