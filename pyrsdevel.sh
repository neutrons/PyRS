#!/bin/sh
python setup.py pyuic
python setup.py build

CMDS=''
for file in "$@"
do
  CMDS="$CMDS $file"
done

MANTIDLOCALPATH=/home/wzz/Mantid_Project/debug/bin/
MANTIDMACPATH=/Users/wzz/MantidBuild/debug-stable/bin/
MANTIDSNSDEBUGPATH=/opt/Mantid/bin/
MANTIDPATH=$MANTIDMACPATH:$MANTIDLOCALPATH:$MANTIDSNSDEBUGPATH
PYTHONPATH=$MANTIDPATH:$PYTHONPATH
echo $PYTHONPATH

PYTHONPATH=build/lib:$PYTHONPATH build/scripts-2.7/pyrsplot $CMD 
