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
PYRSPATH=build/lib.linux-x86_64-2.7/:build/lib/

PYTHONPATH=$PYRSPATH:$PYTHONPATH build/scripts-2.7/pyrsplot $CMD 
