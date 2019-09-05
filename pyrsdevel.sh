#!/bin/sh
python setup.py pyuic
python setup.py build

CMDS=''
for file in "$@"
do
  CMDS="$CMDS $file"
done

# Set mantid path on different platform
MANTIDLOCALPATH=/home/wzz/Mantid_Project/builds/mantid-python2/bin/
MANTIDMACPATH=/Users/wzz/MantidBuild/debug/bin/
MANTIDSNSDEBUGPATH=/opt/mantidnightly/bin/  # NIGHTLY for latest Pseudo-voigt
MANTIDPATH=$MANTIDMACPATH:$MANTIDLOCALPATH:$MANTIDSNSDEBUGPATH
PYTHONPATH=$MANTIDPATH:$PYTHONPATH
PYRSPATH=build/lib.linux-x86_64-2.7/:build/lib/

PYTHONPATH=$PYRSPATH:$PYTHONPATH build/scripts-2.7/pyrsplot $CMD 
