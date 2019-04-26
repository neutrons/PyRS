#!/bin/sh
python setup.py pyuic
python setup.py build

MANTIDLOCALPATH=/home/wzz/Mantid_Project/debug/bin/
MANTIDMACPATH=/Users/wzz/MantidBuild/debug/bin/
MANTIDSNSDEBUGPATH=/SNS/users/wzz/Mantid_Project/builds/debug/bin/
MANTIDPATH=$MANTIDMACPATH:$MANTIDLOCALPATH:$MANTIDSNSDEBUGPATH
PYTHONPATH=$MANTIDPATH:$PYTHONPATH
echo "PYTHON PATH: "
echo $PYTHONPATH
echo


if [ $1 ]; then
    CMD=$1
    CMDS=''
    for file in "$@"
    do
      if [ $file = $1 ] ; 
      then
	  echo "Ignore Item: "
	  echo $file
      else
          CMDS="$CMDS $file"
      fi
    done
else
    CMD=
    echo "PyRS Scripts Options:"
    echo "(1) mask (operation)"
    echo "(2) reduce (data)"
    echo "(3) convert (source data file to standard HDF5)"
fi

echo "User option: $1"

if [ "$1" = "1" ] || [ "$1" = "mask" ] ; then
    echo "Process masks/ROIs"
    PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH ./build/scripts-2.7/create_mask.py $CMDS
fi

if [ "$1" = "2" ] || [ "$1" = "reduce" ]  ; then
    echo "Reduce data"
    PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH ./build/scripts-2.7/reduce_HB2B.py $CMDS
fi

if [ "$1" = "3" ] || [ "$1" = "convert" ] ; then
    echo "Convert binary or image data file to HDF5"
    PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH ./build/scripts-2.7/convert_raw_data.py $CMDS
fi

