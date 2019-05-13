#!/bin/sh
python setup.py pyuic
python setup.py build

echo 
MANTIDLOCALPATH=/home/wzz/Mantid_Project/debug/bin/
MANTIDMACPATH=/Users/wzz/MantidBuild/debug/bin/
MANTIDSNSPATH=/opt/mantidnightly/bin
# MANTIDSNSDEBUGPATH=/SNS/users/wzz/Mantid_Project/builds/debug/bin/
MANTIDPATH=$MANTIDMACPATH:$MANTIDLOCALPATH:$MANTIDSNSPATH
PYTHONPATH=$MANTIDPATH:$PYTHONPATH
echo "PYTHON PATH: "
echo $PYTHONPATH
echo


if [ $1 ]; then
    CMD=$1
    CMDS=''
    for file in "$@"
    do
      echo $file
      if [ $file = $1 ] ; 
      then
	  echo "Ignore "
	  echo $file
      else
          CMDS="$CMDS $file"
      fi
    done
else
    CMD=
    echo "Study PyRS and Mantid Reduction (histogram). "
    echo "(1) reduce without          mask without calibration"
    echo "(2) reduce with    0 degree mask without calibration"
    echo "(3) reduce with   10 degree mask without calibration"
    echo "(4) reduce with   20 degree mask without calibration"
    echo "(5) reduce with   30 degree mask without calibration"
fi

echo "User option: $1"


if [ "$1" = "1" ] || [ "$1" = "" ] ; then
    echo "Comparing instrument geometry"
    PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH ./build/scripts-2.7/reduction_study.py 1
fi

if [ "$1" = "2" ] || [ "$1" = "0d" ] ; then
    echo "Comparing instrument geometry"
    PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH ./build/scripts-2.7/reduction_study.py 2
fi

if [ "$1" = "5" ] || [ "$1" = "30d" ] ; then
    echo "Comparing instrument geometry"
    PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH ./build/scripts-2.7/reduction_study.py 5
fi
