#!/bin/sh
python setup.py pyuic
python setup.py build

echo 
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
    echo "Options: (1) mask (2) reduce (3) mask test (4) reduction test (5) reduction test on 1024x1024"
    echo "(8) Peak processing UI test"
    echo "Options: Test all commands: \"all\""
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

if [ "$1" = "3" ] || [ "$1" = "masktest" ] ; then
	echo "Testing Mask"
	TestArgs="--roi=tests/testdata/masks/Chi_0_Mask.xml --output=Chi_0.hdf5 --operation=reverse --2theta=35."
	PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH ./build/scripts-2.7/create_mask.py $TestArgs
	# --roi=tests/testdata/masks/Chi_0_Mask.xml --output=tests/testdata/masks/Chi_0.hdf5
fi

if [ "$1" = "4" ] || [ "$1" = "reducetest" ] ; then
    echo "Testing Reduction"
    TestArgs1=" ./tests/testdata/LaB6_10kev_35deg-00004_Rotated.tif ./tests/temp/ --mask=tests/testdata/masks/Chi_0.hdf5 --viewraw=1"
    # Mantid reduction engine
    TestArgs2=" ./tests/testdata/LaB6_10kev_35deg-00004_Rotated.tif ./tests/temp/ --mask=tests/testdata/masks/Chi_0.hdf5 --viewraw=0 --instrument=tests/testdata/XRay_Definition_2K.xml --2theta=35."
    PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH ./build/scripts-2.7/reduce_HB2B.py $TestArgs2
fi

if [ "$1" = "5" ] || [ "$1" = "reduce1024" ] ; then
    echo "Testing Reduction: 1024 x 1024"
    TestArgs=" ./tests/testdata/LaB6_10kev_35deg-00004_Rotated.bin ./tests/temp/ --mask=tests/testdata/masks/Chi_0.hdf5 --viewraw=0 --instrument=tests/testdata/XRay_Definition_1K.xml --2theta=35."
    PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH ./build/scripts-2.7/reduce_HB2B.py $TestArgs
fi

