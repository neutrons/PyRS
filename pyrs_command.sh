#!/bin/sh
python setup.py pyuic
python setup.py build

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
    echo "(4) Reduce HZB data (1 sub run)"
    echo "(5) Reduce HZB data (all sub runs)"
fi

MANTIDLOCALPATH=/home/wzz/Mantid_Project/builds/mantid-python2/bin/
MANTIDMACPATH=/Users/wzz/MantidBuild/debug/bin/
MANTIDSNSDEBUGPATH=/opt/mantidnightly/bin/  # NIGHTLY for latest Pseudo-voigt
MANTIDPATH=$MANTIDMACPATH:$MANTIDLOCALPATH:$MANTIDSNSDEBUGPATH
PYTHONPATH=$MANTIDPATH:$PYTHONPATH
PYRSPATH=build/lib.linux-x86_64-2.7/:build/lib/


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
    # PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH ./build/scripts-2.7/convert_raw_data.py $CMDS
    PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH ./build/scripts-2.7/convert_hzb_data.py $CMDS
fi

if [ "$1" = "4" ] || [ "$1" = "hzb1" ]  ; then
    echo "Reduce data test (HBZ) sub run 1"
    # tests/testdata/hzb/hzb_calibration.hdf5 
    PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH ./build/scripts-2.7/reduce_HB2B.py tests/testdata/HZB_Raw_Project.hdf tests/temp/ --instrument=tests/testdata/hzb/HZB_Definition_20190523_0844.txt --subrun=1
fi

if [ "$1" = "5" ] || [ "$1" = "hzball" ]  ; then
    echo "Reduce data test (HBZ) all sub runs"
    PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH ./build/scripts-2.7/reduce_HB2B.py tests/testdata/HZB_Raw_Project.hdf tests/temp/ --instrument=tests/testdata/hzb/HZB_Definition_20190523_0844.txt
fi

if [ "$1" = "6" ] || [ "$1" = "convert" ]  ; then
    echo "Reduce data test (HBZ) all sub runs"
    PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH ./build/scripts-2.7/convert_nexus_to_hidra.py --nexus=/HFIR/HB2B/IPTS-22731/nexus/HB2B_439.nxs.h5 --output=/tmp/HB2B_439.hdf
fi

