#!/bin/sh
python setup.py pyuic
python setup.py build
echo "Not used /home/wzz/Mantid_Project/builds/debug-master/bin/"
MANTIDLOCALPATH=/home/wzz/Mantid_Project/debug/bin/
MANTIDMACPATH=/Users/wzz/MantidBuild/debug/bin/
# MANTIDSNSDEBUGPATH=/SNS/users/wzz/Mantid_Project/builds/debug/bin/
MANTIDSNSDEBUGPATH=/opt/mantidnightly/bin/  # NIGHTLY for latest Pseudo-voigt
MANTIDPATH=$MANTIDMACPATH:$MANTIDLOCALPATH:$MANTIDSNSDEBUGPATH
PYTHONPATH=$MANTIDPATH:$PYTHONPATH
echo "PYTHON PATH"
echo $PYTHONPATH
echo

if [ $1 ]; then
    CMD=$1
else
    CMD=
    echo "1: peak fit, 2: texture, 3: strain stress, 4: manual reduction, 5: instrument geometry calibration"
fi

if [ "$1" = "1" ]
then 
	echo "Test peak fitting file"
        PYTHONPATH=build/lib:$PYTHONPATH scripts/preparetest/convert_hb2b_peaks.py
fi

if [ "$1" = "2" ]
then 
	echo "Test converting HZB data"
        PYTHONPATH=build/lib:$PYTHONPATH scripts/preparetest/convert_hzb_data.py
fi

if [ "$1" = "3" ]
then 
	echo "Doing Nothing"
	# PYTHONPATH=build/lib:$PYTHONPATH build/scripts-2.7/strainstressgui_test.py
fi

