#!/bin/sh
python setup.py pyuic
python setup.py build

if [ $1 ]; then
    CMD=$1
else
    CMD=
    echo "1: peak fit, 2: HZB, 3: XRay, 4: manual reduction, 5: instrument geometry calibration"
fi

MANTIDLOCALPATH=/home/wzz/Mantid_Project/builds/mantid-python2/bin/
MANTIDMACPATH=/Users/wzz/MantidBuild/debug/bin/
MANTIDSNSDEBUGPATH=/opt/mantidnightly/bin/  # NIGHTLY for latest Pseudo-voigt
MANTIDPATH=$MANTIDMACPATH:$MANTIDLOCALPATH:$MANTIDSNSDEBUGPATH
PYTHONPATH=$MANTIDPATH:$PYTHONPATH
PYRSPATH=build/lib.linux-x86_64-2.7/:build/lib/

if [ "$1" = "1" ]
then 
	echo "Test peak fitting file"
        PYTHONPATH=build/lib:$PYTHONPATH scripts/preparetest/convert_hb2b_peaks.py
fi

if [ "$1" = "2" ]
then 
	echo "Test converting HZB data (for reduction and calibration test)"
        PYTHONPATH=build/lib:$PYTHONPATH scripts/preparetest/convert_hzb_data.py
fi

if [ "$1" = "3" ]
then 
	echo "Test converting XRay data (for reduction and calibration test)"
	PYTHONPATH=build/lib:$PYTHONPATH scripts/preparetest/convert_xray_data.py
fi

