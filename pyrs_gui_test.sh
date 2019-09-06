#!/bin/sh
python setup.py pyuic
python setup.py build

if [ $1 ]; then
    CMD=$1
else
    CMD=
    echo "1: peak fit, 2: texture, 3: strain stress, 4: manual reduction, 5: instrument geometry calibration"
fi

MANTIDLOCALPATH=/home/wzz/Mantid_Project/builds/mantid-python2/bin/
MANTIDMACPATH=/Users/wzz/MantidBuild/debug/bin/
MANTIDSNSDEBUGPATH=/opt/mantidnightly/bin/  # NIGHTLY for latest Pseudo-voigt
MANTIDPATH=$MANTIDMACPATH:$MANTIDLOCALPATH:$MANTIDSNSDEBUGPATH
PYTHONPATH=$MANTIDPATH:$PYTHONPATH
PYRSPATH=build/lib.linux-x86_64-2.7/:build/lib/

if [ "$1" = "1" ]
then 
	echo "Test peak fitting module"
        PYTHONPATH=$PYRSPATH:$PYTHONPATH python build/scripts-2.7/peakfitgui_test.py
fi

if [ "$1" = "2" ]
then 
	echo "Test texture calculation module"
        PYTHONPATH=$PYRSPATH:$PYTHONPATH build/scripts-2.7/texturegui_test.py
fi

if [ "$1" = "3" ]
then 
	echo "Test strain/stress calculation module"
	PYTHONPATH=$PYRSPATH:$PYTHONPATH build/scripts-2.7/strainstressgui_test.py
fi

if [ "$1" = "4" ]
then 
	echo "Test maual reduction mdoule"
	PYTHONPATH=$PYRSPATH:$PYTHONPATH build/scripts-2.7/manualreduction_test.py
fi

if [ "$1" = "5" ]
then 
	echo "Test instrument geometry calibration module"
	PYTHONPATH=build/lib:$PYTHONPATH build/scripts-2.7/calibration_gui_test.py
fi

