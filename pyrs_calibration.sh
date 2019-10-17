#!/bin/sh
python setup.py pyuic
python setup.py build

MANTIDLOCALPATH=/home/wzz/Mantid_Project/builds/mantid-python2/bin/
MANTIDMACPATH=/Users/wzz/MantidBuild/debug/bin/
MANTIDSNSDEBUGPATH=/opt/mantidnightly/bin/  # NIGHTLY for latest Pseudo-voigt
MANTIDPATH=$MANTIDMACPATH:$MANTIDLOCALPATH:$MANTIDSNSDEBUGPATH
PYTHONPATH=$MANTIDPATH:$PYTHONPATH
echo

if [ $1 ]; then
    CMD=$1
else
    CMD=""
    echo "1: Prototype X-ray; 2: Prototype HZB; 3: Prototype Chris (X-ray)"
fi

echo "User option: $1"

if [ "$1" = "1" ] || [ "$1" = "prototype" ] ; then
    echo "Protyping calibration"
    PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH python ./prototypes/calibration/Quick_Calibration_general.py
fi

if [ "$1" = "11" ] || [ "$1" = "prototype" ] ; then
    echo "Protyping calibration"
    PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH python ./prototypes/calibration/Quick_Calibration_PeakFit.py
fi

if [ "$1" = "111" ] || [ "$1" = "prototype" ] ; then
    echo "Protyping calibration"
    PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH python ./prototypes/calibration/Quick_Calibration_Class.py
fi

if [ "$1" = "22" ] || [ "$1" = "prototype" ] ; then
    echo "Protyping calibration"
    PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH python ./prototypes/calibration/Quick_Calibration_Corr_LMFIT.py
fi

if [ "$1" = "22" ] || [ "$1" = "prototype" ] ; then
    echo "Protyping calibration"
    PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH python ./prototypes/calibration/Quick_Calibration_Corr_LMFIT.py
fi


if [ "$1" = "31" ] || [ "$1" = "x" ] ; then
    echo "Test calibration prototype algorithm with X-ray data"
    # PYTHONPATH=$PYRSPATH:$PYTHONPATH python scripts/preparetest/convert_xray_data.py
    PYTHONPATH=$PYRSPATH:$PYTHONPATH python scripts/calibrations/calibrate_xray_prototype.py -i tests/testdata/Hidra_XRay_LaB6_10kev_35deg.hdf -o tests/cal.json --instrument=my.txt -m tests/testdata/calibration/xray_test_masks.txt
    PYTHONPATH=$PYRSPATH:$PYTHONPATH python tests/quicktest/plot_reduced_data.py tests/testdata/Hidra_XRay_LaB6_10kev_35deg.hdf
fi


if [ "$1" = "33" ] || [ "$1" = "chris" ] ; then
    echo "Protyping calibration"
    PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH python ./prototypes/calibration/Quick_Calibration_general.py
fi
