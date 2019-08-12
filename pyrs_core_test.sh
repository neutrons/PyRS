#!/bin/sh
python setup.py pyuic
python setup.py build

if [ $1 ]; then
    CMD=$1
else
    CMD=''
    echo "1: Reduction test (r); 2: Peak fitting test (f);  3: Prototype Chris (X-ray)"
fi

MANTIDLOCALPATH=/home/wzz/Mantid_Project/debug/bin/
MANTIDMACPATH=/Users/wzz/MantidBuild/debug/bin/
MANTIDSNSDEBUGPATH=/opt/mantidnightly/bin/  # NIGHTLY for latest Pseudo-voigt
MANTIDPATH=$MANTIDMACPATH:$MANTIDLOCALPATH:$MANTIDSNSDEBUGPATH
PYTHONPATH=$MANTIDPATH:$PYTHONPATH
PYRSPATH=build/lib.linux-x86_64-2.7/:build/lib/

if [ "$1" = "1" ] || [ "$1" = "r" ] ; then
    echo "Reduction Test"
    PYTHONPATH=$PYRSPATH:$PYTHONPATH build/scripts-2.7/reduction_test.py
fi

if [ "$1" = "2" ] || [ "$1" = "f" ] ; then
    echo "Peak Fitting Test"
    PYTHONPATH=build/lib:$PYTHONPATH scripts/preparetest/convert_hb2b_peaks.py
    PYTHONPATH=$PYRSPATH:$PYTHONPATH build/scripts-2.7/fit_peaks_test.py
fi

# PYTHONPATH=build/lib:$PYTHONPATH $CMD build/scripts-2.7/pyrs_core_test.py
# PYTHONPATH=build/lib:$PYTHONPATH $CMD build/scripts-2.7/utilities_test.py
# PYTHONPATH=build/lib:$PYTHONPATH $CMD build/scripts-2.7/peakfitgui_test.py
# PYTHONPATH=build/lib:$PYTHONPATH $CMD build/scripts-2.7/polefigurecal_test.py
# PYTHONPATH=build/lib:$PYTHONPATH $CMD build/scripts-2.7/straincalculationtest.py

# PYTHONPATH=build/lib:$PYTHONPATH $CMD build/scripts-2.7/verify_instrument_builders.py


# PYTHONPATH=build/lib:$PYTHONPATH $CMD build/scripts-2.7/instrument_geometry_test.py
# PYTHONPATH=build/lib:$PYTHONPATH $CMD build/scripts-2.7/test_reduced_hb2b.py 

# echo "Choice:"
# echo $CMD
# PYTHONPATH=build/lib:$PYTHONPATH build/scripts-2.7/compare_reduction_engines_test.py $CMD
