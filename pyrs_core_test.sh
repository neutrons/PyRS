#!/bin/sh
python setup.py pyuic
python setup.py build

if [ $1 ]; then
    CMD=$1
else
    CMD=''
    echo "1: Reduction test (r); 2: Prototype HZB; 3: Prototype Chris (X-ray)"
fi

MANTIDLOCALPATH=/home/wzz/Mantid_Project/debug/bin/
MANTIDMACPATH=/Users/wzz/MantidBuild/debug/bin/
MANTIDSNSDEBUGPATH=/opt/mantidnightly/bin/  # NIGHTLY for latest Pseudo-voigt
MANTIDPATH=$MANTIDMACPATH:$MANTIDLOCALPATH:$MANTIDSNSDEBUGPATH
PYTHONPATH=$MANTIDPATH:$PYTHONPATH

if [ "$1" = "1" ] || [ "$1" = "r" ] ; then
    echo "Reduction Test"
    PYTHONPATH=build/lib:$PYTHONPATH build/scripts-2.7/reduction_test.py
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
