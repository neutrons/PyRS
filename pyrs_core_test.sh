#!/bin/sh
python setup.py pyuic
python setup.py build
if [ $1 ]; then
    CMD=$1
else
    CMD=''
fi
# PYTHONPATH=build/lib:$PYTHONPATH $CMD build/scripts-2.7/peakfitgui_test.py
# PYTHONPATH=build/lib:$PYTHONPATH $CMD build/scripts-2.7/texturegui_test.py
# PYTHONPATH=build/lib:$PYTHONPATH $CMD build/scripts-2.7/pyrs_core_test.py
# PYTHONPATH=build/lib:$PYTHONPATH $CMD build/scripts-2.7/utilities_test.py
PYTHONPATH=build/lib:$PYTHONPATH $CMD build/scripts-2.7/polefigurecal_test.py
