#!/bin/sh
# set the mantidpython to use - default to system installed nightly
if [ -n "$1" ] && [ $(command -v $1) ]; then
    MANTIDPYTHON="$1"
else
    MANTIDPYTHON=mantidpython50
fi

# check that a valid mantidpython was specified
if [ ! $(command -v $MANTIDPYTHON) ]; then
    echo "Failed to find mantidpython \"$MANTIDPYTHON\""
    exit -1
fi

$MANTIDPYTHON setup.py build

# let people know what is going on and launch it
echo "Using \"$(which $MANTIDPYTHON)\""
QT_API=pyqt5 PYTHONPATH=`pwd`/build/lib $MANTIDPYTHON --classic `pwd`/build/scripts-2.7/pyrsplot
