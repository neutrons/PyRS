# check that a valid mantidpython was found
MANTIDPYTHON=mantidpython50
if [ ! $(command -v $MANTIDPYTHON) ]; then
    echo "Failed to find mantidpython \"$MANTIDPYTHON\""
    exit -1
fi

# directory this script is in
DIREC=$(dirname $0)

# launch pyrs
PYTHONPATH=$DIREC/build/lib $MANTIDPYTHON --classic $DIREC/build/scripts-2.7/reduce_manual_HB2B.py $@