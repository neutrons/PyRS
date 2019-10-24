```#! /usr/bin/env bash
set -epu

# Running the setup.py scripts
python setup.py pyuic
python setup.py build

# Setting script-wide scoped variables.
declare MANTIDLOCALPATH=/home/wzz/Mantid_Project/debug/bin/
declare MANTIDMACPATH=/Users/wzz/MantidBuild/debug/bin/
# declare MANTIDSNSDEBUGPATH=/SNS/users/wzz/Mantid_Project/builds/debug/bin/
declare MANTIDPATH=$MANTIDMACPATH:$MANTIDLOCALPATH:$MANTIDSNSDEBUGPATH
declare PYTHONPATH=$MANTIDPATH:$PYTHONPATH

# Printing the current python path.
printf "PYTHON PATH: %s\n" "${PYTHONPATH}"

func_help_message() {
    HELP_MESSAGE="Verify PyRS and Mantid Reduction. Options:
(1) [compare] geometry (XRay Mock)
(2) [compare] 2theta
(4) [compare] reduction-all (without masking)
(5) [compare] reduction-0   (with ROI on 0 degree)
(6) [compare] reduction-10  (with ROI on +10 degree)
(7) [compare] reduction-20  (with ROI on +20 degree)
(8) [compare] reduction-30  (with ROI on +30 degree)
(21) [verify] HBZ geometry: aka HBZ-IDF
(111) [study] reduction study
(104) reduce-pyrs-all
(105) reduce-pyrs-0degree
(106) reduce-pyrs-10degree
Options: Test all commands: \"all\""
    printf "%s\n" "${HELP_MESSAGE}"
}

if [[ $# -ge 1 ]]; then
    CMD="${1}"
    CMDS=''
    for FILE in "$@"; do
      echo "${FILE}"
      if [[ "${FILE}" = "${CMD}" ]]; then
	       echo "Ignore "
	       echo "${FILE}"
      else
          CMDS="${CMDS} ${FILE}"
      fi
    done
else
    func_help_message
    exit 1
fi 

echo "User option: $1"

# Ordered by latest test case

case "${CMD}" in
    1|"geometry")
        echo "Comparing instrument geometry"
        # PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH ./build/scripts-2.7/compare_reduction_engines_test.py 1
        PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH ./build/scripts-2.7/compare_reduction_engines_test.py 2;;
    2|"2theta")
        echo "Comparing converted 2theta from geometry"
        PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH ./build/scripts-2.7/compare_reduction_engines_test.py 3;;
        # PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH ./build/scripts-2.7/compare_reduction_engines_test.py 4;;
    3|"counts")
        echo "With engine changed, no need to compare counts any more";;
    4|"reduction-all")
        echo "Testing Reduction Without ROI/Mask"
        PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH ./build/scripts-2.7/compare_reduction_engines_test.py 5;;
        # PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH ./build/scripts-2.7/compare_reduction_engines_test.py 6;;
    5|"reduction-0")
        echo "Testing Reduction with ROI around solid angle 0 degree"
        PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH ./build/scripts-2.7/compare_reduction_engines_test.py 7;;
        # PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH ./build/scripts-2.7/compare_reduction_engines_test.py 8;;
    6|"reduction-10")
        echo "Testing Reduction with ROI around solid angle +10 degree"
        PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH ./build/scripts-2.7/compare_reduction_engines_test.py 9;;
    7|"reduction-20")
        echo "Testing Reduction with ROI around solid angle +20 degree"
        PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH python ./build/scripts-2.7/compare_reduction_engines_test.py 11;;
    8|"reduction-30")
        echo "Testing Reduction with ROI around solid angle +30 degree"
        PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH python ./build/scripts-2.7/compare_reduction_engines_test.py 13;;
    9|"prototype")
        echo "Process masks/ROIs"
        PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH python ./tests/Quick_Calibration_v2.py;;
    21|"HBZ-IDF")
        echo "Comparing and verifying instrument geometry for HBZ (IDF and PyRS configuration)"
        PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH ./build/scripts-2.7/compare_reduction_engines_test.py 21;;
        # PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH ./build/scripts-2.7/compare_reduction_engines_test.py 22;;
    99|"unknown")
        TestArgs3=" ./tests/testdata/LaB6_10kev_35deg-00004_Rotated.tif ./tests/temp/ --viewraw=1 --mask=tests/testdata/masks/Chi_0.hdf5 --instrument=tests/testdata/XRay_Definition_2K.xml --2theta=-35."
        PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH ./build/scripts-2.7/reduce_HB2B.py $TestArgs3
        TestArgs2=" ./tests/testdata/LaB6_10kev_35deg-00004_Rotated.tif ./tests/temp/ --viewraw=1  --instrument=tests/testdata/XRay_Definition_2K.xml --2theta=-35."
        PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH ./build/scripts-2.7/reduce_HB2B.py $TestArgs2
        TestArgs="--roi=tests/testdata/masks/Chi_0_Mask.xml --output=Chi_0.hdf5 --operation=reverse --2theta=35."
        TestArgs1=" ./tests/testdata/LaB6_10kev_35deg-00004_Rotated.tif ./tests/temp/ --mask=tests/testdata/masks/Chi_0.hdf5 --viewraw=1"
    
        # Mantid reduction engine
        TestArgs2=" ./tests/testdata/LaB6_10kev_35deg-00004_Rotated.tif ./tests/temp/ --mask=tests/testdata/masks/Chi_0.hdf5 --viewraw=0 --instrument=tests/testdata/XRay_Definition_2K.xml --2theta=-35."
        # TestArgs2=" ./tests/testdata/LaB6_10kev_35deg-00004_Rotated.tif ./tests/temp/ --viewraw=0 --instrument=tests/testdata/XRay_Definition_2K.xml --2theta=-35."
        # PyRS reduction engine
        TestArgs3=" ./tests/testdata/LaB6_10kev_35deg-00004_Rotated.tif ./tests/temp/ --mask=tests/testdata/masks/Chi_30.hdf5 --viewraw=0 --instrument=tests/testdata/xray_data/XRay_Definition_2K.txt --2theta=-35."
        PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH ./build/scripts-2.7/reduce_HB2B.py $TestArgs2;;
    105|"reduce-pyrs-0degree")
        echo "Testing Reduction with ROI around solid angle +35 degree"
        # PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH ./build/scripts-2.7/compare_reduction_engines_test.py 11
        # PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH ./build/scripts-2.7/compare_reduction_engines_test.py 12
        # PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH ./build/scripts-2.7/compare_reduction_engines_test.py 13
    
    
        echo "Testing Reduction with ROI around solid angle +35 degree"
        TestArgsMantid=" ./tests/testdata/LaB6_10kev_35deg-00004_Rotated.h5 ./tests/temp/ --viewraw=1  --instrument=tests/testdata/XRay_Definition_2K.xml --2theta=35."
        TestArgsPyRS=" ./tests/testdata/LaB6_10kev_35deg-00004_Rotated.h5 ./tests/temp/ --viewraw=1  --instrument=tests/testdata/XRay_Definition_2K.txt --2theta=35."
        PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH ./build/scripts-2.7/reduce_HB2B.py $TestArgsPyRS
        TestArgsMantid=" ./tests/testdata/LaB6_10kev_35deg-00004_Rotated.h5 ./tests/temp/ --viewraw=1 --mask=tests/testdata/masks/Chi_0.hdf5 --instrument=tests/testdata/XRay_Definition_2K.xml --2theta=35."
        TestArgsPyRS=" ./tests/testdata/LaB6_10kev_35deg-00004_Rotated.h5 ./tests/temp/ --viewraw=1 --mask=tests/testdata/masks/Chi_0.hdf5 --instrument=tests/testdata/XRay_Definition_2K.txt --2theta=35."
        PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH ./build/scripts-2.7/reduce_HB2B.py $TestArgsPyRS;;
    111|"study")
    echo "Reducing (histogram) with solid angle 30 mask"
      # PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH python ./build/scripts-2.7/reduction_study.py tests/testdata/LaB6_10kev_35deg-00004_Rotated_TIF.h5 35. tests/testdata/masks/Chi_30.hdf5 pyrs
      # PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH python ./build/scripts-2.7/reduction_study.py tests/testdata/LaB6_10kev_35deg-00004_Rotated_TIF.h5 35. tests/testdata/masks/Chi_Neg30.hdf5 pyrs
      # PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH python ./build/scripts-2.7/reduction_study.py tests/testdata/LaB6_10kev_35deg-00004_Rotated_TIF.h5 35. tests/testdata/masks/Chi_0.hdf5 pyrs
      # PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH python ./build/scripts-2.7/reduction_study.py tests/testdata/LaB6_10kev_35deg-00004_Rotated_TIF.h5 35. tests/testdata/masks/Chi_10.hdf5 pyrs
      # PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH python ./build/scripts-2.7/reduction_study.py tests/testdata/LaB6_10kev_35deg-00004_Rotated_TIF.h5 35. tests/testdata/masks/Chi_20.hdf5 pyrs
      PYTHONPATH=build/lib:build/lib.linux-x86_64-2.7:$PYTHONPATH python ./build/scripts-2.7/reduction_study.py tests/testdata/LaB6_10kev_35deg-00004_Rotated_TIF.h5 35. None pyrs;;
esac
```