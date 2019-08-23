#!/bin/sh
python setup.py pyuic
python setup.py build

echo
MANTIDLOCALPATH=/home/wzz/Mantid_Project/debug/bin/
MANTIDMACPATH=/Users/wzz/MantidBuild/debug/bin/
MANTIDSNSPATH=/opt/mantidnightly/bin/
MANTIDPATH=$MANTIDMACPATH:$MANTIDLOCALPATH:$MANTIDSNSPATH
PYTHONPATH=$MANTIDPATH:$PYTHONPATH

PYTHONPATH=/SNS/users/hcf/PyRS/build/lib:/SNS/users/hcf/PyRS/build/lib.linux-x86_64-2.7:$PYTHONPATH python scripts/create_mask.py --roi=tests/testdata/masks/NegZ_Mask.xml --operation='reverse' --output=tests/testdata/masks/NegZ_Mask.h5

PYTHONPATH=/SNS/users/hcf/PyRS/build/lib:/SNS/users/hcf/PyRS/build/lib.linux-x86_64-2.7:$PYTHONPATH python scripts/create_mask.py --roi=tests/testdata/masks/NegZ_Mask.xml  --output=tests/testdata/masks/PosZ_Mask.h5

PYTHONPATH=/SNS/users/hcf/PyRS/build/lib:/SNS/users/hcf/PyRS/build/lib.linux-x86_64-2.7:$PYTHONPATH python scripts/create_mask.py --roi=tests/testdata/masks/Chi_30_Mask.xml  --operation='reverse' --output=tests/testdata/masks/Chi_30_Both.h5

PYTHONPATH=/SNS/users/hcf/PyRS/build/lib:/SNS/users/hcf/PyRS/build/lib.linux-x86_64-2.7:$PYTHONPATH python scripts/create_mask.py --roi=tests/testdata/masks/Chi_20_Mask.xml  --operation='reverse' --output=tests/testdata/masks/Chi_20_Both.h5

PYTHONPATH=/SNS/users/hcf/PyRS/build/lib:/SNS/users/hcf/PyRS/build/lib.linux-x86_64-2.7:$PYTHONPATH python scripts/create_mask.py --roi=tests/testdata/masks/Chi_10_Mask.xml  --operation='reverse' --output=tests/testdata/masks/Chi_10_Both.h5


PYTHONPATH=/SNS/users/hcf/PyRS/build/lib:/SNS/users/hcf/PyRS/build/lib.linux-x86_64-2.7:$PYTHONPATH python scripts/create_mask.py --roi=tests/testdata/masks/Chi_30_Both.h5  --roi=tests/testdata/masks/PosZ_Mask.h5 --operation='and' --output=tests/testdata/masks/Step2.hdf5

PYTHONPATH=/SNS/users/hcf/PyRS/build/lib:/SNS/users/hcf/PyRS/build/lib.linux-x86_64-2.7:$PYTHONPATH python scripts/create_mask.py --roi=tests/testdata/masks/Step2.hdf5  --roi=tests/testdata/masks/NegZ_Mask.h5 --operation='or' --output=tests/testdata/masks/Step3.hdf5

PYTHONPATH=/SNS/users/hcf/PyRS/build/lib:/SNS/users/hcf/PyRS/build/lib.linux-x86_64-2.7:$PYTHONPATH python scripts/create_mask.py --roi=tests/testdata/masks/Step3.hdf5 --operation='reverse' --output=tests/testdata/masks/Chi_30.hdf5

PYTHONPATH=/SNS/users/hcf/PyRS/build/lib:/SNS/users/hcf/PyRS/build/lib.linux-x86_64-2.7:$PYTHONPATH python scripts/create_mask.py --roi=tests/testdata/masks/Chi_30_Both.h5  --roi=tests/testdata/masks/NegZ_Mask.h5 --operation='and' --output=tests/testdata/masks/Step2.hdf5 

PYTHONPATH=/SNS/users/hcf/PyRS/build/lib:/SNS/users/hcf/PyRS/build/lib.linux-x86_64-2.7:$PYTHONPATH python scripts/create_mask.py --roi=tests/testdata/masks/Step2.hdf5  --roi=tests/testdata/masks/PosZ_Mask.h5 --operation='or' --output=tests/testdata/masks/Step3.hdf5 

PYTHONPATH=/SNS/users/hcf/PyRS/build/lib:/SNS/users/hcf/PyRS/build/lib.linux-x86_64-2.7:$PYTHONPATH python scripts/create_mask.py --roi=tests/testdata/masks/Step3.hdf5 --operation='reverse' --output=tests/testdata/masks/Chi_30.hdf5


PYTHONPATH=/SNS/users/hcf/PyRS/build/lib:/SNS/users/hcf/PyRS/build/lib.linux-x86_64-2.7:$PYTHONPATH python scripts/create_mask.py --roi=tests/testdata/masks/Chi_20_Both.h5  --roi=tests/testdata/masks/NegZ_Mask.h5 --operation='and' --output=tests/testdata/masks/Step2.hdf5 

PYTHONPATH=/SNS/users/hcf/PyRS/build/lib:/SNS/users/hcf/PyRS/build/lib.linux-x86_64-2.7:$PYTHONPATH python scripts/create_mask.py --roi=tests/testdata/masks/Step2.hdf5  --roi=tests/testdata/masks/PosZ_Mask.h5 --operation='or' --output=tests/testdata/masks/Step3.hdf5 

PYTHONPATH=/SNS/users/hcf/PyRS/build/lib:/SNS/users/hcf/PyRS/build/lib.linux-x86_64-2.7:$PYTHONPATH python scripts/create_mask.py --roi=tests/testdata/masks/Step3.hdf5 --operation='reverse' --output=tests/testdata/masks/Chi_Neg20.hdf5

PYTHONPATH=/SNS/users/hcf/PyRS/build/lib:/SNS/users/hcf/PyRS/build/lib.linux-x86_64-2.7:$PYTHONPATH python scripts/create_mask.py --roi=tests/testdata/masks/Chi_20_Both.h5  --roi=tests/testdata/masks/NegZ_Mask.h5 --operation='and' --output=tests/testdata/masks/Step2.hdf5 

PYTHONPATH=/SNS/users/hcf/PyRS/build/lib:/SNS/users/hcf/PyRS/build/lib.linux-x86_64-2.7:$PYTHONPATH python scripts/create_mask.py --roi=tests/testdata/masks/Step2.hdf5  --roi=tests/testdata/masks/PosZ_Mask.h5 --operation='or' --output=tests/testdata/masks/Step3.hdf5 

PYTHONPATH=/SNS/users/hcf/PyRS/build/lib:/SNS/users/hcf/PyRS/build/lib.linux-x86_64-2.7:$PYTHONPATH python scripts/create_mask.py --roi=tests/testdata/masks/Step3.hdf5 --operation='reverse' --output=tests/testdata/masks/Chi_20.hdf5




PYTHONPATH=/SNS/users/hcf/PyRS/build/lib:/SNS/users/hcf/PyRS/build/lib.linux-x86_64-2.7:$PYTHONPATH python scripts/create_mask.py --roi=tests/testdata/masks/Chi_10_Both.h5  --roi=tests/testdata/masks/NegZ_Mask.h5 --operation='and' --output=tests/testdata/masks/Step2.hdf5 

PYTHONPATH=/SNS/users/hcf/PyRS/build/lib:/SNS/users/hcf/PyRS/build/lib.linux-x86_64-2.7:$PYTHONPATH python scripts/create_mask.py --roi=tests/testdata/masks/Step2.hdf5  --roi=tests/testdata/masks/PosZ_Mask.h5 --operation='or' --output=tests/testdata/masks/Step3.hdf5 

PYTHONPATH=/SNS/users/hcf/PyRS/build/lib:/SNS/users/hcf/PyRS/build/lib.linux-x86_64-2.7:$PYTHONPATH python scripts/create_mask.py --roi=tests/testdata/masks/Step3.hdf5 --operation='reverse' --output=tests/testdata/masks/Chi_Neg10.hdf5

PYTHONPATH=/SNS/users/hcf/PyRS/build/lib:/SNS/users/hcf/PyRS/build/lib.linux-x86_64-2.7:$PYTHONPATH python scripts/create_mask.py --roi=tests/testdata/masks/Chi_10_Both.h5  --roi=tests/testdata/masks/NegZ_Mask.h5 --operation='and' --output=tests/testdata/masks/Step2.hdf5 

PYTHONPATH=/SNS/users/hcf/PyRS/build/lib:/SNS/users/hcf/PyRS/build/lib.linux-x86_64-2.7:$PYTHONPATH python scripts/create_mask.py --roi=tests/testdata/masks/Step2.hdf5  --roi=tests/testdata/masks/PosZ_Mask.h5 --operation='or' --output=tests/testdata/masks/Step3.hdf5 

PYTHONPATH=/SNS/users/hcf/PyRS/build/lib:/SNS/users/hcf/PyRS/build/lib.linux-x86_64-2.7:$PYTHONPATH python scripts/create_mask.py --roi=tests/testdata/masks/Step3.hdf5 --operation='reverse' --output=tests/testdata/masks/Chi_10.hdf5
