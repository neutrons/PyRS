sh pyrs_command.sh 1 --roi=tests/testdata/masks/NegZ_Mask.xml --output=tests/testdata/masks/NegZ.hdf5 --note=ROI_NegZ
sh pyrs_command.sh 1 --roi=tests/testdata/masks/NegZ_Mask.xml --output=tests/testdata/masks/PosZ.hdf5 --note=ROI_PosZ --operation=reverse 
sh pyrs_command.sh 1 --roi=tests/testdata/masks/Chi_0_Mask.xml --opeation=reverse --output=tests/testdata/masks/Chi_0.hdf5 --note=35_Degree_SolidAngle_0
sh pyrs_command.sh 1 --roi=tests/testdata/masks/Chi_10_Mask.xml --roi=tests/testdata/masks/PosZ.hdf5 --operation=and --output=tests/testdata/masks/Chi_10.hdf5 --note=35_Degree_SolidAngle_Pos10
sh pyrs_command.sh 1 --roi=tests/testdata/masks/Chi_10_Mask.xml --roi=tests/testdata/masks/NegZ.hdf5 --operation=and --output=tests/testdata/masks/Chi_Neg10.hdf5 --note=35_Degree_SolidAngle_Neg10
sh pyrs_command.sh 1 --roi=tests/testdata/masks/Chi_20_Mask.xml --roi=tests/testdata/masks/PosZ.hdf5 --operation=and --output=tests/testdata/masks/Chi_20.hdf5 --note=35_Degree_SolidAngle_Pos20
sh pyrs_command.sh 1 --roi=tests/testdata/masks/Chi_20_Mask.xml --roi=tests/testdata/masks/NegZ.hdf5 --operation=and --output=tests/testdata/masks/Chi_Neg20.hdf5 --note=35_Degree_SolidAngle_Neg20
sh pyrs_command.sh 1 --roi=tests/testdata/masks/Chi_30_Mask.xml --roi=tests/testdata/masks/PosZ.hdf5 --operation=and --output=tests/testdata/masks/Chi_30.hdf5 --note=35_Degree_SolidAngle_Pos30
sh pyrs_command.sh 1 --roi=tests/testdata/masks/Chi_30_Mask.xml --roi=tests/testdata/masks/NegZ.hdf5 --operation=and --output=tests/testdata/masks/Chi_Neg30.hdf5 --note=35_Degree_SolidAngle_Neg30
