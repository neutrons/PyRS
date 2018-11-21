import sys
from pyrs.core import reductionengine
from pyrs.utilities import checkdatatypes


def main(argv):
    """
    main body
    :param argv:
    :return:
    """
    if len(argv) != 3:
        print('Auto-reducing HB2B: {} [NeXus File Name] [Target Directory]'.format(argv[0]))
        sys.exit(-1)

    # parse input & check
    nexus_name = argv[1]
    output_dir = argv[2]
    checkdatatypes.check_file_name(nexus_name, True, False, False, description='NeXus file name')
    checkdatatypes.check_file_name(output_dir, True, True, True, description='Auto reduction output')

    # call for reduction
    reduction_engine = reductionengine.ReductionEngine()
    reduction_engine.reduce_rs_nexus(nexus_name, auto_mapping_check=True, output_dir=output_dir, do_calibration=True,
                                     allow_calibration_unavailable=True)

    return


if __name__ == '__main__':
    main(sys.argv)
