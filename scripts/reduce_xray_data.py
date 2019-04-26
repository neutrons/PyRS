import sys
from pyrs.core import reductionengine
from pyrs.utilities import checkdatatypes


def parse_information_file(info_file_name):
    """ Parse information file
    Example:
    # name   value      unit
    File     xxx.TIFF (or xxx.bin)
    Mask     mask1.h5   mask2.h5
    cal::arm  0.01     meter
    2theta    30.      degree
    :param info_file_name:
    :return:
    """
    checkdatatypes.check_file_name(info_file_name, True, False, False, description='Information file name')

    # import file
    info_file = open(info_file_name, 'r')
    info_lines = info_file.readlines()
    info_file.close()

    # parse
    info_dict = dict()
    for line in info_lines:
        line = line.strip()
        if line == '':
            continue   # empty line
        elif line.startswith('#'):
            continue   # information line

        terms = line.split()
        if terms[0] == 'file':
            info_dict['file'] = terms[1]
        elif terms[0] == 'mask':
            info_dict['mask'] = terms[1:]
        else:
            info_dict[terms[0]] = terms[1], terms[2]   # log name = log value, unit
    # END-FOR

    return info_dict


def main(argv):
    """
    Main method to reduce X-ray 2D data as a prototype for HB2B reduction
    :param argv:
    :return:
    """
    if len(argv) < 4:
        print('Auto-reducing HB2B: {} [Information File] [Output File Name]  [Method]'.format(argv[0]))
        sys.exit(-1)

    # parse input & check
    information_file_name = argv[1]
    data_info_dict = parse_information_file(information_file_name)

    output_file_name = argv[2]
    checkdatatypes.check_file_name(output_file_name, False, True, False, description='Output file name')

    reduction_method = argv[3]

    # call for reduction
    reduction_engine = reductionengine.HB2BReductionManager()
    reduction_engine.load_data(data_info_dict['file'])
    reduction_engine.set_2theta(data_info_dict['2theta'][0], data_info_dict['2theta'][1])
    reduction_engine.set_instrument_calibration(data_info_dict)

    # check whether reduction will use Mantid's methods
    if reduction_method.lower().count('mantid'):
        use_mantid = True
    else:
        use_mantid = False

    # reduce to 2theta as a whoe
    reduction_engine.reduce_to_2theta(use_mantid, plot=True, output=output_file_name)

    # now mask:
    for mask_h5 in data_info_dict['mask']:
        reduction_engine.reduce_rs_nexus(use_mantid, mask_h5, plot=True, output=output_file_name)

    return


if __name__ == '__main__':
    main(sys.argv)
