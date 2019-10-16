# Import mantid mask files (in XML format), do operation on them if necessary and save to HDF5
import sys


def parse_inputs(input_args):
    """
    parse inputs
    Example
    roi=file1.xml  mask=file2.xml operation=AND/OR/REVERT/INFO output=new.h5 pixel=2048
    :param input_args:
    :return:
    """
    roi_list = list()
    mask_list = list()

    operation = 'INFO'
    h5_name = None
    linear_pixel_size = 2048

    for argument in input_args:
        terms = argument.split('=')
        arg_type = terms[0].strip().lower()
        arg_value = terms[1].strip()
        if arg_type == 'roi':
            # ROI xml file
            roi_list.append(arg_value)
        elif arg_type == 'mask':
            # Mask xml file
            mask_list.append(arg_value)
        elif arg_type == 'operation':
            operation = arg_value.upper()
        elif arg_type == 'output':
            h5_name = arg_value
        elif arg_type == 'pixel':
            linear_pixel_size = int(arg_value)
        else:
            print('[ERROR] Input argument {} is not supported'.format(arg_type))
    # END-FOR

    return roi_list, mask_list, operation, h5_name, linear_pixel_size


def main(argv):
    """

    :param argv:
    :return:
    """
    if len(argv) < 2:
        print()
        sys.exit(-1)

    roi_xml_list, mask_xml_list, mask_operation, out_h5_name, square_det_size = parse_inputs(argv)

    # parse ROI and mask XML (Mantid) files
    mask_array_list = list()
    for roi_xml in roi_xml_list:
        masking_array = file_utilities.load_mantid_mask(square_det_size, roi_xml, False)
        mask_array_list.append(masking_array)
    for mask_xml in mask_xml_list:
        masking_array = file_utilities.load_mantid_mask(square_det_size, mask_xml, True)
        mask_array_list.append(masking_array)

    if mask_operation == 'INFO':
        # do statistic and plot mask/ROI in 2D
        for mask_array in mask_array_list:
            show_info(mask_array)
    else:
        if mask_operation == 'AND':
            result_mask_array = binary_operation_and(mask_array_list)
        elif mask_operation == 'OR':
            result_mask_array = binary_operation_or(mask_array_list)

        if out_h5_name is not None:
            export_masking_array(out_h5_name)

        show_info(result_mask_array)
    # END-IF-ELSE

    return


if __name__ == '__main__':
    main(sys.argv)
