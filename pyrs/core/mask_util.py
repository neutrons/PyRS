# Zoo of methods to work with masks including Mantid and PyRS special


def load_mantid_mask(pixel_number, mantid_mask_xml, is_mask):
    """ Load Mantid mask file in XML format
    Assumption: PixelID (detector ID) starts from 0 and there is NO gap
    :param mantid_mask_xml:
    :param pixel_number: total pixel number
    :return: a vector
    """
    checkdatatypes.check_file_name(mantid_mask_xml, True, False, False, 'Mantid XML mask file')
    checkdatatypes.check_int_variable('(Total) pixel number', pixel_number, (1024**2, 2048**2+1))

    # load file to lines
    mask_file = open(mantid_mask_xml, 'r')
    mask_lines = mask_file.readlines()
    mask_file.close()

    # get detector ID range line
    det_id_line = None
    for line in mask_lines:
        if line.count('<detid') > 0:
            det_id_line = line.strip()
            break
    # END-FOR

    if det_id_line is None:
        raise RuntimeError('Mask file {} does not have masked detector IDs'.format(mantid_mask_xml))

    # parse
    masked_det_pair_list = det_id_line.split('>')[1].split('<')[0].strip().split(',')
    # print ('[DB...BAT] Masked detectors range: {}'.format(masked_det_pair_list))

    # create vector with 1 (for not masking)
    masking_array = np.zeros((pixel_number,), 'float')
    if is_mask:
        # is given string are mask then default is not masked
        masking_array += 1.
    # is ROI default = 0

    masked_specs = 0
    for masked_det_pair in masked_det_pair_list:
        # get range
        terms = masked_det_pair.split('-')
        start_detid = int(terms[0])
        end_detid = int(terms[1])
        # check range
        if end_detid >= pixel_number:
            raise RuntimeError('Detector ID {} is out of range of given detector size {}'
                               ''.format(end_detid, pixel_number))
        # mask or ROI
        if is_mask:
            masking_array[start_detid:end_detid+1] = 0.
        else:
            masking_array[start_detid:end_detid+1] = 1.
        # stat
        masked_specs += end_detid - start_detid + 1
    # END-FOR

    print ('[DB...CHECK] Masked spectra = {}, Sum of masking array = {}'
           ''.format(masked_specs, sum(masking_array)))

    return masking_array
