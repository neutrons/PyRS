import h5py


def parse_gold_file(file_name):
    """

    Parameters
    ----------
    file_name

    Returns
    -------
    ~dict or ~numpy.ndarray
        gold data in array or dictionary of arrays

    """
    # Init output
    data_dict = dict()

    # Parse file
    gold_file = h5py.File(file_name, 'r')
    data_set_names = list(gold_file.keys())
    for name in data_set_names:
        if isinstance(gold_file[name], h5py.Dataset):
            data_dict[name] = gold_file[name].value
        else:
            data_dict[name] = gold_file[name]['x'].value, gold_file[name]['y'].value
    # END-FOR

    if len(data_dict) == 1 and data_dict.keys()[0] == 'data':
        # only 1 array with default name
        return data_dict['data']

    return data_dict
