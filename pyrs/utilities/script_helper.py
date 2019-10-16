# Methods general to scripts are in this module
from pyrs.utilities import checkdatatypes
import getopt


def convert_opt_operations(opt_opts):
    """
    Convert a 7-tuple list (long-name, short-name, target name, type, default value, is mandatory, document)
    :param opt_opts: list of operation
    :return: (1) operation_dict  key as either long name --xxx, or short name -x),
                                 value = parameter name, value type  (removed: default value, mandatory)
             (2) list of mandatory parameters
             (3) optional default value dictionary: key = parameter name, value = default default value
             (4) dictionary for helping message: key = parameter name, value = short name, long name, document
    """

    checkdatatypes.check_type('Command options', opt_opts, list)

    # split a 4 tuple to 2 3-tuples
    opt_dict = dict()
    man_param_list = list()
    default_param_dict = dict()
    info_dict = dict()

    for t4 in opt_opts:
        if len(t4) != 7:
            raise RuntimeError('Item {} is not defined properly.  7 and only 7 items are allowed')
        long_name_i, short_name_i, target_name_i, type_i, default_i, mandatory_i, doc_i = t4

        # long/full opt name
        if long_name_i is not None:
            if len(long_name_i) < 2:
                raise RuntimeError('Long name {} in {} is not allowed. At least 2 letters'
                                   ''.format(long_name_i, t4))
            opt_dict['--{}'.format(long_name_i)] = target_name_i, type_i  # , default_i, mandatory_i

        # short opt name
        if short_name_i is not None:
            if len(short_name_i) != 1:
                raise RuntimeError('Short name {} in {} is not allowed. 1 and only 1 letter'
                                   ''.format(short_name_i, t4))
            opt_dict['-{}'.format(short_name_i)] = target_name_i, type_i  # , default_i, mandatory_i

        # Mandatory
        if mandatory_i:
            man_param_list.append(target_name_i)
        else:
            default_param_dict[target_name_i] = default_i

        # Information
        info_dict[target_name_i] = short_name_i, long_name_i, doc_i, mandatory_i, default_i

    # END-FOR

    return opt_dict, man_param_list, default_param_dict, info_dict


def parse_arguments(argv, opt_operation_list):
    """ Parse arguments to (opt, args) format
    :param argv: sys.argv
    :param opt_operation_list: list of user-defined opt operation:
                (long-name, short-name, target name, type, default value, is mandatory, document)
    :return:
    """
    short_list_string = ''
    long_list = list()
    for t7 in opt_operation_list:
        long_name = t7[0]
        short_name = t7[1]
        if short_name is not None:
            short_list_string += '{}:'.format(short_name)
        if long_name is not None:
            long_list.append('{}='.format(long_name))
    # END-FOR

    # add help
    long_list.append('help')
    short_list_string += 'h'

    print('[DB...BAT] Short name list: {}; Long name list: {}'.format(short_list_string, long_list))
    # Example: "hdi:o:l:g:G:r:R:"

    try:
        opts, args = getopt.getopt(argv, short_list_string, long_list)
    except getopt.GetoptError as get_err:
        raise RuntimeError('Unable to get-opt from input arguments ({}) due to {}'
                           ''.format(argv, get_err))

    return opts, args


def process_arguments(argv_list,  opt_operation_list):
    """ Process arguments including
    (1) converting user inputs to a dictionary
    (2) throwing exception if any mandatory argument is not given
    (3) setting default values to optional arguments
    (4) print out helping message
    This is the method serving as main entry point for script's main method
    :param argv_list:
    :param opt_operation_list:
    :return:  argument value dictionary or None (help)
    """
    # Check inputs
    opt_operate_dict, mandatory_param_list, optional_param_default_dict, info_dict = \
        convert_opt_operations(opt_operation_list)

    # Parse inputs to args and opts: script name (argv[0]) MUST NOT be in the arg list sent to parse_commands
    command_opts, command_args = parse_arguments(argv_list[1:], opt_operation_list)

    # user does not have any valid inputs
    if len(command_opts) == 0:
        print('Run "{0} -h" or {0} --help" for help'.format(command_args[0]))
        return None

    # Read each input
    arguments_dict = dict()
    for opt_i, arg_i in command_opts:
        # check supported or not
        if opt_i == '--help' or opt_i == '-h':
            arguments_dict['help'] = True
        elif opt_i not in opt_operate_dict:
            # None supported
            print('[WARNING] User input argument {} is not supported. Supported keys are {}'
                  ''.format(opt_i, opt_operate_dict.keys()))
        else:
            # parse
            param_name_i, type_i = opt_operate_dict[opt_i]
            arguments_dict[param_name_i] = type_i(arg_i)
    # END-

    # Check helper
    if 'help' in arguments_dict:
        print_helper(info_dict)
        return None

    # Check mandatory
    err_msg = ''
    for mandatory_i in mandatory_param_list:
        if mandatory_i not in arguments_dict:
            err_msg += 'Argument {} with option {} is mandatory and must be given\n' \
                       ''.format(mandatory_i, info_dict[mandatory_i])
    # END-FOR
    if err_msg != '':
        raise RuntimeError('Missing mandatory arguments:\n{}'.format(err_msg))

    # Set up default
    for optional_i in optional_param_default_dict:
        # check whether been specified or not
        if optional_i not in arguments_dict:
            arguments_dict[optional_i] = optional_param_default_dict[optional_i]
    # END-FOR

    return arguments_dict


def print_helper(arg_info_dict):
    """
    Print helping information
    :param arg_info_dict: dictionary (key = parameter name, value = (-a, --abcd, doc, mandatory, default)
    :return: None
    """
    checkdatatypes.check_dict('Script argument information', arg_info_dict)

    help_str = ''
    for arg_name_i in sorted(arg_info_dict.keys()):
        short_opt, long_opt, doc, mandatory, default = arg_info_dict[arg_name_i]

        help_str += '{}, {}: {}. '.format(short_opt, long_opt, doc)
        if mandatory:
            help_str += 'This is mandatory\n'
        else:
            help_str += 'This is optional with default value {}\n'.format(default)
    # END-FOR

    # print on screen
    print(help_str)

    return help_str


def test_main():
    """
    Test main
    :return:
    """
    import sys
    mock_opt_operations = [('input', 'i', 'inputfile', str, None, True, 'Input HIDRA project file'),
                           ('masks', 'm', 'masksfiles', str, None, False,
                            'Path to an ASCI file containing list of path to mask files, '
                            'separated by ":", ", " or "\n"'),
                           ('instrument', None, 'instrument', str, None, False, 'Path to instrument file'),
                           ('output', 'o', 'outputfile', str, None, True, 'Output calibration in JSON format'),
                           ('binsize', 'b', 'binsize', float, 0.01, False, '2theta step')]

    process_arguments(sys.argv, mock_opt_operations)

    return


if __name__ == '__main__':
    test_main()
