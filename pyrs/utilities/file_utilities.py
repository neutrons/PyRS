# Class providing a series of static methods to work with files
import os
import platform
import time
import checkdatatypes


def check_creation_date(file_name):
    """
    check the create date (year, month, date) for a file
    :except RuntimeError: if the file does not exist
    :param file_name: 
    :return: 
    """
    checkdatatypes.check_file_name(file_name, check_exist=True)

    # get the creation date in float (epoch time)
    if platform.system() == 'Windows':
        # windows not tested
        epoch_time = os.path.getctime(file_name)
    else:
        # mac osx/linux
        stat = os.stat(file_name)
        try:
            epoch_time = stat.st_birthtime
        except AttributeError:
            # We're probably on Linux. No easy way to get creation dates here,
            # so we'll settle for when its content was last modified.
            epoch_time = stat.st_mtime
        # END-TRY
    # END-IF-ELSE

    # convert epoch time to a string as YYYY-MM-DD
    file_create_time = time.localtime(epoch_time)
    file_create_time_str = time.strftime('%Y-%m-%d', file_create_time)

    return file_create_time_str


# testing
print (check_creation_date('__init__.py'))
