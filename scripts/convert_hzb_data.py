# TODO - TONIGHT 0 - CONTINUE FROM HERE!
# Convert HZB data to standard transformed-HDF5 format considering all the sample log values and instrument information
# This may be executed only once and moved out of build
from pyrs.utilities import file_util
from pyrs.utilities import rs_project_file


def parse_hzb_tiff(tiff_file_name):
    """
    Parse HZB TIFF (image) data to numpy array (1D)
    :param tiff_file_name:
    :return: (1) 1D array (column major, from lower left corner) (2) (num_row, num_cols)
    """
    # is it rotated?
    # rotated
    counts_matrix = file_util.load_rgb_tif(raw_tiff_name=tiff_file_name, pixel_size=None, rotate=True)

    counts_array = counts_matrix.flatten()

    return counts_array, counts_matrix.shape


def import_hzb_summary():
    """
    import and parse HZB summary file in EXCEL format
    :return:
    """
    """
    Index([    u'E3 file',        u'Tiff',  u'Tiff_Index',       u'Index',
               u'2th',         u'mon',       u'2th.1',  u'Unnamed: 7',
                u'L2',        u'ADET', u'Unnamed: 10', u'Unnamed: 11',
       u'Unnamed: 12', u'Unnamed: 13',        u'SDET'],
      dtype='object')

    """

    # convert all the entry (column name) to dictionary of numpy arrays
    # aaa = numpy.array(a)

    # ['Tiff'][i] = E3-Y2O3 42-50
    # ['Tiff Index] = 1, 2, 3, ...
    # 2th = 2th.1
    # L2: unit == mm

    # TIF name:
    # tif_name = '{}_{:05}.tiff'.format(df['Tiff'][0], int(df['Tiff_Index'][0]))

    return


def main(argv):
    """
    main for the workflow to create the HDF5
    :param argv:
    :return:
    """
    # process inputs ...

    # parse EXCEL spread sheet to ...

    # start project file
    project_file = rs_project_file.HydraProjectFile()

    # add scan data
    for file_index in blabla:
        project_file.add_scan()

    # save
    project_file.save()

    return


if __name__ == '__main__':
    main()
