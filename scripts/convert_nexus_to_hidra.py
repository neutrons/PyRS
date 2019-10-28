#!/usr/bin/python
import os
from pyrs.core.nexus_conversion import NeXusConvertingApp


def main(options):
    converter = NeXusConvertingApp(options.nexus)
    converter.convert()

    # generate output filename from input nexus file name
    projectfile = os.path.basename(options.nexus).split('.')[0] + '.h5'
    projectfile = os.path.join(options.outputdir, projectfile)

    converter.save(projectfile)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Convert HB2B data to Hidra Project File')
    parser.add_argument('nexus', help='name of nexus file')
    parser.add_argument('outputdir', help='Path to output directory')

    options = parser.parse_args()

    main(options)
