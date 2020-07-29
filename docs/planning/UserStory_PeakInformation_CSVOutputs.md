PeakInformation CSV Output:
===========================

| **ID** | **Complexity** |  **Est.** |  **Author** |
| ------ | ---------------|-----------|-------------|
| PEAK-CSV-OUT | Medium | TBD | Jeff Bunn / Chris Fancher |
                                       

As a neutron beamline user, I want to be able to output my data into a readily readable .csv format
after I have fit the project file. This file shows all peakcollections in the project file.

Acceptance Criteria:
--------------------

For the first output, henceforth referred to as PeakInformation, The output is generated
by me through pressing of a button in the Peak Fitting GUI. The button is labeled
"Export Peak Information." This button should only be pressable once the user has loaded
a project file, and performed a fit of one or more peak ranges in the project file.
The format for the output will be .csv and a template for the output file is given below.
This PeakInformation file will contain information from only a single, fit, project file.
The name of the PeakInformation file will be automatically generated and I will only give
the output directory for this file.

Description, Additional Detail, Context
---------------------------------------

PeakInformation Template File: HB2B_1320.csv
