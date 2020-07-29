GridInformation CSV Output:
===========================

| **ID** | **Complexity** |  **Est.** |  **Author** |
| ------ | ---------------|-----------|-------------|
| GRID-CSV-OUT | Medium | TBD | Jeff Bunn / Chris Fancher |
                                       

As a neutron beamline user, I want to output my data into a readily readable .csv format
after a strain grid has been established in the strain-stress GUI. This datafile
will retain the provenance (if multiple subruns share the same vx, vy, vz coordinates)
and pedigree (which project file a specific subrun is contained within) of the data.

Acceptance Criteria:
--------------------

After I have established a strain/stress grid in the GUI, I want to be able to output
this data into a easily transferrable format, henceforth referred to as GridInformation file.
A button in the GUI will say "Export Grid Information." Also a toggle switch will appear next
to the button with the option of either a "summary" or a "detailed." By default,
the "summary" option will be selected. The "detailed" option will only be available for selection
if any of the directions contain more than one project file selection. If only one file
is selected for each direction then only a summary file can be exported. When I press
the "Export Grid Information" I will be presented with a dialog box and a file save option
will be given. The user will define the filename for the GridInformation file. The summary
file will show the grid information from the current selected peak (defined in GUI above)
and will contain spatial information on stress and strain. A specified format is detailed
below for both the detailed and summary case.

Description, Additional Detail, Context
---------------------------------------

GridInformation Summary Output Template File: HB2B_StressStrain_peak0_Summary.csv  
GridInformation Detailed Output Template File: HB2B_StressStrain_peak0_Detailed.csv
