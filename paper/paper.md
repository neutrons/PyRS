---
title: 'pyRS: A Python package for the reduction and analysis of neutron residual stress data'
tags:
  - Python
  - Mechanics of Materials
  - Neutron Diffraction
  - Residual Stress
authors:
  - name: Chris M. Fancher # note this makes a footnote saying 'co-first author'
    orcid: 0000-0002-3952-5168
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Jean Bilheux
    orcid: 0000-0003-2172-6487
    affiliation: 2
  - name: Wenduo Zhou # note this makes a footnote saying 'co-first author'
    orcid: 0000-0002-2848-5324
    affiliation: "3"
  - name: Ross E. Whitfield # note this makes a footnote saying 'co-first author'
    orcid: 0000-0002-9852-1044
    affiliation: "3"
  - name: Jose Borreguero # note this makes a footnote saying 'co-first author'
    orcid: 0000-0002-0866-8158
    affiliation: "3"
  - name: William F. Godoy # note this makes a footnote saying 'co-first author'
    orcid: 0000-0002-2590-5178
    affiliation: "3"
  - name: Steven Hahn # note this makes a footnote saying 'co-first author'
    orcid: 0000-0002-2018-7904
    affiliation: "3"
  - name: Peter F. Peterson # note this makes a footnote saying 'co-first author'
    orcid: 0000-0002-1353-0348
    affiliation: "3"
affiliations:
 - name: Materials Science & Technology Division, Oak Ridge National Laboratory, Oak Ridge, TN
   index: 1
 - name: Neutron Scattering Division, Oak Ridge National Laboratory, Oak Ridge, TN
   index: 2
 - name: Computer Science and Mathematics Division, Oak Ridge National Laboratory, Oak Ridge, TN
   index: 3
date: 01 November 2021
bibliography: paper.bib
---

# Statement of need

The 2nd Generation Neutron Residual Stress Facility (NRSF2) residual stress mapping instrument at the High Flux Isotope Reactor (HFIR) at ORNL was recently rebuilt with a modern detector and control system. Upgrading from a LabVIEW-based control software (SPICE) to an Experimental Physics and Industrial Control System (EPICS) based control software with the neutron Event Distributor (nED) data acquisition system brought additional experimental flexibility.[@White2019; @Vodopivec2017] The transition from a control system that measured discrete data to an event-based data structure.[@PETERSON201524] deprecated the data reduction and analysis software. The design of `pyRS` relied on years of experience with the previous reduction and analysis software (NRSF View) to ensure the new software builds upon the strengths and improves on the weakness of NRSF View. The lack of a unified analysis and visualization of residual stress data was among the most significant needs.  

# Summary
`pyRS` is a python software package that was designed to meet the data reduction and analysis needs of the neutron residual stress mapping user community at Oak Ridge National Laboratory (ORNL). `pyRS` incorporates separate modules that provide a streamlined workflow for reducing raw neutron events into 1D intensity vs. scattering angle and subsequent analysis to extract the interatomic spacing and intensity for residual stress and texture analysis. Users can access the modules through either a graphical or command-line interface. `pyRS` saves data into a single HDF5 file.[@hdf5], in which the metadata, reduced diffraction data, and peak analysis results are passed between different modules.

# Overview of pyRS
`pyRS` was designed with three distinct graphical user interfaces (GUIs) that enable users to 1) reduce neutron event data, 2) perform single-peak fitting of reduced data, and 3) combine single-peak fitting results for residual stress analysis and subsequent visualization. Figure \ref{fig:data_flow} provides an overview for how data flow through and where users interact with `pyRS`.

<p style="text-align: center;">

![Overview of how \texttt{pyRS} takes in raw neutron data ($\color{red}{red}$) and user inputs ($\color{blue}{blue}$) into the Data Reduction and Data Analysis components. The Data Reduction creates a HIDRA Project File that is then appended with analysis results. Note that the user can specify the inputs through a graphical or python scripting interface.\label{fig:data_flow}](Data_Flow-01.png)

</p>

* Data Reduction
  * *Filter Events and Logs*: The High-Intensity Diffractometer for Residual Stress Analysis (HIDRA) stores raw measured neutron events in HDF5 files using an event data structure using the NeXus standard schema.[@Konnecke2015] The event NeXus data structure stores information about the pixel position and detection time with respect to the start of a measurement. HIDRA leverages this flexibility to encode scan_index metadata signals that `pyRS` uses to filter events into separate datasets. `pyRS` reduces measured events based on how scan_index increments throughout a single NeXus file. `pyRS` reconstructs the measured 2-dimensional diffraction datasets by first filtering the event index array based on the scan_index time, then histogramed based on pixel position (np.histogram).[@Harris2020] Metadata events are filtered using the Mantid framework time-filtering algorithm.[@Arnold2014a]. Users can specify to exclude unwanted logs.
  * *Integrate 2D into 1D data*: Calibration information about the position of the detector in space (XYZ shifts and rotations about the engineering position) are used to determine the angular position of the detector pixels. Pixel angular position and intensity data are histogramed to construct raw Intensity vs. scattering angle datasets based on the default or user-defined angular range.
  * *Normalize by Vanadium (optional)*: raw Intensity vs. scattering angle is normalized by the incoherent scattering intensity from a Vanadium sample if a Vanadium run number is defined.
  * A HIDRA project file is created that stores the calibration information, Intensity vs. scattering data, and metadata logs
* Data analysis
  * *Peak Fitting Analysis*
    * Reduced 1D data are analyzed using single-peak fitting to extract information about the position, intensity, full-width half maximum of N peaks within the detector field of view. Users can define specific peak fitting ranges using the graphical interface or using a JSON formatted text file. Users can export the graphically select peak ranges into a JSON file for later use. Peak fitting results are automatically appended into the loaded HIDRA project file. Alternatively, users can export a CSV summary of the results.
  * *Residual Stress Analysis*
    * Residual stress analysis requires peak fitting results for 2 or 3 orthogonal directions. `pyRS` does not limit users to only defining a single HIDRA project file per direction. `pyRS` can merge multiple project files based on the spatial position metadata logs. `pyRS` determines residual stresses by:
    \begin{equation}\label{eq:stress}
    \sigma_{ii}=\frac{E}{\left ( 1 + \nu \right )}\left [ \varepsilon_{ii} + \frac{\nu}{1-2\nu} \left ( \varepsilon_{11} + \varepsilon_{22} + \varepsilon_{33} \right )\right ]
    \end{equation}

    where

    \begin{equation}\label{eq:strain}
    \varepsilon_{ii}(x,y,z)=\frac{d^{ii}_{hkl}(x,y,z)}{d^0} - 1
    \end{equation}

    and

    \begin{equation}\label{eq:d}
    d_{hkl}(x,y,z) = \frac{\lambda}{2sin\theta(x,y,z)}
    \end{equation}

# Acknowledgements

The authors are grateful for the contributions of E. Andrew Payzant, Garrett Granroth, Jeff R. Bunn and Anibal Ramirez Cuesta for reviewing the paper. A portion of this research used resources at the High Flux Isotope Reactor, a DOE Office of Science User Facility operated by the Oak Ridge National Laboratory. Work at Oak Ridge National Laboratory was sponsored by the Division of Scientific User Facilities, Office of Basic Energy Sciences, US Department of Energy, under Contract no. DE-AC05-00OR22725 with UT-Battelle, LLC. The United States Government retains and the publisher, by accepting the article for publication, acknowledges that the United States Government retains a non-exclusive, paid-up, irrevocable, world-wide license to publish or reproduce the published form of this manuscript, or allow others to do so, for United States Government purposes.  The Department of Energy will provide public access to these results of federally sponsored research in accordance with the DOE Public Access Plan (<http://energy.gov/downloads/doe-public-access-plan>).

# References
