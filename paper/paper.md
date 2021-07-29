---
title: 'pyRS: A Python package for the reduction and analysis of neutron data measured at the High Intensity Diffractometer for Residual Stress Analysis at the High Flux Isotope Reactor'
tags:
  - Python
  - Mechanics of Materials
  - Neutron Diffraction
  - Residual Stress
authors:
  - name: Chris M. Fancher # note this makes a footnote saying 'co-first author'
    orcid: 0000-0002-3952-5168
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Peter F. Peterson # note this makes a footnote saying 'co-first author'
    affiliation: "2, 3"
  - name: Jean Bilheux
    affiliation: 2
  - name: Wenduo Zhou # note this makes a footnote saying 'co-first author'
    affiliation: "2, 3"
  - name: Ross E. Whitfield # note this makes a footnote saying 'co-first author'
    affiliation: "2, 3"
  - name: Jose Borreguero # note this makes a footnote saying 'co-first author'
    affiliation: "2, 3"
  - name: William Godoy # note this makes a footnote saying 'co-first author'
    affiliation: "2, 3"
  - name: Steven Hahn # note this makes a footnote saying 'co-first author'
    affiliation: "2, 3"
affiliations:
 - name: Materials Science & Technology Division, Oak Ridge National Laboratory, Oak Ridge, TN
   index: 1
 - name: Neutron Scattering Division, Oak Ridge National Laboratory, Oak Ridge, TN
   index: 2
 - name: Computer Science and Mathematics Division, Oak Ridge National Laboratory, Oak Ridge, TN
   index: 3
date: 28 July 2021
bibliography: paper.bib
---

# Summary
The python Residual Stress analysis (pyRS) software package incorporates components for detector calibration, reduction of measured 2D data into intensity vs. scattering angle, peak fitting, and residual stress/strain analysis.
pyRS is designed as a flexible software package that incorporates separate modules for 2D reduction, peak analysis, and stress analysis.
These modules provide a streamlined workflow for reducing raw neutron events into 1D intensity vs. scattering angle and subsequent analysis to extract the interatomic spacing and intensity for residual stress and texture analysis.
pyRS saves data into a single hdf5 file to streamline data storage by storing metadata, reduced diffraction data, and peak analysis results within a single hdf5 file (named HIDRA project file) and are passed between different modules.


# Statement of need

The former 2nd Generation Neutron Residual Stress Facility (NRSF2) located at the High Flux Isotope Reactor (HFIR) at Oak Ridge National Laboratory (ORNL) recently underwent an extensive upgrade with a modernization of the detector and control software. The new High-Intensity Diffractometer for Residual stress Analysis (HIDRA) instrument is designed with the flexibility to handle the needs of a diverse user community. The upgrade required the development of a new data reduction and analysis workflow with a streamlined user experience. The python Residual Stress Analysis (pyRS) software package incorporates components for detector calibration, reduction of measured 2D data into intensity vs. scattering angle, peak fitting, and residual stress/strain analysis.


`Gala` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike.

# pyRS
Users interact with pyRS through either a GUI or custom python scripts.
pyRS was designed with three distinct GUIs that enables users to 1) reduce neutron event data, 2) perform single-peak fitting of reduced data, and 3) combine single-peak fitting results for residual stress analysis and subsequent visualization. 

\begin{equation}\label{eq:d}
d_{hkl}(x,y,z) = \frac{\lambda}{2sin\theta(x,y,z)}
\end{equation}

\begin{equation}\label{eq:strain}
\varepsilon_{ii}(x,y,z)=\frac{d^{ii}_{hkl}(x,y,z)}{d^0} - 1
\end{equation}

\begin{equation}\label{eq:stress}
\sigma_{ii}=\frac{E}{\left ( 1 + \nu \right )}\left [ \varepsilon_{ii} + \frac{\nu}{1-2\nu} \left ( \varepsilon_{11} + \varepsilon_{22} + \varepsilon_{33} \right )\right ]
\end{equation}

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Acknowledgements

The authors are grateful for the contributions of E. Andrew Payzant, Garrett Granroth, and Anibal Ramirez Cuesta for reviewing the paper and William Godoy, Steven Hahn, Fahima Islam, and Mathieu Doucet for their contributions to the software. A portion of this research used resources at the High Flux Isotope Reactor, a DOE Office of Science User Facility operated by the Oak Ridge National Laboratory.

# References
