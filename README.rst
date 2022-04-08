
Python-based Visualization of Far-Field Computations from the MATLAB library "Optical Tweezers Toolbox"
===============================================================

Personally, I find dealing with MATLAB directly to be a pain
and I'm much more comfortable producing visualizations with 
Python. This collection of software is meant to make it easy to 
produce high-quality visualisations of the optical field that 
results from typical Mie scattering problems.

Introduction
------------

The physical description of the incident optical field and the 
subsequent scattering are all performed by MATLAB, making use of
the T-matrix formalism. See the link below for a complete 
description of the software together with accompanying literature.

`Optical Tweezers Toolbox <https://www.mathworks.com/matlabcentral/fileexchange/73541-ott-optical-tweezers-toolbox>`_ - Lenton, Isaac C. D., et al., 

Install
-------

NOT SETUP TO BE INSTALLABLE WITH PIP OR ANYTHING LIKE THAT AT THIS POINT

Users will need to `install the MATLAB engine for Python <https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html>`_ specific 
to their personal work environment, as well as the ``unwrap`` package for 
two-dimensional phase unwrapping. Other than that, standard NumPy and 
Scipy installations should cover everything you neeed. The explicit
requirements will be enumerated soon.


License
-------

The package is distributed under an open license (see LICENSE file for
information).


Authors
-------

Charles Blakemore (chas.blakemore@gmail.com)