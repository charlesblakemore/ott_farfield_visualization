
Python-based Visualization of Field Computations from the MATLAB library "Optical Tweezers Toolbox"
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

`Optical Tweezers Toolbox <https://www.mathworks.com/matlabcentral/fileexchange/73541-ott-optical-tweezers-toolbox>`_ - Lenton, Isaac C. D., et al.

As of now, the code is relatively simple and serves more as an 
example wrapper of the MATLAB code. It only handles input TEM00 
modes with linear polarization, but the modularization should
allow this functionality to be expanded once the structure is 
understood by the user.

There was an idea at the outset of this work to also include a 
simple ray-tracing scheme to propagate the light from the trap
itself through a typical output imaging system. A single example
exists that mimics the imaging system in place in on the optical
traps at Stanford. Note that this analysis makes a number of
assumptions inherent to ray-tracing (paraxial, thin lens, etc.)
which should be taken into account when interpreting results.

Also, it should be noted that the user doesn't actually have to 
plot anything. They can instead use the basic MATLAB wrapper 
functions and accompanying data processing routines to load the
farfield solution of the scattered electric field as viewed on 
a sphere "infinitely" far away from the trapped dielectric 
particle

The example script ``scripts/generate_farfield_scattering.py`` 
first runs the simulation, loads the data into python structures,
then uses plotting routines. These are three separate activities
that can be performed indepentely. For example, a particular
scattering configuration can be simulated and saved, and then
subsequent plotting (maybe with different ray-tracing/imaging
schemes) can be done using the saved dataset. The user may also
take the datasets themselves and do what they will with them.


Install
-------

Non-Python prerequisites
````````````````````````

Users will need to `install the MATLAB engine for Python <https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html>`_ specific 
to their personal work environment. This step can often manifest
some package compatibility issues as the MATLAB version you have 
installed requires very specific versions of python and the 
matlabengine package in order to function. Thus "matlabengine" is
not in the requirements file.

This version of the software has been tested with MATLAB R2021b,
Python 3.9.16, and matlabengine=9.11.19.

Eventually, some functions (implemented but not yet working) will 
make use of ffmpeg to make movies, although maybe a purely 
pythonic solution will be better.


Python packages
````````````````

Use pip to install packages::

   pip install --upgrade pip
   pip install -r ./ott_visualization/requirements.txt


From sources
````````````

To install system-wide, noting the path to the src since no wheels
exist on PyPI, as well as the ``-e`` flag so one can edit the code 
and have the import calls to reflect those changes, install in 
developer mode::

   pip install -e ott_visualization

If you don't want a global installation (i.e. if multiple users will
engage with and/or edit this library) and you don't want to use venv
or some equivalent::

   pip install -e ott_visualization --user

where pip is pip3 for Python3 (tested on Python 3.9.16). Be careful 
NOT to use ``sudo``, as the latter two installations make a file
``easy-install.pth`` in either the global or the local directory
``lib/python3.X/site-packages/easy-install.pth``, and sudo will
mess up the permissions of this file such that uninstalling is very
complicated.


Uninstall
---------

If installed without ``sudo`` as instructed, uninstalling should be 
as easy as::

   pip uninstall ott_visualization

If installed using ``sudo`` and with the ``-e`` and ``--user`` flags, 
the above uninstall will encounter an error.

Navigate to the file ``lib/python3.X/site-packages/easy-install.pth``, 
located either at  ``/usr/local/`` or ``~/.local`` and ensure there
is no entry for ``opt_lev_analysis``.


License
-------

The package is distributed under an open license (see LICENSE file for
information).


Authors
-------

Charles Blakemore (chas.blakemore@gmail.com)