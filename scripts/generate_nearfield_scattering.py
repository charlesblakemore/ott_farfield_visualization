import sys, os

from tqdm import tqdm

import numpy as np
import matlab.engine

import ott_plotting


ott_plotting.update_base_plotting_directory(\
        '/home/cblakemore/plots/ott_nearfield')


base_data_path = '../raw_data/'

### Pack up all the simulation parameters into a dictionary with string keys
### that match the expected MATLAB variable names. This is the only reasonably
### compact way of passing keyword arguments via the MATLAB/Python engine API
###
### REMEMBER: Cartesian offsets are in SI base units (meters) and correspond
### to the position of the BEAM relative to the microsphere/scatterer.
###    e.g. zOffset = +10um corresponds to a microsphere BELOW the focus
simulation_parameters = {
                  'datapath': base_data_path, \
                    'radius': 2.5e-6, \
                'n_particle': 1.39, \
                  'n_medium': 1.00, \
                'wavelength': 1064.0e-9, \
                        'NA': 0.095, \
                   'xOffset': 0.0e-6, \
                   'yOffset': 0.0e-6, \
                   'zOffset': 0.0e-6, \
                     'xSpan': 20.0e-6, \
                     'zSpan': 40.0e-6, \
                        'nx': 151, \
                        'nz': 151, \
              'polarisation': 'X', \
                      'Nmax': 200, \
                'resimulate': True
}



plot_parameters = {
                      'beam': 'tot', \
                      'rmax': 0.01, \
                      'save': False, \
                      'show': True, \
                   'plot_2D': True, \
                 'view_elev': -45.0, \
                 'view_azim': 20.0, \
              'max_radiance': 0.0, \
                 'max_field': 0.0, \
              'unwrap_phase': True, \
    # 'manual_phase_plot_lims': (-3.0*np.pi, 2.0*np.pi), \
            'label_position': True, \
}



plot_parameters['fig_id'] = 'tests/near_field_init'
simulation_parameters['zOffset'] = 0.0e-6

##########################################################################
##########################################################################
##########################################################################



### Build the MATLAB formatted argument list from the dictionary
### defined at the top of this script
arglist = [[key, simulation_parameters[key]] \
            for key in simulation_parameters.keys()]

### Start the MATLAB engine and run the computation
engine = matlab.engine.start_matlab()
engine.addpath('../lib', nargout=0)
matlab_datapath \
    = engine.compute_near_field(\
        *[arg for argtup in arglist for arg in argtup], \
        nargout=1, background=False)

### Load the data that MATLAB computed and saved
x_grid, z_grid, efield\
    = ott_plotting.load_nearfield_data(\
            matlab_datapath,\
            beam=plot_parameters['beam'])


### Plot everything!

ott_plotting.plot_2D_nearfield(
    x_grid, z_grid, efield, simulation_parameters, \
    **{**plot_parameters, 'show': False})

ott_plotting.plot_2D_farfield_components(
    x_grid, z_grid, efield, simulation_parameters, 
    **{**plot_parameters, 'show': True})
