import sys, os

from tqdm import tqdm

import numpy as np
import matlab.engine

import ott_plotting


ott_plotting.update_base_plotting_directory(\
        '/home/cblakemore/plots/ott_axial_forces')


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
                    'radius': 0.798e-6, \
                       'rho': 1550.0, \
                'n_particle': 1.33, \
                  'n_medium': 1.00, \
                'wavelength': 1064.0e-9, \
                        'NA': 0.9, \
                   'xOffset': 0.0e-6, \
                   'yOffset': 0.0e-6, \
                     'zSpan': 15.0e-6, \
                        'nz': 501, \
                      'Nmax': 200, \
              'polarisation': 'X', \
                'resimulate': True
}



# plot_parameters = {
#                       'beam': 'tot', \
#                       'rmax': 0.01, \
#                       'save': False, \
#                       'show': True, \
#                    'plot_2D': True, \
#                    'plot_3D': False, \
#                  'view_elev': -45.0, \
#                  'view_azim': 20.0, \
#         'max_radiance_trans': 10.0, \
#          'max_radiance_refl': 0.16, \
#               'unwrap_phase': True, \
#     'manual_phase_plot_lims': (-3.0*np.pi, 2.0*np.pi), \
#             'label_position': True, \
# }



# plot_parameters['fig_id'] = 'tests/max_val_test'

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
    = engine.compute_axial_force(\
        *[arg for argtup in arglist for arg in argtup], \
        nargout=1, background=False)

### Load the data that MATLAB computed and saved, handling the 
### transmitted and reflected cases separately since they may 
### propagate through distinct optical systems
zaxis, trapping_efficiency, levitation_power\
    = ott_plotting.load_force_data(\
            matlab_datapath)

### Plot everything!
ott_plotting.plot_axial_force(\
    zaxis, trapping_efficiency, levitation_power)