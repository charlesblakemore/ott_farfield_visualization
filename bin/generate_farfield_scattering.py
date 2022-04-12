import sys, os

from tqdm import tqdm

import numpy as np
import matlab.engine

import farfield_plotting
farfield_plotting.base_plotting_directory = '/home/cblakemore/plots/ott_farfield'



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
          'radius': 3.76e-6, \
      'n_particle': 1.39, \
        'n_medium': 1.00, \
      'wavelength': 1064.0e-9, \
              'NA': 0.095, \
         'xOffset': 0.0e-6, \
         'yOffset': 0.0e-6, \
         'zOffset': 0.0e-6, \
        'thetaMin': 0.0, \
        'thetaMax': float(np.pi/6.0), \
          'ntheta': 1001, \
            'nphi': 101, \
    'polarisation': 'X', \
            'Nmax': 300
}




beam = 'tot'
transmitted = True
rmax = 0.004
save_fig = False
show_fig = True

plot_2D = True

plot_3D = True
view_elev = -40.0
view_azim = 20.0

max_radiance_val = 0.0
manual_phase_plot_lims = (-np.pi, 3.0*np.pi)


simulation_parameters['zOffset'] = -150.0e-6

##########################################################################
##########################################################################
##########################################################################

if beam == 'inc':
    title = 'Incident Gaussian Beam'
elif beam == 'scat':
    title = 'Scattered Beam'
else:
    title = 'Total Beam'

if transmitted:
    title += ', Transmitted Hemisphere'
else:
    title += ', Back-reflected Hemisphere'

### Update this label for movie frames
ms_position = [simulation_parameters['xOffset'], \
               simulation_parameters['yOffset'], \
               simulation_parameters['zOffset']]


### Build the MATLAB formatted argument list from the dictionary
### defined at the top of this script
arglist = [[key, simulation_parameters[key]] \
            for key in simulation_parameters.keys()]

### Start the MATLAB engine and run the computation
engine = matlab.engine.start_matlab()
matlab_datapath \
    = engine.compute_far_field(\
        *[arg for argtup in arglist for arg in argtup], \
        nargout=1, background=False)

### Load the data that MATLAB computed and saved
theta_grid, r_grid, efield \
    = farfield_plotting.load_data(matlab_datapath, beam=beam, \
                                  transmitted=transmitted)


ray_tracing = farfield_plotting.ray_tracing_propagation(10.0e-3, 1.0) \
                @ farfield_plotting.ray_tracing_thin_lens(10.0e-3) \
                @ farfield_plotting.ray_tracing_propagation(50.0e-3, 1.0) \
                @ farfield_plotting.ray_tracing_thin_lens(40.8e-3) \
                @ farfield_plotting.ray_tracing_propagation(90.8e-3, 1.0) \
                @ farfield_plotting.ray_tracing_thin_lens(50.8e-3) \
                @ farfield_plotting.ray_tracing_propagation(50.8e-3, 1.0) \



# ### Plot everything!
if plot_2D:
    
    if plot_3D and show_fig:
        show_2D_fig = False
    elif show_fig:
        show_2D_fig = True

    farfield_plotting.plot_2D_farfield(
        theta_grid, r_grid, efield, simulation_parameters, \
        ms_position=ms_position, rmax=rmax, title=title, \
        # ray_tracing_matrix=farfield_plotting.get_simple_ray_tracing_matrix(), \
        manual_phase_plot_lims=manual_phase_plot_lims, \
        ray_tracing_matrix=ray_tracing, \
        save=save_fig, show=show_2D_fig)

if plot_3D:
    farfield_plotting.plot_3D_farfield(
        theta_grid, r_grid, efield, simulation_parameters, \
        ms_position=ms_position, rmax=rmax, title=title, \
        # ray_tracing_matrix=farfield_plotting.get_simple_ray_tracing_matrix(), \
        manual_phase_plot_lims=manual_phase_plot_lims, \
        ray_tracing_matrix=ray_tracing, \
        view_elev=view_elev, view_azim=view_azim, \
        save=save_fig, show=show_fig)

