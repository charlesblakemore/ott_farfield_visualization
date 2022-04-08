import sys, os, time

from tqdm import tqdm

import numpy as np
import matlab.engine

import farfield_plotting
farfield_plotting.base_plotting_directory = '/home/cblakemore/plots/ott_farfield'



start = time.time()

base_data_path = '../raw_data/ztrans_movie_transmitted_test/'

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
            'Nmax': 100
}

param_to_sweep = 'zOffset'
# param_array = np.linspace(0.0, -100.0, 101)
param_array = np.linspace(0.0, -100.0, 2)
param_scale = 1e-6
save_suffix = '_um'
# save_suffix = ''




beam = 'tot'
transmitted = True
rmax = 0.05
save_fig = False
show_fig = True

# max_radiance_val = 8.4
max_radiance_val = 8.4**2
manual_phase_plot_lims = ()



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

param_ind = 0
for param_ind in tqdm(range(len(param_array))):

    param_val = param_array[param_ind]
    param_str = f'{param_to_sweep}_{int(param_val):d}{save_suffix}'
    sys.stdout.flush()
    param_ind += 1

    simulation_parameters['datapath'] \
        = os.path.join(base_data_path, param_str)
    simulation_parameters[param_to_sweep] \
        = float(param_scale*param_val)

    arglist = [[key, simulation_parameters[key]] \
                for key in simulation_parameters.keys()]
    engine = matlab.engine.start_matlab()
    matlab_datapath \
        = engine.compute_far_field(\
            *[arg for argtup in arglist for arg in argtup], \
            nargout=1, background=False)

    theta_grid, r_grid, efield \
        = farfield_plotting.load_data(matlab_datapath, beam=beam, \
                                      transmitted=transmitted)


    ms_position = [simulation_parameters['xOffset'], \
                   simulation_parameters['yOffset'], \
                   simulation_parameters['zOffset']]

    farfield_plotting.plot_2D_farfield(
        theta_grid, r_grid, efield, figname=f'test/frame_{param_ind:04d}.png', \
        ms_position=ms_position, rmax=rmax, title=title, \
        polarisation=simulation_parameters['polarisation'], \
        save=save_fig, show=show_fig)

stop = time.time()

print(f'Total sim time: {stop-start:0.2f}')
print()