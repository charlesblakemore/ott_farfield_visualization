import os, math, sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import unwrap

import scipy.constants as constants

from ott_plotting_util import global_dict
import ott_plotting_util as util

plt.rcParams.update({'font.size': 14})





def load_force_data(path):
    '''
    Loads data saved by the 'compute_axial_force.m' MATLAB function, (and 
    maybe the radial equivalent in the future) which saves a standard set 
    of data with conventional names to a single directory. This function 
    looks at that directory and tries to load data according to a default 
    assumed format
    '''

    ### These datafiles should be sampled along a single axis (presumably 
    ### the z-axis) for this to work properly
    data_pts = np.loadtxt(\
        os.path.join(path, 'axial_force_points.txt'), \
        delimiter=',')

    trapping_efficiency = np.loadtxt(\
        os.path.join(path, 'axial_force_efficiency.txt'), \
        delimiter=',')
    levitation_power = np.loadtxt(\
        os.path.join(path, 'axial_force_levitation_power.txt'), \
        delimiter=',')

    zaxis = data_pts[2]

    return zaxis, trapping_efficiency, levitation_power






def plot_axial_force(zaxis, trapping_efficiency, levitation_power):

	fig1, ax1 = plt.subplots(1,1,figsize=(7,5))

	ax1.plot(zaxis*1e6, trapping_efficiency, lw=4)
	ax1.set_xlabel('Axial Position [um]')
	ax1.set_ylabel('Trapping Efficiency [N/W]')

	fig1.tight_layout()



	fig2, ax2 = plt.subplots(1,1,figsize=(7,5))

	ax2.plot(zaxis*1e6, levitation_power, lw=4)
	ax2.set_xlabel('Axial Position [um]')
	ax2.set_ylabel('Levitation Power [W]')

	fig2.tight_layout()

	plt.show()
