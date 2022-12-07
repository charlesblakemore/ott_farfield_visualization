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





##########################################################################
###########    SELECTING, RE-CASTING, AND NORMALZING DATA    #############
##########################################################################

def load_nearfield_data(path, beam='tot'):
    '''
    Loads data saved by the 'compute_far_field.m' MATLAB function, which 
    saves a standard set of data with conventional names to a single 
    directory. This function looks at that directory and tries to load 
    data according to a default assumed format
    '''

    if beam not in ['inc', 'scat', 'int', 'tot']:
        raise ValueError("Beam type must be of types: "\
                            "'inc', 'scat', 'int', 'tot'")

    ### These datafiles should be sampled on a regular (x, z) grid for 
    ### this to work properly
    data_pts = np.loadtxt(\
        os.path.join(path, f'nearfield_points.txt'), \
        delimiter=',')
    npts = data_pts.shape[1]

    field_real = np.loadtxt(\
        os.path.join(path, f'nearfield_{beam}_real.txt'), \
        delimiter=',')
    field_imag = np.loadtxt(\
        os.path.join(path, f'nearfield_{beam}_imag.txt'), \
        delimiter=',')

    # all_x = data_pts[0]
    # all_z = data_pts[1]

    ### Build meshgrids from the MATLAB-exported data
    x_grid, z_grid = np.meshgrid(data_pts[0], data_pts[1], indexing='ij')

    ### Initialize grid-like arrays for the re-cast electric field
    nx = len(data_pts[0])
    nz = len(data_pts[1])
    print(nx, nz)
    efield_grid = np.zeros((nx, nz, 3), dtype=np.complex128)

    ### Extract the single column data to the gridded format
    for i in range(nx):
        for k in range(nz):
            efield_grid[i,k,:] = field_real[:,k*nx+i] \
                                    + 1.0j * field_imag[:,k*nx+i]

    return x_grid, z_grid, efield_grid








##########################################################################
##########################    2D PLOTTING    #############################
##########################################################################


def plot_2D_nearfield(\
        x_grid, z_grid, efield, simulation_parameters, field_axis=None, \
        title=True, max_radiance=0.0, manual_phase_plot_lims=(), \
        phase_sign=1.0, plot_microsphere=True, label_position=None, \
        show=True, save=False, figname='', fig_id='', beam='', \
        verbose=True, **kwargs):

    if verbose:
        print()
        print('Generating 2D plot...')
        sys.stdout.flush()

    global global_dict
    base_plotting_directory = global_dict['base_plotting_directory']

    polarisation = simulation_parameters['polarisation']
    wavelength = simulation_parameters['wavelength']
    zOffset = simulation_parameters['zOffset']

    if label_position:
        ms_position = [simulation_parameters['xOffset'], \
                       simulation_parameters['yOffset'], \
                       simulation_parameters['zOffset']]
    else:
        ms_position = None

    ### Decide which Efield component to plot
    if field_axis is None:
        if polarisation == 'X':
            field_axis = 0
        elif polarisation == 'Y':
            field_axis = 1
        else:
            raise ValueError('Field axis to plot is unclear')

    radiance = np.abs(efield[:,:,field_axis])**2
    phase = np.angle(efield[:,:,field_axis])

    phase *= phase_sign

    fig, axarr = plt.subplots(1, 2, figsize=(10,10), sharex=True, sharey=True)
    if title:
        title = util._build_title(beam)
        fig.suptitle('Trap Near-Field: ' + title, \
                      fontsize=16, fontweight='bold')

    ### Plot a contour for the microsphere if so desired
    if plot_microsphere:

        for i in range(2):
            microsphere = plt.Sphere(\
                            (0,0), simulation_parameters['radius'], \
                            fc=None, ec='k', ls='--', lw=3, zorder=99)
            axarr[i].add_patch(microsphere)

    ### Plot the radiance and phase of the electric field specifically 
    ### definining the filled contour levels to allow for algorithmic 
    ### construction of the colorbars

    if not max_radiance:
        max_radiance = np.max(radiance)

    rad_levels = np.linspace(0, max_radiance, 501)
    rad_cont = axarr[0].pcolormesh(x_grid, z_grid, radiance, \
                                   vmin=0, vmax=max_radiance, \
                                   cmap='plasma', zorder=3, \
                                   shading='gouraud')

    if len(manual_phase_plot_lims):
        min_phase = np.floor(manual_phase_plot_lims[0]/np.pi)
        max_phase = np.ceil(manual_phase_plot_lims[1]/np.pi)
    else:
        min_phase = np.floor(np.min(phase[inds])/np.pi)
        max_phase = np.ceil(np.max(phase[inds])/np.pi)

    phase_levels = np.linspace(min_phase, max_phase, 501)*np.pi
    phase_cont = axarr[1].contourf(x_grid, z_grid, phase, \
                                   levels=phase_levels, \
                                   cmap='plasma', zorder=3)
    phase_ticks = []
    phase_ticklabels = []
    for i in range(int(max_phase - min_phase)+1):
        phase_val = min_phase + i
        phase_ticks.append(phase_val*np.pi)
        if not phase_val:
            phase_ticklabels.append('0')
        elif phase_val == 1:
            phase_ticklabels.append('$\\pi$')
        elif phase_val == -1:
            phase_ticklabels.append('$-\\pi$')
        else:
            phase_ticklabels.append(f'{int(phase_val):d}$\\pi$')

    ### Clean up the axes and labels
    for i in range(2):
        axarr[i].set_xlim(np.min(x_grid[0,:]), np.max(x_grid[0,:]))
        axarr[i].set_ylim(np.min(z_grid[:,0]), np.max(z_grid[:,0]))
        axarr[i].set_xlabel('Radial Coordinate [$\\mu$m]')

    axarr[0].set_ylabel('Axial Coordinate [$\\mu$m]')

    # ### Add a note with the plotted value of rmax, i.e. the size of the
    # ### circular aperture displayed at the end
    # fig.text(0.5, 0.1, f'{1000*rmax:0.1f} mm radius\naperture', fontsize=16, \
    #           ha='center', va='center')

    ### Add a note with the relative positions of beam and scatterer, noting
    ### that the offset in the filename/simulation is BEAM relative to the
    ### MS at the origin, so that we need to invert the coordinates. I also
    ### want consistent sizing so the plots can be combined into a movie
    if ms_position is not None:
        try:
            iter(ms_position)
        except Exception:
            raise IOError('MS position needs to be iterable')
        assert len(ms_position) == 3, 'MS position should 3 coordinates'
        val_str = 'MS position:\n('
        for var in ms_position:
            if var > 0:
                sign_str = '-'
            else:
                sign_str = ' '
            val_str += sign_str + f'{np.abs(var)*1e6:0.2f}, '
        val_str = val_str[:-2] + ')'

        ms_label = fig.text(0.5, 0.85, f'{val_str} $\\mu$m', \
                             fontsize=12, ha='center', va='center')
        ms_label.set(fontfamily='monospace')

    ### These labels need to have the same vertical extent otherwise they
    ### fuck up with the axis sizing
    axarr[0].set_title('Radiance')
    axarr[1].set_title('Phase')

    ### Do a tight_layout(), but then pull in the sides of the figure a bit
    ### to make room for colorbars
    fig.tight_layout()
    fig.subplots_adjust(left=0.075, right=0.925, top=0.85, bottom=0.05)

    ### Make the colorbars explicitly first by defining and inset axes
    ### and then plotting the colorbar in the new inset
    rad_inset = inset_axes(axarr[0], width="4%", height="85%", \
                           loc='center left', \
                           bbox_to_anchor=(-0.07, 0, 1, 1), \
                           bbox_transform=axarr[0].transAxes, \
                           borderpad=0)
    nlabel = 5
    rad_ticks = np.linspace(0, max_radiance_val, nlabel)
    if max_radiance_val > 1.0:
        rad_cbar = fig.colorbar(rad_cont, cax=rad_inset, \
                                 ticks=rad_ticks, format='%0.1f')
    else:     
        rad_cbar = fig.colorbar(rad_cont, cax=rad_inset, \
                                 ticks=rad_ticks, format='%0.2f')
    rad_inset.yaxis.set_ticks_position('left')

    ### Same thing for the phase
    phase_inset = inset_axes(axarr[1], width="4%", height="85%", \
                             loc='center right', \
                             bbox_to_anchor=(0.07, 0, 1, 1), \
                             bbox_transform=axarr[1].transAxes, \
                             borderpad=0)
    # phase_cbar = fig1.colorbar(phase_cont, cax=phase_inset, \
    #                            ticks=[-np.pi, 0, np.pi])
    # phase_cbar.ax.set_yticklabels(['$-\\pi$', '0', '$+\\pi$'])
    phase_cbar = fig.colorbar(phase_cont, cax=phase_inset, ticks=phase_ticks)
    phase_cbar.ax.set_yticklabels(phase_ticklabels)

    if save:
        if not len(figname):
            figname = os.path.join(fig_id, \
                        f'{beam}beam_nearfield.png')
        savepath = os.path.join(base_plotting_directory, figname)
        if verbose:
            print('Saving figure to:')
            print(f'     {savepath}')
        util._make_all_pardirs(savepath,confirm=False)
        fig.savefig(savepath, dpi=150)

    if show:
        plt.show()

    return fig, axarr





def plot_2D_nearfield_components(\
        x_grid, y_grid, efield, simulation_parameters, field_axis=None, \
        title=True, max_field=0.0, manual_phase_plot_lims=(), \
        phase_sign=1.0, plot_microsphere=True, label_position=None, \
        show=True, save=False, figname='', fig_id='', beam='', \
        verbose=True, **kwargs):

    if verbose:
        print()
        print('Generating 2D plot...')
        sys.stdout.flush()

    global global_dict
    base_plotting_directory = global_dict['base_plotting_directory']

    polarisation = simulation_parameters['polarisation']
    wavelength = simulation_parameters['wavelength']
    zOffset = simulation_parameters['zOffset']

    if label_position:
        ms_position = [simulation_parameters['xOffset'], \
                       simulation_parameters['yOffset'], \
                       simulation_parameters['zOffset']]
    else:
        ms_position = None

    ### Decide which Efield component to plot
    if field_axis is None:
        if polarisation == 'X':
            field_axis = 0
        elif polarisation == 'Y':
            field_axis = 1
        else:
            raise ValueError('Field axis to plot is unclear')

    efield_x = efield[:,:,0].real
    efield_y = efield[:,:,1].real

    fig, axarr = plt.subplots(1, 2, figsize=(10,10), sharex=True, sharey=True)
    if title:
        title = util._build_title(beam)
        fig.suptitle('Trap Near-Field Components: ' + title, \
                      fontsize=16, fontweight='bold')

    ### Plot a contour for the microsphere if so desired
    if plot_microsphere:
        for i in range(2):
            microsphere = plt.Sphere(\
                            (0,0), simulation_parameters['radius'], \
                            fc=None, ec='k', ls='--', lw=3, zorder=99)
            axarr[i].add_patch(microsphere)

    ### Plot the radiance and phase of the electric field specifically 
    ### definining the filled contour levels to allow for algorithmic 
    ### construction of the colorbars

    if not max_field:
        max_field = np.max(np.abs(\
            [np.max(efield_x), np.min(efield_x), \
             np.max(efield_y), np.min(efield_y)] ))

    rad_levels = np.linspace(-1.0*max_field, max_field, 501)
    x_cont = axarr[0].pcolormesh(x_grid, z_grid, efield_x, \
                                 vmin=-1.0*max_field, vmax=max_field, \
                                 cmap='RdBu', zorder=3, \
                                 shading='gouraud')
    y_cont = axarr[1].pcolormesh(x_grid, z_grid, efield_y, \
                                 vmin=-1.0*max_field, vmax=max_field, \
                                 cmap='RdBu', zorder=3, \
                                 shading='gouraud')

    ### Clean up the axes and labels
    for i in range(2):
        axarr[i].set_xlim(np.min(x_grid[0,:]), np.max(x_grid[0,:]))
        axarr[i].set_ylim(np.min(z_grid[:,0]), np.max(z_grid[:,0]))
        axarr[i].set_xlabel('Radial Coordinate [$\\mu$m]')

    axarr[0].set_ylabel('Axial Coordinate [$\\mu$m]')

    # ### Add a note with the plotted value of rmax, i.e. the size of the
    # ### circular aperture displayed at the end
    # fig.text(0.5, 0.1, f'{1000*rmax:0.1f} mm radius\naperture', fontsize=16, \
    #           ha='center', va='center')

    ### Add a note with the relative positions of beam and scatterer, noting
    ### that the offset in the filename/simulation is BEAM relative to the
    ### MS at the origin, so that we need to invert the coordinates. I also
    ### want consistent sizing so the plots can be combined into a movie
    if ms_position is not None:
        try:
            iter(ms_position)
        except Exception:
            raise IOError('MS position needs to be iterable')
        assert len(ms_position) == 3, 'MS position should 3 coordinates'
        val_str = 'MS position:\n('
        for var in ms_position:
            if var > 0:
                sign_str = '-'
            else:
                sign_str = ' '
            val_str += sign_str + f'{np.abs(var)*1e6:0.2f}, '
        val_str = val_str[:-2] + ')'

        ms_label = fig.text(0.5, 0.85, f'{val_str} $\\mu$m', \
                             fontsize=12, ha='center', va='center')
        ms_label.set(fontfamily='monospace')

    ### These labels need to have the same vertical extent otherwise they
    ### fuck up with the axis sizing
    axarr[0].set_title('X-Component')
    axarr[1].set_title('Y-Component')

    ### Do a tight_layout(), but then pull in the sides of the figure a bit
    ### to make room for colorbars
    fig.tight_layout()
    fig.subplots_adjust(right=0.925)#, top=0.85, bottom=0.05)

    ### Make the colorbars explicitly first by defining and inset axes
    ### and then plotting the colorbar in the new inset
    field_inset = inset_axes(\
                    axarr[1], width="4%", height="85%", \
                    loc='center right', \
                    bbox_to_anchor=(0.07, 0, 1, 1), \
                    bbox_transform=axarr[1].transAxes, \
                    borderpad=0)
    nlabel = 5
    field_ticks = np.linspace(-1.0*max_field, max_field, nlabel)
    if max_field > 1.0:
        field_cbar = fig.colorbar(y_cont, cax=field_inset, \
                                  ticks=field_ticks, format='%0.1f')
    else:     
        field_cbar = fig.colorbar(y_cont, cax=field_inset, \
                                 ticks=field_ticks, format='%0.2f')
    field_inset.yaxis.set_ticks_position('right')

    if save:
        if not len(figname):
            figname = os.path.join(fig_id, \
                        f'{beam}beam_nearfield.png')
        savepath = os.path.join(base_plotting_directory, figname)
        if verbose:
            print('Saving figure to:')
            print(f'     {savepath}')
        util._make_all_pardirs(savepath,confirm=False)
        fig.savefig(savepath, dpi=150)

    if show:
        plt.show()

    return fig, axarr