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
#####################   RAY TRACING      #################################
##########################################################################

def ray_tracing_propagation(distance, index=1.0):
    return np.array([[1, 0], [distance/index, 1]])

def ray_tracing_thin_lens(focal_length):
    return np.array([[1, -1.0/focal_length], [0, 1]])


def get_simple_ray_tracing_matrix():
    '''
    Simple optical system with meridional ray tracing, given we make the
    assumption of a point source in this farfield visualization.
    
    The parabolic mirror and all lens have been assumed ideal for this 
    simple treatment. Zemax will do better eventually
    '''

    T1 = ray_tracing_propagation(50.8e-3, 1.0) # Propagation to parabolic mirror
    La = ray_tracing_thin_lens(50.8e-3)        # Recollimation
    T2 = ray_tracing_propagation(90.8e-3, 1.0) # To telescope
    Lb = ray_tracing_thin_lens(40.0e-3)        # First telescope lens
    T3 = ray_tracing_propagation(50.0e-3, 1.0) # refracting telescope configuration
    Lc = ray_tracing_thin_lens(10.0e-3)        # First telescope lens
    T4 = ray_tracing_propagation(10.0e-3, 1.0) # projection onto QPD

    return T4 @ Lc @ T3 @ Lb @ T2 @ La @ T1






##########################################################################
###########    SELECTING, RE-CASTING, AND NORMALZING DATA    #############
##########################################################################

def load_farfield_data(path, beam='tot', transmitted=True):
    '''
    Loads data saved by the 'compute_far_field.m' MATLAB function, which 
    saves a standard set of data with conventional names to a single 
    directory. This function looks at that directory and tries to load 
    data according to a default assumed format
    '''

    if beam not in ['inc', 'scat', 'tot']:
        raise ValueError("Beam type must be of types: 'inc', 'scat', 'tot'")

    if transmitted:
        trans_str = 'trans'
    else:
        trans_str = 'refl'

    ### These datafiles should be sampled on a regular (theta, phi) grid for 
    ### this to work properly
    data_pts = np.loadtxt(\
        os.path.join(path, f'farfield_points_{trans_str}.txt'), \
        delimiter=',')

    field_real = np.loadtxt(\
        os.path.join(path, f'farfield_{beam}_{trans_str}_real.txt'), \
        delimiter=',')
    field_imag = np.loadtxt(\
        os.path.join(path, f'farfield_{beam}_{trans_str}_imag.txt'), \
        delimiter=',')

    all_theta = data_pts[0]
    all_phi = data_pts[1]

    ### Select the appropriate hemisphere and find all the unique values
    ### of the azimuthal angle
    phipts = np.unique(all_phi)
    if transmitted:
        theta_inds = (all_theta < np.pi/2)
        thetapts = np.unique(all_theta[theta_inds])
    else:
        theta_inds = all_theta > np.pi/2
        thetapts = np.pi - np.unique(all_theta[theta_inds])

    if not np.sum(theta_inds):
        raise ValueError('Specified datafile does not include data from the '\
                + f'selected hemisphere: (transmitted={str(transmitted):s})')

    ### Index all the data for our desired hemisphere
    field_real = field_real[:,theta_inds]
    field_imag = field_imag[:,theta_inds]

    ### Build meshgrids from the MATLAB-exported data
    theta_grid, phi_grid = np.meshgrid(thetapts, phipts, indexing='ij')

    ### Initialize grid-like arrays for the re-cast electric field
    ntheta = len(thetapts)
    nphi = len(phipts)
    efield_grid = np.zeros((ntheta, nphi, 3), dtype=np.complex128)

    ### Extract the single column data to the gridded format
    for i in range(ntheta):
        for k in range(nphi):
            efield_grid[i,k,:] = field_real[:,k*ntheta+i] \
                                    + 1.0j * field_imag[:,k*ntheta+i]

    return theta_grid, phi_grid, efield_grid





def _project_efield(theta_grid, phi_grid, efield_rtp, polarisation, \
                    unwrap_phase): 

    ### First convert the spherical components of the electric field to
    ### cartesian components to match our usual linearly polarized scheme
    if polarisation == 'X':
        efield = efield_rtp[:,:,1] * np.cos(theta_grid) * np.cos(phi_grid) \
                    + efield_rtp[:,:,2] * (-1) * np.sin(phi_grid)
    elif polarisation == 'Y':
        efield = efield_rtp[:,:,1] * np.cos(theta_grid) * np.sin(phi_grid) \
                    + efield_rtp[:,:,2] *  np.cos(phi_grid)

    ### Compute the radiance and phase
    radiance = np.abs(efield)**2
    phase = np.angle(efield)
    if unwrap_phase:
        phase = unwrap.unwrap(phase)

    ### Find the row at which theta=r=0, i.e. points along the optical axis
    ### so the phase can have some sensible origin and be 0 at that origin
    theta_pts = np.mean(theta_grid, axis=-1)
    if theta_pts[0] < theta_pts[-1]:
        center_axis = 0
    elif theta_pts[0] >= theta_pts[-1]:
        center_axis = -1

    ### Offset the phase map to have 0-phase at the center since it's
    ### arbitrary. Maybe we don't want to do this?
    center_phase = np.mean(phase[center_axis,:])
    phase -= center_phase

    return radiance, phase








##########################################################################
##########################    2D PLOTTING    #############################
##########################################################################


def plot_2D_farfield(theta_grid, phi_grid, efield_rtp, \
                     simulation_parameters, title=True, \
                     max_radiance_trans=0.0, max_radiance_refl=0.0, \
                     unwrap_phase=True, transmitted=True, \
                     manual_phase_plot_lims=(), \
                     label_position=None, rmax=0.01, phase_sign=1.0,  \
                     plot_sin_approx_breakdown=False, \
                     ray_tracing_matrix=get_simple_ray_tracing_matrix(), \
                     show=True, save=False, \
                     figname='', fig_id='', beam='', \
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

    radiance, phase = _project_efield(theta_grid, phi_grid, efield_rtp, \
                                      polarisation, unwrap_phase)
    phase *= phase_sign

    ### Correct for sampling the electric field on the wrong farfield 
    ### sphere, since the coordinates are centered on the scatterer
    ### but the beam is often translated relative to the scatterer
    phase -= theta_grid**2 * zOffset * np.pi/wavelength

    ### Project the theta angular coordinate of the transmitted rays to a
    ### a polar radial coordinate, assuming phi is maintained and ignoring
    ### the angular distribution of the resultant rays (which should vanish
    ### for proper alignment)
    r_grid = np.abs( ray_tracing_matrix[1,0] * theta_grid )


    fig, axarr = plt.subplots(1, 2, figsize=(12,6), sharex=True, sharey=True,\
                               subplot_kw=dict(projection='polar'))
    if title:
        title = util._build_title(beam, transmitted)
        fig.suptitle('Image from Output Optics: ' + title, \
                      fontsize=16, fontweight='bold')

    ### Plot a contour for the sine approximation breakdown (with a label)
    ### if so desired
    if plot_sin_approx_breakdown:
        derp_phi = np.linspace(0, 2.0*np.pi, 100)
        derp_r = np.abs(np.pi/6.0 * ray_tracing_matrix[1,0]) * np.ones(100)
        for i in range(2):
            line = axarr[i].plot(derp_phi, derp_r, ls='--', lw=3, \
                                 color='w', zorder=4)
            util._labelLine(axarr[i].get_lines()[0], 3*np.pi/2, \
                       y_offset=-0.005, label='$\\pi/6$ half-cone', \
                       va='bottom', zorder=99)

    ### Plot the radiance and phase of the electric field specifically 
    ### definining the filled contour levels to allow for algorithmic 
    ### construction of the colorbars
    if transmitted and max_radiance_trans:
        max_radiance_val = max_radiance_trans
    elif not transmitted and max_radiance_refl:
        max_radiance_val = max_radiance_refl
    else:
        max_radiance_val = 0

    if not max_radiance_val:
        max_radiance_val = np.max(radiance)

    rad_levels = np.linspace(0, max_radiance_val, 501)
    rad_cont = axarr[0].pcolormesh(phi_grid, r_grid, radiance, \
                                   vmin=0, vmax=max_radiance_val, \
                                   cmap='plasma', zorder=3, \
                                   shading='gouraud')
    inds = r_grid < rmax
    if len(manual_phase_plot_lims):
        min_phase = np.floor(manual_phase_plot_lims[0]/np.pi)
        max_phase = np.ceil(manual_phase_plot_lims[1]/np.pi)
    else:
        min_phase = np.floor(np.min(phase[inds])/np.pi)
        max_phase = np.ceil(np.max(phase[inds])/np.pi)
    phase_levels = np.linspace(min_phase, max_phase, 501)*np.pi
    phase_cont = axarr[1].contourf(phi_grid, r_grid, phase, \
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

    ### Clean up the axes and labels that we don't really care about
    for i in range(2):
        axarr[i].set_rmax(rmax)
        axarr[i].set_yticks([rmax])
        axarr[i].set_yticklabels([])
        axarr[i].set_xticks([])
        axarr[i].grid(False)

    ### Add a note with the plotted value of rmax, i.e. the size of the
    ### circular aperture displayed at the end
    fig.text(0.5, 0.1, f'{1000*rmax:0.1f} mm radius\naperture', fontsize=16, \
              ha='center', va='center')

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
            if transmitted:
                figname = os.path.join(fig_id, \
                            f'{beam}beam_output_image.png')
            else:
                figname = os.path.join(fig_id, \
                            f'{beam}beam_reflected_output_image.png')
        savepath = os.path.join(base_plotting_directory, figname)
        if verbose:
            print('Saving figure to:')
            print(f'     {savepath}')
        util._make_all_pardirs(savepath,confirm=False)
        fig.savefig(savepath, dpi=150)

    if show:
        plt.show()

    return fig, axarr







def plot_3D_farfield(theta_grid, phi_grid, efield_rtp, \
                     simulation_parameters, title='', \
                     max_radiance_trans=0.0, max_radiance_refl=0.0, \
                     unwrap_phase=True, transmitted=True, \
                     manual_phase_plot_lims=(), \
                     label_position=None, rmax=0.01, phase_sign=1.0, \
                     ray_tracing_matrix=get_simple_ray_tracing_matrix(), \
                     view_elev=+40.0, view_azim=20.0, \
                     show=True, save=False, \
                     figname='', fig_id='', beam='', \
                     verbose=True, **kwargs):

    if verbose:
        print()
        print('Generating 3D plot...')
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

    title = util._build_title(beam, transmitted)

    radiance, phase = _project_efield(theta_grid, phi_grid, efield_rtp, \
                                      polarisation, unwrap_phase)
    phase *= phase_sign

    ### Correct for sampling the electric field on the wrong farfield 
    ### sphere, since the coordinates are centered on the scatterer
    ### but the beam is often translated relative to the scatterer
    phase -= theta_grid**2 * zOffset * np.pi/wavelength

    ### Make an array of colors corresponding to the radiance of each
    ### ray in the output farfield
    if transmitted and max_radiance_trans:
        max_radiance_val = max_radiance_trans
    elif not transmitted and max_radiance_refl:
        max_radiance_val = max_radiance_refl
    else:
        max_radiance_val = 0

    if not max_radiance_val:
        max_radiance_val = np.max(radiance)

    radiance_norm = colors.Normalize(vmin=0, vmax=max_radiance_val)
    radiance_smap = cm.ScalarMappable(norm=radiance_norm, cmap='plasma')
    radiance_colors = radiance_smap.to_rgba(radiance)

    ### Project the theta angular coordinate of the transmitted rays to a
    ### a polar radial coordinate, assuming phi is maintained and ignoring
    ### the angular distribution of the resultant rays (which should vanish
    ### for proper alignment)
    r_grid = np.abs( ray_tracing_matrix[1,0] * theta_grid )

    # misalignment_phase = (ray_tracing_matrix[0,0] * theta_grid * 10.0e-3)\
    #                         *2.0*np.pi/1064.0e-9

    inds = r_grid <= rmax
    if len(manual_phase_plot_lims):
        inds *= ( (phase > manual_phase_plot_lims[0]) * \
                    (phase < manual_phase_plot_lims[1]) )
    radiance_colors[np.logical_not(inds)] = (1.0, 1.0, 1.0, 0.0)

    X = r_grid * np.cos(phi_grid)
    Y = r_grid * np.sin(phi_grid)
    Z = phase #+ misalignment_phase

    fig, ax = plt.subplots(1, 1, figsize=(9,7), \
                           subplot_kw=dict(projection='3d'))

    # if title:
    #     fig.suptitle('Image from Output Optics: ' + title, \
    #                   fontsize=16, fontweight='bold')

    wavefront_surf = ax.plot_surface(1e3*X, 1e3*Y, Z, \
                        rstride=1, cstride=1, \
                        facecolors=radiance_colors, linewidth=0, \
                        antialiased=False, shade=False)

    ax.view_init(elev=view_elev, azim=view_azim)

    ax.set_xlabel('X coord [mm]', labelpad=10)
    ax.set_ylabel('Y coord [mm]', labelpad=10)
    ax.set_zlabel('Phase [rad]', labelpad=5)

    ax.set_xlim(-1e3*rmax, 1e3*rmax)
    ax.set_ylim(-1e3*rmax, 1e3*rmax)


    if len(manual_phase_plot_lims):
        min_phase = np.floor(manual_phase_plot_lims[0]/np.pi)
        max_phase = np.ceil(manual_phase_plot_lims[1]/np.pi)
    else:
        min_phase = np.floor(np.min(phase[inds])/np.pi)
        max_phase = np.ceil(np.max(phase[inds])/np.pi)

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

    ax.set_zlim(min_phase, max_phase)
    ax.set_zticks(phase_ticks)
    ax.set_zticklabels(phase_ticklabels)

    # plt.subplots_adjust(left=0, right=1.0, top=1.0, bottom=0.05)
    fig.tight_layout()


    # ### Add a note with the plotted value of rmax, i.e. the size of the
    # ### circular aperture displayed at the end
    # fig.text(0.5, 0.1, f'{100*rmax:0.1f} cm radius\naperture', fontsize=16, \
    #           ha='center', va='center')

    # ### Add a note with the relative positions of beam and scatterer, noting
    # ### that the offset in the filename/simulation is BEAM relative to the
    # ### MS at the origin, so that we need to invert the coordinates. I also
    # ### want consistent sizing so the plots can be combined into a movie
    # if ms_position is not None:
    #     try:
    #         iter(ms_position)
    #     except Exception:
    #         raise IOError('MS position needs to be iterable')
    #     assert len(ms_position) == 3, 'MS position should 3 coordinates'
    #     val_str = 'MS position:\n('
    #     for var in ms_position:
    #         if var > 0:
    #             sign_str = '-'
    #         else:
    #             sign_str = ' '
    #         val_str += sign_str + f'{np.abs(var)*1e6:0.2f}, '
    #     val_str = val_str[:-2] + ')'

    #     ms_label = fig.text(0.5, 0.85, f'{val_str} $\\mu$m', \
    #                          fontsize=12, ha='center', va='center')
    #     ms_label.set(fontfamily='monospace')


    if save:
        if not len(figname):
            if transmitted:
                figname = os.path.join(fig_id, \
                            f'{beam}beam_output_image_3d.png')
            else:
                figname = os.path.join(fig_id, \
                            f'{beam}beam_reflected_output_image_3d.png')
        savepath = os.path.join(base_plotting_directory, figname)
        if verbose:
            print('Saving figure to:')
            print(f'     {savepath}')
        util._make_all_pardirs(savepath,confirm=False)
        fig.savefig(savepath, dpi=150)

    if show:
        plt.show()

    return fig, ax



