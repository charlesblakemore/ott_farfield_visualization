import os, sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import unwrap

import scipy.constants as constants

plt.rcParams.update({'font.size': 14})




base_plotting_directory = os.path.abspath('../plots/')



##########################################################################
#####################   RAY TRACING      #################################
##########################################################################

def _ray_tracing_propagation(distance, index=1.0):
    return np.array([[1, 0], [distance/index, 1]])

def _ray_tracing_thin_lens(focal_length):
    return np.array([[1, -1.0/focal_length], [0, 1]])


def get_simple_ray_tracing_matrix():
    '''
    Simple optical system with meridional ray tracing, given we make the
    assumption of a point source in this farfield visualization.
    
    The parabolic mirror and all lens have been assumed ideal for this 
    simple treatment. Zemax will do better eventually
    '''

    T1 = _ray_tracing_propagation(50.8e-2, 1.0) # Propagation to parabolic mirror
    La = _ray_tracing_thin_lens(50.8e-2)        # Recollimation
    T2 = _ray_tracing_propagation(90.8e-2, 1.0) # To telescope
    Lb = _ray_tracing_thin_lens(40.0e-2)        # First telescope lens
    T3 = _ray_tracing_propagation(50.0e-2, 1.0) # refracting telescope configuration
    Lc = _ray_tracing_thin_lens(10.0e-2)        # First telescope lens
    T4 = _ray_tracing_propagation(10.0e-2, 1.0) # projection onto QPD

    return T4 @ Lc @ T3 @ Lb @ T2 @ La @ T1






##########################################################################
###########    SELECTING, RE-CASTING, AND NORMALZING DATA    #############
##########################################################################

def load_data(path, beam='tot', transmitted=True):
    '''
    Loads data saved by the 'compute_far_field.m' MATLAB function, which 
    saves a standard set of data with conventional names to a single 
    directory. This function looks at that directory and tries to load 
    data according to a default assumed format
    '''

    if beam not in ['inc', 'scat', 'tot']:
        raise ValueError("Beam type must be of types: 'inc', 'scat', 'tot'")

    ### These datafiles should be sampled on a regular (theta, phi) grid for 
    ### this to work properly
    data_pts = np.loadtxt(os.path.join(path, 'farfield_points.txt'), \
                          delimiter=',')

    field_real = np.loadtxt(os.path.join(path, f'farfield_{beam}_real.txt'), \
                          delimiter=',')
    field_imag = np.loadtxt(os.path.join(path, f'farfield_{beam}_imag.txt'), \
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









##########################################################################
##########################    2D PLOTTING    #############################
##########################################################################


def plot_2D_farfield(theta_grid, phi_grid, efield_rtp, title='', \
                     polarisation='X', max_radiance_val=0.0, \
                     unwrap_phase=True, transmitted=True, \
                     manual_phase_plot_lims=(), \
                     ms_position=None, rmax=5.0e-2, \
                     plot_sin_approx_breakdown=False, \
                     ray_tracing_matrix=get_simple_ray_tracing_matrix(), \
                     show=True, save=False, \
                     figname='', fig_id='', beam=''):

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

    ### Offset the phase map to have 0-phase at the center since it's
    ### arbitrary. Maybe we don't want to do this?
    if transmitted:
        center_phase = np.mean(phase[0,:])
    else:
        center_phase = np.mean(phase[-1,:])
    phase -= center_phase

    ### Project the theta angular coordinate of the transmitted rays to a
    ### a polar radial coordinate, assuming phi is maintained and ignoring
    ### the angular distribution of the resultant rays (which should vanish
    ### for proper alignment)
    r_grid = np.abs( ray_tracing_matrix[1,0] * theta_grid )


    fig, axarr = plt.subplots(1, 2, figsize=(12,6), sharex=True, sharey=True,\
                               subplot_kw=dict(projection='polar'))

    if title:
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
            _labelLine(axarr[i].get_lines()[0], 3*np.pi/2, \
                       y_offset=-0.005, label='$\\pi/6$ half-cone', \
                       va='bottom', zorder=99)

    ### Plot the radiance and phase of the electric field specifically 
    ### definining the filled contour levels to allow for algorithmic 
    ### construction of the colorbars
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
    fig.text(0.5, 0.1, f'{100*rmax:0.1f} cm radius\naperture', fontsize=16, \
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
    rad_cbar = fig.colorbar(rad_cont, cax=rad_inset, \
                             ticks=rad_ticks, format='%0.1f')
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
                            f'{beam}beam_reflected_output_image.svg')
        savepath = os.path.join(base_plotting_directory, figname)
        print('Saving figure to:')
        print(f'     {savepath}')
        print()
        _make_all_pardirs(savepath,confirm=False)
        fig.savefig(savepath, dpi=150)

    if show:
        plt.show()

    return fig, axarr










def _labelLine(line, x, x_offset=0.0, y_offset=0.0, \
               alpha=0.0, label=None, align=True, **kwargs):

    ax = line.axes
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if (x < xdata[0]) or (x > xdata[-1]):
        print('x label location is outside data range!')
        return

    #Find corresponding y co-ordinate and angle of the line
    ip = 1
    for i in range(len(xdata)):
        if x < xdata[i]:
            ip = i
            break

    y = ydata[ip-1] + (ydata[ip]-ydata[ip-1])*(x-xdata[ip-1])/(xdata[ip]-xdata[ip-1])

    if not label:
        label = line.get_label()

    if align:
        #Compute the slope
        dx = xdata[ip] - xdata[ip-1]
        dy = ydata[ip] - ydata[ip-1]
        ang = math.degrees(math.atan2(dy,dx))

        #Transform to screen co-ordinates
        pt = np.array([x,y]).reshape((1,2))
        trans_angle = ax.transData.transform_angles(np.array((ang,)),pt)[0]

    else:
        trans_angle = 0

    #Set a bunch of keyword arguments
    if 'color' not in kwargs:
        kwargs['color'] = line.get_color()

    if ('horizontalalignment' not in kwargs) and ('ha' not in kwargs):
        kwargs['ha'] = 'center'

    if ('verticalalignment' not in kwargs) and ('va' not in kwargs):
        kwargs['va'] = 'center'

    if 'backgroundcolor' not in kwargs:
        kwargs['backgroundcolor'] = ax.get_facecolor()

    if 'clip_on' not in kwargs:
        kwargs['clip_on'] = True

    if 'zorder' not in kwargs:
        kwargs['zorder'] = 2.5
        
    t = ax.text(x+x_offset, y+y_offset, label, rotation=trans_angle, **kwargs)
    t.set_bbox(dict(alpha=alpha))




def _make_all_pardirs(path, confirm=True):
    '''Function to help pickle from being shit. Takes a path
       and looks at all the parent directories etc and tries 
       making them if they don't exist.
       INPUTS: path, any path which needs a hierarchy already 
                     in the file system before being used
       OUTPUTS: none
       '''

    parts = path.split('/')
    parent_dir = '/'
    for ind, part in enumerate(parts):
        if ind == 0 or ind == len(parts) - 1:
            continue
        parent_dir += part
        parent_dir += '/'
        if not os.path.isdir(parent_dir):
            if confirm:
                print()
                print('Make this directory?')
                print(f'    {parent_dir}')
                print()
                answer = input('(Y/N): ')
            else:
                answer = 'Y'

            if ('y' in answer) or ('Y' in answer):
                os.mkdir(parent_dir)






