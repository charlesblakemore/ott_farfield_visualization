import os, math, sys, re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import unwrap

import scipy.constants as constants


##########################################################################
##########################################################################

from farfield_plotting import *
#from nearfield_plotting import *
#from force_plotting import *

##########################################################################
##########################################################################

plt.rcParams.update({'font.size': 14})


def update_base_plotting_directory(new_dir):
    global global_dict
    global_dict['base_plotting_directory'] = new_dir



### This one doesn't work yet. Mostly just wrote it down so I remembered
### the ffmpeg command to string frames together
def make_movie(directory, framerate=10, slow_framerate=2, \
               slow_frames=(), outfile=''):
    '''.'''

    directory = os.path.join(global_dict['base_plotting_directory'], directory)

    frames = util._find_movie_frames(directory)

    if not len(outfile):
        outfile = directory.rstrip('/') + '.mp4'
    else:
        assert outfile[-4:] == '.mp4', 'Output file must be mp4 format'

    instruction_file = open('./instructions.txt', 'w')

    duration = 1.0 / framerate
    slow_duration = 1.0 / slow_framerate

    for frame in frames:

        index = int( re.findall(r"\d{4}", frame)[-1] )

        instruction_file.write( "file '{:s}'\n".format(frame) )
        if len(slow_frames):
            if (index > slow_frames[0]) and (index < slow_frames[1]):
                instruction_file.write( "duration {:0.3f}\n".format(slow_duration) )
            else:
                instruction_file.write( "duration {:0.3f}\n".format(duration) )
        else:
            instruction_file.write( "duration {:0.3f}\n".format(duration) )

    instruction_file.close()

    try:
        os.system('rm {:s}'.format(outfile))
    except:
        print("Couldn't (or didn't need to) remove output file with same name.")

    os.system(f'ffmpeg -f concat -safe 0 -i instructions.txt -vsync vfr -pix_fmt yuv420p {outfile}')
    os.system('rm ./instructions.txt')

    return