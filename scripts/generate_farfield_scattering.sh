#!/bin/bash

START=$SECONDS

##########################################################################
####################    MATLAB AND OTT SETTINGS    #######################
##########################################################################

OUTPATH="../raw_data/"

### Simulation accuracy. Time scales like NMAX^3, with NMAX=100
### taking about 10 seconds (~8-9 second overhead on all) for simple
### configurations, and NMAX=500 taking about 200 seconds
# NMAX="50"
NMAX="50"

### Scatterer variables 
RBEAD="3.76e-6"
N_PARTICLE="1.39"
# N_PARTICLE="1.0"
N_MEDIUM="1.0"


### Trapping beam variables
WAVELENGTH0="1064.0e-9"
# NA="0.2"
NA="0.095"
POLARISATION="X"

XOFFSET="0.0e-6"
YOFFSET="0.0e-6"
ZOFFSET="-2.0e-6"


### Output sampling
NTHETA="701"
NPHI="701"



##########################################################################
######################    PYTHON/PLOT SETTINGS    ########################
##########################################################################

### Custom saving options (for making movies and other things)
# PLOT_BASE=""
PLOT_BASE="/Users/manifestation/Stanford/beads/plots/ott_farfield/test"
# FIGNAME=""
FIGNAME="derp.png"

### NOTE: YOU WILL NEED TO EDIT THE PYTHON FILE YOURSELF IF YOU WANT TO
### CHANGE THE RAY-TRACING ANALYSIS. THE INITIAL HARD-CODED IMPLEMENTATION
### IS PRETTY SIMPLE

# BEAM="inc"
# BEAM="scat"
BEAM="tot"

### Boolean for transmitted vs. reflected
TRANS=1


### 2D aperture at output to examine
RMAX="0.075"
SINE_BREAKDOWN=0


### 3D view settings
ELEV="40.0"
AZIM="20.0"





##########################################################################
##########################################################################
##########################################################################


echo ""
echo "    ------------------------------    "
echo "     Executing MATLAB Calculation     "
echo "    ------------------------------    "

M_ARGSTR1="${RBEAD} ${N_PARTICLE} ${N_MEDIUM} ${WAVELENGTH0}"
M_ARGSTR2="${NA} ${POLARISATION} ${XOFFSET} ${YOFFSET} ${ZOFFSET}"
M_ARGSTR3="${NTHETA} ${NPHI} ${NMAX} ${OUTPATH}"
M_ARGSTR="${M_ARGSTR1} ${M_ARGSTR2} ${M_ARGSTR3}"

matlab -nodesktop -nosplash -r "compute_far_field ${M_ARGSTR}; exit;"

echo ""
echo "COMPUTATION DURATION: $(($SECONDS-START)) seconds"
echo ""



echo ""
echo "    -----------------------------    "
echo "     Plotting Result with Python     "
echo "    -----------------------------    "
echo ""

P_ARGSTR1="${RBEAD} ${N_PARTICLE} ${NA} ${XOFFSET} ${YOFFSET} ${ZOFFSET}"
P_ARGSTR2="${TRANS} ${BEAM} ${RMAX} ${SINE_BREAKDOWN} ${ELEV} ${AZIM}"
P_ARGSTR="${P_ARGSTR1} ${P_ARGSTR2} ${PLOT_BASE} ${FIGNAME}"

python3 plot_farfield.py ${P_ARGSTR}



exit 0





