#!/bin/bash

START=$SECONDS

NBATCH=13
NFRAME=101




make_frame(){

    ##########################################################################
    ########################    ARGUMENT PARSING    ##########################
    ##########################################################################

    i=$1
    NFRAME=$2
    echo "Frame [$((i+1))] of [$NFRAME]"



    ##########################################################################
    #########################    MOVIE SETTINGS    ###########################
    ##########################################################################

    ### Custom saving options (for making movies and other things)
    PLOT_BASE="/home/cblakemore/plots/ott_farfield/"
    TITLE="zsweep_0-50um_trans"
    SAVEPATH="${PLOT_BASE}${TITLE}"

    START_Z="-100"
    STOP_Z="0"
    DELTA_Z=$(bc <<< "scale=2; ($STOP_Z - $START_Z) / ($NFRAME - 1)")



    ##########################################################################
    ####################    MATLAB AND OTT SETTINGS    #######################
    ##########################################################################

    OUTPATH="../raw_data/"

    ### Simulation accuracy. Time scales like NMAX^3, with NMAX=100
    ### taking about 10 seconds (~8-9 second overhead on all) for simple
    ### configurations, and NMAX=500 taking about 200 seconds
    # NMAX="50"
    NMAX="200"

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

    ### Output sampling
    NTHETA="1001"
    NPHI="1001"



    ##########################################################################
    ######################    PYTHON/PLOT SETTINGS    ########################
    ##########################################################################

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
    ###############    EXECUTING THE CALCULATION AND PLOT    #################
    ##########################################################################

    FIGNAME=$(printf "frame_%04d.png" "$i")
    ZOFFSET_VAL=$(bc <<< "scale=2; $START_Z + ($i * $DELTA_Z)")
    ZOFFSET=$(printf "%0.2fe-6" "$ZOFFSET_VAL")

    M_ARGSTR1="${RBEAD} ${N_PARTICLE} ${N_MEDIUM} ${WAVELENGTH0}"
    M_ARGSTR2="${NA} ${POLARISATION} ${XOFFSET} ${YOFFSET} ${ZOFFSET}"
    M_ARGSTR3="${NTHETA} ${NPHI} ${NMAX} ${OUTPATH}"
    M_ARGSTR="${M_ARGSTR1} ${M_ARGSTR2} ${M_ARGSTR3}"
    matlab -nodisplay -r "compute_far_field ${M_ARGSTR}; exit;"

    P_ARGSTR1="${RBEAD} ${N_PARTICLE} ${NA} ${XOFFSET} ${YOFFSET} ${ZOFFSET}"
    P_ARGSTR2="${TRANS} ${BEAM} ${RMAX} ${SINE_BREAKDOWN} ${ELEV} ${AZIM}"
    P_ARGSTR="${P_ARGSTR1} ${P_ARGSTR2} ${SAVEPATH} ${FIGNAME}"
    python3 plot_farfield.py ${P_ARGSTR}

}
export -f make_frame


seq 0 1 $((NFRAME-1)) | parallel -j$NBATCH make_frame {} $NFRAME


echo ""
echo "SCRIPT DURATION: $(($SECONDS-START)) seconds"
echo ""


exit 0





