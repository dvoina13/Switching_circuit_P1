# Single circuit in V1 capable of switching contexts during movement using an inhibitory population as a switch

### Doris Voina, Stefano Recanatesi, Brian Hu, Eric Shea-Brown, Stefan Mihalas

[bioarxiv](https://www.biorxiv.org/content/10.1101/2020.09.24.309500v1)

### Abstract.
As  animals  adapt  to  their  environments,  their  brains  are  tasked  with  processing stimuli  in  different  sensory  contexts.   Whether  these  computations  are context dependent or independent, they are all implemented in the same neural tissue.  A crucial question is what neural architectures can respond flexibly to a range of stimulus conditions and switch between them.  This is a particular case of flexible architecture that permits multiple related computations within asingle circuit. Here, we address this question in the specific case of the visual system circuitry, focusing on context integration, defined as the integration of feedforward and surround information across visual space. We show that a biologically inspired microcircuit with multiple inhibitory cell types can switch between visual processing of the static context and the moving context. In our model, the VIP population acts as the switch and modulates the visual circuit through a disinhibitory motif. Moreover, the VIP population is efficient, requiring only a relatively small number of neurons to switch contexts. This circuit eliminates noise in videos by using appropriate lateral connections for contextual spatio-temporal surround modulation, having superior denoising performance compared to circuits where only one context is learned.  Our findings shed light on a minimally complex architecture that is capable of switching between two naturalistic contexts using few switching units.

### List of files included

* scripts for generating videos from the BSDS dataset (Results, sec. 2.2)
    * when sliding window is moving horizontally
    generate_moving_camera_BASE.m
    generate_moving_jumps_simple.m

    * when sliding window is moving in any direction
    generate_moving_jumps.m
    
* script for constructing firing rate due to classical receptive fields -- fn's (Results, sec. 2.1, 2.2):
    Prep_forPython_bsr_34_simple.py    

* scripts for constructing weight matrices --fnn's (Results, sec. 2.2, ): 
    * for static weight (W_static)
    create_fn_fnn34_withPytorch_static_simple.py
    * for moving weight (W_moving)
    create_fn_fnn34_withPytorch_simple.py

* script that solves optimization problem to infer weights to and from the VIP: W_{vip->pyr}, W_{vip->sst}, W_{pyr->vip} (Results, sec. 2.4, 2.5):
    * without recurrent connection between VIP and PYR
    ANN_34filters_natImgsVids_conv_firstModel.py    
    * with recurrent connection between VIP and PYR
    ANN_34filters_natImgsVids_conv_simple.py

* analyze reconstructions for image/video denoising (Results, sec. 2.6):
    analyze_reconstructions.py

* script to infer activities of PYR, VIP neurons (Results, sec. 2.7)
    compute_activities_newest.py

* scripts to analyze the weight matrices and activities (regression/GLM analysis) (Results, sec. 2.7):
    infer_weight_distribution.py
    GLM_regression_analysis.py

* Other files
    * sample weights (.npy)
    * sample filters (data_filts_px3_v2.mat is the main one used with 34 filters)
