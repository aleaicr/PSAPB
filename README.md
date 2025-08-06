# PPBLP - Probabilistic Serviceability Analysis for Crowd-Induced Lateral Vibrations in Footbridges
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8190981.svg)](https://doi.org/10.5281/zenodo.8190981) 

Authors: Alexis Contreras R. and Gastón Fermandois C. 

This repository contains the files and source code to probabilistically assess the lateral performance of pedestrian bridges under crowd-induced lateral vibrations. The method is explained in our publication:

Contreras, A., Fermandois, G. (2025) - Probability of exceeding serviceability criteria for crowd-induced lateral vibrations in footbridges (in review).
 

## Methodology
The problem is addressed in two stages. The first stage is modeling, which takes into account the bridge model, pedestrian lateral foot-force model, crowd behavior model, interaction between pedestrians and bridge, and uncertainty incorporation. The second stage is simulation results analysis and probabilistic analysis.

## Requirements
* Matlab 2022a
* Simulink

## Instructions / How to use
### Assumptions
* The bridge can be modeled as an equivalent simply supported beam with sinusoidal modal shapes.
*  
### Calibrations
#### Bridge model calibration
The following parameters must be calibrated for the bridge under study (consider equivalent properties).

* Lenght $L$
* Linear density $\rho_{lin}$ (rho_lin)
* Modal damping $\xi$ (xi)
* Distributed stiffness ($EI$)
* Number of modes (n_modes)

These parameters must be obtained by replicating the modal shapes of the specific bridge. To help with this step, the first thing that the execution of main.m does is to show the eigenvalue analysis results (poles, frequencies, and damping ratio). Additionally, there is a bridgeCalibration.m file to only show the eigen analysis results ($p$, $\omega_n$, $\xi_n$), the resulting modal shapes ($\psi_n$), the equivalent EOM matrices ($M_e$, $C_e$, $K_e$, $G_e$) and the state-space matrices ($A$,$B$,$C$,$D$)

#### Pedestrian foot-force model calibration
The pedestrian lateral-foot-force model randomly selects the properties of each pedestrian within a distribution of these parameters, so the mean, standard deviation, and simulation limits must be defined instead of a specific value. The distributions should be defined for the following parameters.
* Pedestrian mass ($m_i$)
* Pedestrian longitudinal walking speed ($v_i$)
* Pedestrian gait frequency ($f_i$) (vertical gait frequency, the model calculates and correlates the lateral with the walking speed)
* Walking speed and gait frequency correlation ($\rho_{fv}$)
* Van der Pol parameters ($a_i$, $\lambda_i$)

A Matlab-based Graphic User Interface (GUI) is under development, which will allow visualization of the force exerted by pedestrians (without the bridge's feedback) after defining the parameter values.

#### Simulation parameters
The simulation requires the following information:

* Number of simulations to perform (n_sim)
* Integration time step (t_step) (Runge-Kutta ode4 is used for the numerical integration)
* Time to simulate the pedestrian quantity (tpq)
* Maximum number of pedestrians to be used (n_max)
* Maximum size of a group of pedestrians (n_step) (group sizes of 2,3,5...)
* Step size of the pedestrian quantity (np_step) (stripes are 1 by 1, 10 by 10, ...) (stripes: 10 20 30 40 ... until stripe 200 pedestrians instead of 1by1) 

### Execute simulation
After defining all the inputs in main.m, run the script to perform the n_sim simulations for every stripe. 

The results will be saved in the folder defined in inputs; the predetermined names are yN.txt, ypN.txt, and yppN.txt for maximum bridge displacement, maximum bridge velocity, and maximum bridge acceleration for all time steps.

### Probabilistic analysis
Once n_sim simulations were performed, the probAnalysis.m file must be used to generate the desired fragility curves and fragility surfaces.

### Licence
This is an academic tool only.
