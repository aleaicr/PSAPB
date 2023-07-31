# PPBLP - Probabilistic Pedestrian Bridges Lateral Performance
authors: Alexis Contreras R. and Gastón Fermandois C. 

This repository contains the files and source code to probabilistically assess the lateral performance of pedestrian bridges under crowd-induced lateral vibrations. The method is explained in our publication:

Contreras, A., Fermandois, G. (year) - Probability of exceeding serviceability criteria for crowd-induced lateral vibrations in footbridges
doi: 

## Methodology
The methodology divdes the problem in two stages. First, the modeling stage, includes the bridge's model, the pedestrian lateral-foot-forces model, how the crowd behaves and move along the bridge, and how is the bridge-pedestrian interaction. The second stage, is the probabilistic analysis.

## Requirements
* Matlab 2022a
* Simulink

## Instructions / How to use
### Assumptions
* The bridge can be modeled as an equivalent simply supported beam with sinusoidal modal shapes. 
*  
### Calibrations
#### Brirdge model calibration
The following parameters must be calibrated for the bridge under study.

* Lenght $L$
* Linear density $\rho_{lin}$ (rho_lin)
* Modal damping $\xi$ (xi)
* Distributed stiffness ($EI$)
* Number of modes (n_modes)

These parameters, must be obtained by replicating the modal shapes of the specific bridge. To help on this step, the first thing that the excecution of main.m does, is to show the eigeinanalysis results (poles, frequencies and damping ratio). Additionally, there is a bridgeCalibration.m file to only show the eigenanalysis results ($p$, $\omega_n$, $\xi_n$), the resulting modal shapes ($\psi_n$), the equivalent EOM matrices ($M_e$, $C_e$, $K_e$, $G_e$) and the Assummed Modes Method matrices ($A$,$B$,$C$,$D$)

If the bridge cannot be modeled as a simply supported beam with sinusoidal modes, the mode shapes must be changed, or the model must be modified completely

#### Pedestrian foot-force model calibration
The pedestrian lateral-foot-force model randomly selects the properties of each pedestrian within a distribution of these parameters, so the mean, standard deviation and simulation limits must be defined instead of a specific value. The distributions should be defined for the following parameters.
* Pedestrian mass ($m_i$)
* Pedestrian longitudinal walking speed ($v_i$)
* Pedestrian gait frequency ($f_i$) (vertical gait frequency, the model calculates and correlates the lateral with the walking speed)
* Walking speed and gait frequency correlation ($\rho_{fv}$)
* Van der Pol parameters ($a_i$, $\lambda_i$)

The following table shows the distributions values used in the case study.

| Propertie | Distribution | Mean | StandardDeviation  | maxValue | minValue | Source |
|-|-|-|-|-|-|-|
| Mass (kg) | Normal | 71.91 | 14.89 | 40 | 50 | Johnson et al 2008 |
| Walking speed (m/s) | Normal | 1.3 | 0.13 | 0.1 | 10 | Pachi et al 2005 |
| Gait frequency (hz) | Normal | 1.8 | 0.11 | 1.2 | 2.4 | Were calibrated to obtain the normal distribution in Pachi et al 2005 using their correlation rhofv = 0.51 |
| $a_i$ (-) | Normal | 

A Matlab-based Graphic User Interface (GUI) is under development, this will allow to visualize the force exerted by pedestrians (without the bridge's feedback) after defining the parameters values.

#### Simulation parameters
The simulation needs the following information:

* Number of simulations to be performed (n_sim)
* Integration time step (t_step) (Runge-Kutta ode4 is used for the numerical integration)
* Time to simulate the pedestrian quantity (tpq)
* Maximum number of pedestrians to be used (n_max)
* Maximum size of a group of pedestrians (n_step) (group sizes of 2,3,5...)
* Step size of the pedestrian quantity (np_step) (stripes are 1 by 1, 10 by 10, ...) (stripes: 10 20 30 40 ... until stripe 200 pedestrians instead of 1by1) 

### Excecute simulation
After defining all the inputs in main.m, run the script to perform the n_sim simulations for every stripe. The results will be saved on the folder defined in inputs, the predetermined names are yN.txt for maximum bridge displacement in function of time, ypN.txt for maximum bridge velocity in function of time and yppN.txt for maximum bridge acceleration in function of time.

### Probabilistic analysis
Once n_sim simulations were performed, probAnalysis.m file must be used to generate the desired fragility curves and fragility surfaces.

### Licence
This is only an academic tool.
