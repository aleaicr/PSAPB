(last readme update: July 27th 2023)
# PPBLP - Probabilistic Pedestrian Bridges Lateral Performance
authors: Alexis Contreras R. and Gastón Fermandois C. 

This repository contains the files and source code to probabilistically assess the latearl performance of pedestrian bridges under crowd-induced lateral vibrations. The method is explained in our publication:

Contreras A., Fermandois G. (year) - Probability of exceeding serviceability criteria for crowd-induced lateral vibrations in footbridges
doi: 

## Methodology
The methodology divdes the problem in two stages. First, the modelling stage, in this stage the model of the bridge, the pedestrian lateral-foot-forces, how the crowd behaves and how they will move along the bridge, and how is the bridge-pedestrian interaction must me defined. The second stage, is the probabilistic analysis.

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

* Lenght (L)
* Linear density (\rho)
* Modal damping (\xi)
* Distributed stiffness (EI)
* Number of modes (n_modes)

These parameters, must be obtained by replicating the modal shapes of the specific bridge. To help on this step, the first thing that the excecution of main.m does, is to show the eigeinanalysis results (poles, frequencies and damping ratio). Additionally, there is a bridgeCalibration.m file to only show the eigenanalysis results (p,\omega_n,\xi_n), the resulting modal shapes (psi_n), the equivalent EOM matrices (Me,Ce,Ke,Ge) and the Assummed Modes Method matrices (A,B,C,D)

If the bridge cannot be modeled as a simply supported beam with sinusoidal modes, the mode shapes must be changed, or the model must be modified completely

#### Pedestrian foot-force model calibration
The pedestrian lateral-foot-foce model randomly select the properties of each pedestrian within a distribution of these parameters, so the mean, standard deviation and simulation limits must be defined instead of a specific value. The distributions should be defined for the following parameters.
* Pedestrian mass (mi)
* Pedestrian longitudinal walking speed (vi)
* Pedestrian gait frequency (fi) (vertical gait frequency, the model calculates and correlates the lateral with the walking speed)
* Walking speed and gait frequency correlation (rhofv)
* Van der Pol parameters (ai, lambdai)

The following table shows the distributions values used in the case study.

| Propertie | Distribution | Mean | StandardDeviation  | maxValue | minValue | Source |
|-|-|-|-|-|-|-|
| Mass (kg) | Normal | 71.91 | 14.89 | 40 | 50 | Johnson et al 2008 |
| Walking speed (m/s) | Normal | 1.3 | 0.1 | 10 | Pachi et al 2005 |
| Gait frequency (hz) | Normal | 1.8 | 0.11 | 1.2 | 2.4 | Were calibrated to obtain the normal distribution in Pachi et al 2005 using their correlation rhofv = 0.51 |
| ai (-) | Normal | 

A Matlab-based Graphic User Interface (GUI) is under development, this will allow to visualize the force exerted by pedestrians (without the bridge's feedback) after defining the parameters values.

#### Simulation parameters
The simulation needs the following information:

* Number of simulations to be performed (n_sim)
* Integration time step (t_step) (Runge-Kutta ode4 is used for the numerical integration)
* Time to simulate the pedestrian quantity (tpq)
* Maximum number of pedestrians to be used (n_max)
* Maximum size of a group of pedestrians (n_step) (group of 2,3,5...)
* Step size of the pedestrian quantity (np_step) (stripes are 1by1, 10by10, ...) (stripes: 10 20 30 40 .. 200 pedestrians instead of 1by1) 

### Excecute simulation


### Probabilistic analysis
Once n_sim simulations were performed, the 


### Licence
This is only an academic tool.
