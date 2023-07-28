(last readme update: July 27th 2023)
# PPBLP - Probabilistic Pedestrian Bridges Lateral Performance
authors: Alexis Contreras R. and Gastón Fermandois C. 

This repository contains the files and source code to probabilistically assess the latearl performance of pedestrian bridges under crowd-induced lateral vibrations. The method is explained in our publication:

Contreras A., Fermandois G. (year) - Probability of exceeding serviceability criteria for crowd-induced lateral vibrations in footbridges
doi: 

The method divdes the problem in three stages:
* The bridge model, in this study we used a simplified model of an equivalent continuos beam and its dynamics is obtained by the Assumed Modes Method (further complexity is recommended for specific bridges).
* The pedestrian lateral foot-force model, the used in this study is novel one that makes use of a Van der Pol oscillator that replicates empirical measurements of this force.
* Crowd-Bridge interaction ...

## Model
Bridge model ... 

* Assumption that the bridge can be modeled as a simple supported beam.
* 

Pedestrian model ...

Interaction ...

* Pedestrians do not vary the modal mass, damping and stiffness.
* No forced synchronization is used in this model as there is no evidence that this issue happen in every bridge.

Crowd behavior ...

* Pedestrians move along the bridge

Simulation

* For each simulation random properties are generated for every pedestrian
* Then simulink --> equivalent coordinates --> physical response

Serviceability condition

* 

Probabilities, fragility curves, 

*

## How to use
### Brirdge model calibration
As the bridge is modeled as an equivalent continuous beam, three parameters must be calibrated.

* Lenght (L)
* Linear density (\rho)
* Modal damping (\xi)
* Distributed Stiffness (EI)

These parameters, must be obtained by replication of the modal shapes of the specific bridges. To help on this step, the first thing that the main.m exacution does, is to show the eigein analysis results (poles, frequencies and damping ratio), also, there is a bridgeCalibration.m file to only show the eigen analysis results (p,\omega_n,\xi_n), the resulting modal shapes (psi_n), the equivalent EOM matrices (Me,Ce,Ke,Ge) and the Assummed Modes Method matrices (A,B,C,D)

If the bridge can not be modeled as a simple supported beam, the modal shapes must be changed or the entire model.

### Pedestrian foot-force model calibration

* Lateral gait frequency (w_i)


The following table shows the distributions used, values and limits.
Pedestrian mass | m --> Normal | Media, desvEst, maxValue, minValue
w --> Normal |

A Matlab-based Graphic User Interface (GUI) is under development, this will allow to visualize the force exerted by pedestrians (without the bridge's feedback) after defining the parameters values.

