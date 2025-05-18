# Pedestrian Lateral Footforce
![image](https://github.com/user-attachments/assets/be12fb77-627b-47ac-b53e-2353c38fc93d)

After calibration for multiple masses and walking frequencies
Add new variables 
k1=1.5 (1 is enough, but looks more accurate with 1.5)
k2=4
k3=1

So
lambda1 = k1 x lambda1
lambda2 = k2 x lambda2
lambda3 = k3 x lambda3

I will consider them constants for now, but can also be variables with uncertainty. There are not enough studies to analyse the uncertainty.

# Pedestrian incorporation rate
dt (time to wait to another person to enter to the bridge) is in fact Poisson. with this change I propose to reconsider to use number of pedestrians walking on the bridge (Pi=P) as the crowd load intensity measure and use the average crowd density (Mu=mu, the red line)
 ![image](https://github.com/user-attachments/assets/f3cbf596-8e3f-4e16-ac02-97685d0334b8)
Mu=100; Mu=400

I'm not sure if Simulink can have a variable vector size in the signals. would be grate to optimize this, because the one I created is simulating every pedestrian (200) even if they are not on the bridge yet (Vector is size 200 in every time step).

This wouldn't allow us to define a maximum amount of pedestrians but a maximum crowd density (when defining a maximum probability of exceedence)

# Variable Pededstrian walking speed in function of crowd density around him.
![image](https://github.com/user-attachments/assets/d5323ef5-f29b-4c11-896c-c409254b4557)

To incorporate this model uncertainty (there are multiple deterministic models) we can do a Logic Tree to incorporate all of them.
