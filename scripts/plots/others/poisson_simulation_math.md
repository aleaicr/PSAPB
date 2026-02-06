# Mathematical Formulation of Stochastic Pedestrian Load Simulation

## 1. Introduction
This document details the mathematical framework and simulation algorithm used to model the temporal variability of the crowd load on a bridge. The simulation generates a stochastic process representing the number of pedestrians $N(t)$ on the bridge at any given time $t$, derived from a target average density.

## 2. Model Parameters
The simulation is governed by the following parameters:

*   **Bridge Length ($L$)**: The distance pedestrians must traverse $[m]$.
*   **Mean Walking Velocity ($\mu_v$)**: The average speed of a pedestrian $[m/s]$.
*   **Velocity Standard Deviation ($\sigma_v$)**: Variability in pedestrian speed $[m/s]$.
*   **Target Mean Crowd Size ($E[N]$)**: The expected (average) number of pedestrians on the bridge at stationary state.

## 3. Stochastic Arrival Process (Poisson Process)

The arrival of pedestrians is modeled as a **Poisson Process**. This implies that the time intervals between consecutive pedestrian arrivals are independent and exponentially distributed.

### 3.1 Derivation of Arrival Rate ($\lambda$)
To achieve the target mean crowd size $E[N]$, we first determine the mean service time (average time a pedestrian spends on the bridge), denoted as $E[T_{cross}]$.

$$ E[T_{cross}] \approx \frac{L}{\mu_v} $$

From Little's Law in Queueing Theory ($L_{queue} = \lambda W$), which holds for stationary systems, the relationship between the average number of customers in the system ($E[N]$), the arrival rate ($\lambda$), and the average time in the system ($E[T_{cross}]$) is:

$$ E[N] = \lambda \cdot E[T_{cross}] $$

Solving for the required arrival rate $\lambda$ (pedestrians per second):

$$ \lambda = \frac{E[N]}{E[T_{cross}]} = \frac{E[N] \cdot \mu_v}{L} $$

### 3.2 Inter-arrival Times
The time $\Delta t_k$ between the $(k-1)$-th and $k$-th pedestrian arrival follows an Exponential distribution with rate parameter $\lambda$:

$$ \Delta t_k \sim \text{Exp}(\lambda) $$
$$ P(\Delta t \le x) = 1 - e^{-\lambda x}, \quad x \ge 0 $$

The absolute arrival time for the $k$-th pedestrian, $t_{in, k}$, is given by the cumulative sum:

$$ t_{in, k} = \sum_{i=1}^{k} \Delta t_i $$

## 4. Pedestrian Dynamics (Service Process)

Each pedestrian $k$ is assigned a unique walking velocity $v_k$, sampled from a Normal distribution:

$$ v_k \sim \mathcal{N}(\mu_v, \sigma_v^2) $$
*Note: velocities are clipped to a minimum ($v_{min} = 0.5 m/s$) to avoid unrealistic travel times.*

The residence time (or service time) $\tau_k$ for pedestrian $k$ is strictly determined by their velocity:

$$ \tau_k = \frac{L}{v_k} $$

The departure time $t_{out, k}$ is exactly:

$$ t_{out, k} = t_{in, k} + \tau_k $$

## 5. System State $N(t)$

The system can be described as an $M/G/\infty$ queue:
*   **M**: Markovian (Poisson) arrivals.
*   **G**: General distribution of service times (derived from the inverse of the Normal velocity distribution).
*   **$\infty$**: Infinite servers (pedestrians do not impede each other in this 1D model; the bridge has infinite capacity for flow).

The instantaneous number of pedestrians $N(t)$ is defined as the number of pedestrians who have arrived but not yet departed by time $t$:

$$ N(t) = \sum_{k} \mathbb{1}(t_{in, k} \le t < t_{out, k}) $$

Where $\mathbb{1}(\cdot)$ is the indicator function.

## 6. Discrete Event Simulation Algorithm

The code implements a discrete event simulation with the following steps:

1.  **Generate Arrivals**:
    It iteratively samples $\Delta t \sim \text{Exp}(\lambda)$ until $\sum \Delta t > T_{sim}$. This produces a sequence of arrival events $\{t_{in, k}\}$.

2.  **Assign Velocities & Departures**:
    For each $k$, sample $v_k$, calculate $\tau_k$, and determine $t_{out, k}$. This produces a sequence of departure events $\{t_{out, k}\}$.

3.  **Event Aggregation**:
    Create a unified list of events. Each event is a tuple $(t, \delta)$, where:
    *   Arrival: $(t_{in, k}, +1)$
    *   Departure: $(t_{out, k}, -1)$

4.  **Time Evolution**:
    Sort all events by time $t$. Iterate through the sorted events to compute the cumulative sum (path integral) of the state changes $\delta$.

    $$ N(t_j) = \sum_{m=1}^{j} \delta_m $$
    where $t_j$ is the time of the $j$-th event in the sorted list.

This results in a piecewise constant function (a step function) representing the exact crowd load history.
