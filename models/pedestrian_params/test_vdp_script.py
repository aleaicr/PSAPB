#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Imports
import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import cm

# Styling
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['legend.framealpha'] = 0.9
plt.rcParams['legend.edgecolor'] = 'none'

# Colors
COLOR_BLUE_PAPER = '#0f62fe'
COLOR_RED_PAPER = '#da1e28'
COLOR_BLACK = '#000000'
COLOR_GRAY_DARK = '#333333'
COLOR_GRAY_LIGHT = '#c1c7cd'

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_figure(fig, name):
    """ 
    To save figures
    """
    png_path = os.path.join(OUTPUT_DIR, f"{name}.png")
    pdf_path = os.path.join(OUTPUT_DIR, f"{name}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"Saved {name} to {OUTPUT_DIR}")


def sampling_distributions(distribution: str, n_samples: int, **kwargs):
    """
    Samples from various probability distributions (truncated if min/max given).
    
    Parameters
    ----------
    distribution : str
        Name of the distribution: 'normal', 'multivariatenormal', 'uniform'.
    n_samples : int
        Number of samples to generate.
    **kwargs : 
        - For 'normal': mu_normal, sigma_normal (required), min_value, max_value (optional)
        - For 'multivariatenormal': mu_multi, covariance_multi (required), min_value, max_value (optional)
        - For 'uniform': range_uniform (required as [min, max])
    
    Returns
    -------
    sample : np.ndarray
        - For 'normal' or 'uniform': (n_samples,) 1D array.
        - For 'multivariatenormal': (n_samples, D) 2D array.
    """
    distribution = distribution.lower()
    min_val = kwargs.get('min_value', None)
    max_val = kwargs.get('max_value', None)
    
    if distribution == 'normal':
        mu = kwargs.get('mu_normal')
        sigma = kwargs.get('sigma_normal')
        if mu is None or sigma is None:
            raise ValueError("For 'normal', 'mu_normal' and 'sigma_normal' are required.")
        
        sample = np.random.normal(mu, sigma, n_samples)
        
        # Resample out of bounds
        if min_val is not None or max_val is not None:
            min_v = min_val if min_val is not None else -np.inf
            max_v = max_val if max_val is not None else np.inf
            out_of_bounds = (sample < min_v) | (sample > max_v)
            while np.any(out_of_bounds):
                n_resample = np.sum(out_of_bounds)
                sample[out_of_bounds] = np.random.normal(mu, sigma, n_resample)
                out_of_bounds = (sample < min_v) | (sample > max_v)
        
        return sample
    
    elif distribution == 'multivariatenormal':
        mu_vec = kwargs.get('mu_multi')
        cov_mat = kwargs.get('covariance_multi')
        if mu_vec is None or cov_mat is None:
            raise ValueError("For 'multivariatenormal', 'mu_multi' and 'covariance_multi' are required.")
        
        mu_vec = np.asarray(mu_vec).flatten()
        cov_mat = np.asarray(cov_mat)
        D = len(mu_vec)
        
        sample = np.random.multivariate_normal(mu_vec, cov_mat, n_samples)
        
        # Handle truncated sampling
        if min_val is not None or max_val is not None:
            if min_val is None:
                min_v = np.full(D, -np.inf)
            elif np.isscalar(min_val):
                min_v = np.full(D, min_val)
            else:
                min_v = np.asarray(min_val).flatten()
            
            if max_val is None:
                max_v = np.full(D, np.inf)
            elif np.isscalar(max_val):
                max_v = np.full(D, max_val)
            else:
                max_v = np.asarray(max_val).flatten()
            out_of_bounds_rows = np.any((sample < min_v) | (sample > max_v), axis=1)
            while np.any(out_of_bounds_rows):
                n_resample = np.sum(out_of_bounds_rows)
                sample[out_of_bounds_rows, :] = np.random.multivariate_normal(mu_vec, cov_mat, n_resample)
                out_of_bounds_rows = np.any((sample < min_v) | (sample > max_v), axis=1)
        return sample
    
    elif distribution == 'uniform':
        range_uniform = kwargs.get('range_uniform')
        if range_uniform is None or len(range_uniform) != 2:
            raise ValueError("For 'uniform', 'range_uniform' [min, max] is required.")
        min_u, max_u = range_uniform
        if min_u >= max_u:
            raise ValueError("In 'range_uniform', min must be less than max.")
        sample = np.random.uniform(min_u, max_u, n_samples)
        return sample
    
    else:
        raise ValueError(f"Unknown distribution: {distribution}. Supported: 'normal', 'multivariatenormal', 'uniform'.")



def eom_multi_pedestrian(t, x, N_p, lambda_1, lambda_2, lambda_3, a_p, omega_p, ypp_func):
    """
    Computes the derivative of the state vector for the multi-pedestrian VdP oscillator.
    """
    dxdt = np.zeros(2 * N_p)
    
    v_p = x[0::2]       # displacements (N_p, 1)
    v_p_dot = x[1::2]
    
    ypp = ypp_func(t)   # scalar bridge acc
    
    # Compute v_p_ddot
    v_p_ddot = (
        -(lambda_1 * v_p_dot**2 + lambda_2 * v_p**2 - lambda_3 * a_p) * v_p_dot
        - omega_p**2 * v_p
        - ypp
    )
    
    dxdt[0::2] = v_p_dot
    dxdt[1::2] = v_p_ddot
    
    return dxdt

def run_simulation(t_end, A_y, f_bridge, N_p=1, return_full=False):
    """
    Run the VdP oscillator simulation.
    
    Parameters
    ----------
    t_end : float
        End time for simulation (seconds)
    A_y : float
        Amplitude of bridge acceleration [m/s^2]
    f_bridge : float
        Frequency of bridge acceleration [Hz]
    N_p : int
        Number of pedestrians to simulate (default: 1)
    return_full : bool
        If True, return full simulation data; otherwise just return data for figures
    
    Returns
    -------
    dict with simulation results
    """
    #### MOVE THIS TO MAIN
    g = 9.81        # m/s^2
    b = 2 * np.pi 
    
    # Scaling factors
    k1 = 1.5        # lambda_1 = k1 * lambda_p / (b * A^2)
    k2 = 3.5        # lambda_2 = k2 * lambda_p * b / A^2
    k3 = 0.8        # lambda_3 = k3 * lambda_p * b
    alpha_s = 0.5 
    
    # Initial Conditions 
    x0_range = [-0.03, 0.03]
    x0p_range = [-0.3, 0.3]
    
    # Bridge parameters
    omega_y = 2 * np.pi * f_bridge  # rad/s
    
    # Mass
    mu_m = 71.91
    sigma_m = 14.89
    m_min = 40.0
    m_max = 120.0
    
    # Frequency and walking velocity
    mu_v = 1.3
    sigma_v = 0.13
    v_min = 0.8
    v_max = 2.0
    rhofv = 0.51
    mu_freq = 1.6
    sigma_freq = 0.11
    freq_min = 1.0
    freq_max = 2.2
    
    mu_fv = np.array([mu_freq, mu_v])
    sigmas = np.array([sigma_freq, sigma_v])
    rhos = np.array([[1, rhofv], [rhofv, 1]])
    min_fv = [freq_min, v_min]
    max_fv = [freq_max, v_max]
    covariance_fv = np.diag(sigmas) @ rhos @ np.diag(sigmas)
    
    # Van der Pol oscillator parameters
    mu_ai = 1.0
    ai_min = 0.8
    ai_max = 2 * mu_ai - ai_min
    sigma_ai = (ai_max - mu_ai) / 3
    mu_lambdai = 10.0
    lambdai_min = 8.0
    lambdai_max = 2 * mu_lambdai - lambdai_min
    sigma_lambdai = (lambdai_max - mu_lambdai) / 3
    
    # sampling
    sample_m = sampling_distributions('normal', N_p, 
                                      mu_normal=mu_m, sigma_normal=sigma_m,
                                      min_value=m_min, max_value=m_max)
    samples_fv = sampling_distributions('multivariatenormal', N_p,
                                        mu_multi=mu_fv, covariance_multi=covariance_fv,
                                        min_value=min_fv, max_value=max_fv)
    samples_freq = samples_fv[:, 0] # 0 column have the frequency
    samples_ai = sampling_distributions('normal', N_p,
                                        mu_normal=mu_ai, sigma_normal=sigma_ai,
                                        min_value=ai_min, max_value=ai_max)
    samples_lambdai = sampling_distributions('normal', N_p,
                                             mu_normal=mu_lambdai, sigma_normal=sigma_lambdai,
                                             min_value=lambdai_min, max_value=lambdai_max)
    v_p_0_samples = x0_range[0] + (x0_range[1] - x0_range[0])*np.random.rand(N_p)
    v_p_dot_0_samples = x0p_range[0] + (x0p_range[1]-x0p_range[0]) * np.random.rand(N_p)
    
    # transform
    A = 2*1/b**2
    mass = sample_m                     # (N_p,1)
    f_vertical = samples_freq           # (N_p,1)
    f_lateral_p = f_vertical / 2        # gait frequency as a function of the vertical
    omega_p = b * f_lateral_p           # natural frequency [rad/s]
    
    lambda_p = samples_lambdai          # (N_p,1)
    a_p = samples_ai                    # (N_p,1)
    lambda_1 = k1 * lambda_p / (b * A**2)
    lambda_2 = k2 * lambda_p * b / A**2
    lambda_3 = k3 * lambda_p * b
    
    # -------------------------------------------------------------------------
    # Initial Conditions
    # -------------------------------------------------------------------------
    x0 = np.empty(2 * N_p)
    x0[0::2] = v_p_0_samples
    x0[1::2] = v_p_dot_0_samples
    
    # -------------------------------------------------------------------------
    # Define bridge acceleration function
    # -------------------------------------------------------------------------
    def ypp_t(t):
        """Bridge lateral acceleration (sinusoidal)."""
        return A_y * np.sin(omega_y * t)
    
    # -------------------------------------------------------------------------
    # Solve ODE
    # -------------------------------------------------------------------------
    print(f"Running simulation for {t_end} seconds...")
    tspan = (0, t_end)
    
    # Determine max step based on simulation length
    max_step = min(0.01, t_end / 10000)  # Ensure adequate resolution
    
    sol = solve_ivp(
        lambda t, x: eom_multi_pedestrian(t, x, N_p, lambda_1, lambda_2, lambda_3, a_p, omega_p, ypp_t),
        tspan,
        x0,
        method='RK45',
        dense_output=True,
        max_step=max_step
    )
    
    t = sol.t
    x = sol.y.T  # shape: (n_times, 2*N_p)
    
    # Extract states
    v_p = x[:, 0::2]        # (n_times, N_p)
    v_p_dot = x[:, 1::2]    # (n_times, N_p)
    
    # Compute bridge acceleration at each time step
    ypp_values = np.array([ypp_t(ti) for ti in t])  # (n_times,)
    
    # Compute v_p_ddot (lateral acceleration of oscillator)
    v_p_ddot = (
        -(lambda_1 * v_p_dot**2 + lambda_2 * v_p**2 - lambda_3 * a_p) * v_p_dot
        - omega_p**2 * v_p
        - ypp_values[:, np.newaxis]
    )
    
    # Compute lateral force
    F_lateral = alpha_s * mass * v_p_ddot  # (n_times, N_p)
    
    print(f"Simulation complete. {len(t)} time steps.")
    
    return {
        't': t,
        'v_p': v_p,
        'v_p_dot': v_p_dot,
        'F_lateral': F_lateral,
        'f_lateral_p': f_lateral_p,
        'N_p': N_p,
        'mass': mass  # Added for weight normalization
    }


def generate_figures():
    """Generate all three figures with proper academic paper styling."""
    
    print("\n" + "="*60)
    print("  Running short simulation for Footforce and Phase plots")
    print("="*60)
    
    sim_short = run_simulation(t_end=15.0, A_y=0.0, f_bridge=0.9)
    
    t_short = sim_short['t']
    v_p = sim_short['v_p']
    v_p_dot = sim_short['v_p_dot']
    F_lateral = sim_short['F_lateral']
    f_lateral_p = sim_short['f_lateral_p']
    N_p = sim_short['N_p']
    
    # Time window for figures (after stabilization)
    t_start_fig = 1.0
    t_end_fig = 6.0
    idx_fig = (t_short >= t_start_fig) & (t_short <= t_end_fig)
    
    t_fig = t_short[idx_fig] - t_start_fig
    F_lateral_fig = F_lateral[idx_fig, :]
    v_p_fig = v_p[idx_fig, :]
    v_p_dot_fig = v_p_dot[idx_fig, :]
    
    print("\nGenerating Figure 1: Pedestrian Lateral Foot-force...")
    
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    for i in range(N_p):
        ax1.plot(t_fig, F_lateral_fig[:, i], 
                 color=COLOR_BLUE_PAPER, linewidth=2.0)
    
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel(r'$F_{y,p}$ [N]')
    ax1.set_xlim([0, 5])
    ax1.set_xticks(np.arange(0, 6, 1))
    ax1.grid(True, linestyle='-', alpha=0.3)
    ax1.set_axisbelow(True)
    
    # Add horizontal zero line
    ax1.axhline(y=0, color=COLOR_GRAY_DARK, linewidth=0.8, linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    save_figure(fig1, "fig1_lateral_footforce")
    plt.close(fig1)
    
    print("Generating Figure 2: Phase Diagram...")
    
    fig2, ax2 = plt.subplots(figsize=(8, 7))
    
    for i in range(N_p):
        ax2.plot(v_p_fig[:, i], v_p_dot_fig[:, i], 
                 color=COLOR_RED_PAPER, linewidth=1.8)
    
    ax2.set_xlabel(r'$v_p$ [m]')
    ax2.set_ylabel(r'$\dot{v}_p$ [m/s]')
    ax2.grid(True, linestyle='-', alpha=0.3)
    ax2.set_axisbelow(True)
    
    # Add origin marker
    ax2.axhline(y=0, color=COLOR_GRAY_DARK, linewidth=0.8, linestyle='-', alpha=0.5)
    ax2.axvline(x=0, color=COLOR_GRAY_DARK, linewidth=0.8, linestyle='-', alpha=0.5)
    
    # Set symmetric limits
    max_disp = np.max(np.abs(v_p_fig)) * 1.1
    max_vel = np.max(np.abs(v_p_dot_fig)) * 1.1
    ax2.set_xlim([-max_disp, max_disp])
    ax2.set_ylim([-max_vel, max_vel])
    
    plt.tight_layout()
    save_figure(fig2, "fig2_phase_diagram")
    plt.close(fig2)
    
    # Long simulation for Figure 3 (DFT) - 2000 seconds with 50 pedestrians
    N_p_dft = 50  # Number of pedestrians for DFT analysis
    
    print("\n" + "="*60)
    print(f"  Running long simulation (2000s) for DFT plot with {N_p_dft} pedestrians")
    print("  A_y = 0.0 m/s², f_bridge = 1.3 Hz")
    print("="*60)
    
    sim_long = run_simulation(t_end=2000.0, A_y=0.0, f_bridge=1.3, N_p=N_p_dft)
    
    t_long = sim_long['t']
    F_lateral_long = sim_long['F_lateral']
    f_lateral_p_long = sim_long['f_lateral_p']
    
    # Use stabilized data (skip first 10 seconds for long simulation)
    t_start_dft = 10.0
    idx_dft = t_long >= t_start_dft
    
    F_lateral_dft = F_lateral_long[idx_dft, :]
    t_dft = t_long[idx_dft]
    
    ##### Figure 3: DFT of Lateral Footforce
    print("Generating Figure 3: DFT of Lateral Footforce...")
    
    # Compute sampling frequency
    dt = np.mean(np.diff(t_dft))  # Average time step
    Fs = 1 / dt                   # Sampling frequency [Hz]
    N_samples = F_lateral_dft.shape[0]
    
    print(f"  Sampling frequency: {Fs:.2f} Hz")
    print(f"  Number of samples: {N_samples}")
    print(f"  Frequency resolution: {Fs/N_samples:.6f} Hz")
    
    # Compute DFT for each pedestrian
    Y = np.fft.fft(F_lateral_dft, axis=0)
    P2 = np.abs(Y)                                  # Two-sided spectrum magnitude
    P1 = P2[:N_samples // 2 + 1, :]                 # Single-sided spectrum
    
    # Normalize by number of samples (optional, for cleaner amplitude interpretation)
    P1_normalized = P1 / N_samples
    
    # Frequency vector
    f_vec = Fs * np.arange(N_samples // 2 + 1) / N_samples
    
    # Limit frequency range to 0-10 Hz for display
    f_max_display = 10.0
    idx_f_display = f_vec <= f_max_display
    
    # Figure
    fig3, ax3 = plt.subplots(figsize=(12, 7))
    
    # Get number of pedestrians
    N_p_plot = sim_long['N_p']
    
    # Sort pedestrians by their lateral frequency for visual coherence
    sort_indices = np.argsort(f_lateral_p_long)
    
    # gradient
    cmap = cm.get_cmap('plasma', N_p_plot)
    
    # Plot each pedestrian's DFT with gradient color
    for idx, i in enumerate(sort_indices):
        color = cmap(idx / (N_p_plot - 1) if N_p_plot > 1 else 0.5)
        ax3.plot(f_vec[idx_f_display], P1[idx_f_display, i], 
                 color=color, linewidth=0.8, alpha=0.7)
    
    # Calculate mean lateral frequency
    f_lateral_mean = np.mean(f_lateral_p_long)
    print(f"  Mean lateral frequency: {f_lateral_mean:.3f} Hz")
    
    # Labels
    harmonic_labels = [
        (1.3, 5.0e6, r'$f_{\ell}$'),       # 1st harmonic
        (3.3, 3.7e6, r'$3f_{\ell}$'),      # 3rd harmonic
        (5.0, 0.7e6, r'$5f_{\ell}$'),      # ...
        (6.7, 0.7e6, r'$7f_{\ell}$'),
        (8.7, 0.5e6, r'$9f_{\ell}$'), 
    ]
    
    for x_pos, y_pos, label in harmonic_labels:
        ax3.text(x_pos, y_pos, label, 
                ha='left', va='bottom', fontsize=14, fontweight='bold',
                color=COLOR_BLACK)
    
    ax3.set_xlabel('Frequency [Hz]')
    ax3.set_ylabel(r'$|DFT(F_{y,p})|$')
    ax3.set_xlim([0, f_max_display])
    ax3.set_ylim(bottom=0)
    ax3.grid(True, linestyle='-', alpha=0.3)
    ax3.set_axisbelow(True)
    
    plt.tight_layout()
    save_figure(fig3, "fig3_dft_lateral_footforce")
    plt.close(fig3)
    
    # Figure 4
    print("Generating Figure 4: DFT of Lateral Footforce (semilogy)...")
    
    fig4, ax4 = plt.subplots(figsize=(12, 7))
    
    # Plot each pedestrian's DFT with gradient color (same as fig3)
    for idx, i in enumerate(sort_indices):
        color = cmap(idx / (N_p_plot - 1) if N_p_plot > 1 else 0.5)
        ax4.semilogy(f_vec[idx_f_display], P1[idx_f_display, i], 
                     color=color, linewidth=0.8, alpha=0.7)
    
    # Add labels on top of each harmonic peak (adjusted for log scale)
    # Format: (x_position, y_position, label_text)
    harmonic_labels_log = [
        (1.3, 6.0e6, r'$f_{\ell}$'),       # 1st harmonic (fundamental)
        (3.3, 5.0e6, r'$3f_{\ell}$'),      # 3rd harmonic
        (5.0, 1.0e6, r'$5f_{\ell}$'),
        (6.7, 8.0e5, r'$7f_{\ell}$'),
        (8.7, 5.0e5, r'$9f_{\ell}$'),
    ]
    
    for x_pos, y_pos, label in harmonic_labels_log:
        ax4.text(x_pos, y_pos, label, 
                ha='left', va='bottom', fontsize=14, fontweight='bold',
                color=COLOR_BLACK)
    
    ax4.set_xlabel('Frequency [Hz]')
    ax4.set_ylabel(r'$|DFT(F_{y,p})|$')
    ax4.set_xlim([0, f_max_display])
    ax4.grid(True, linestyle='-', alpha=0.3, which='both')
    ax4.set_axisbelow(True)
    
    plt.tight_layout()
    save_figure(fig4, "fig4_dft_lateral_footforce_log")
    plt.close(fig4)
    
    # Figure 5: Weight-Normalized Lateral Footforce (from short simulation)
    print("Generating Figure 5: Weight-Normalized Lateral Footforce...")
    
    g = 9.81  # m/s²
    mass_short = sim_short['mass']  # (N_p,) array
    weight = mass_short * g  # Weight of each pedestrian
    
    # Use a shifted time window (start from 1.6s instead of 1.0s) to begin at bottom of oscillation
    t_start_fig5 = 1.6
    t_end_fig5 = 6.6
    idx_fig5 = (t_short >= t_start_fig5) & (t_short <= t_end_fig5)
    
    t_fig5 = t_short[idx_fig5] - t_start_fig5  # Shift to start at 0
    F_lateral_fig5 = F_lateral[idx_fig5, :]
    
    # Normalize F_lateral by weight
    F_normalized = F_lateral_fig5 / weight  # Divide by weight for each pedestrian
    
    fig5, ax5 = plt.subplots(figsize=(14, 6))
    
    for i in range(N_p):
        ax5.plot(t_fig5, F_normalized[:, i], 
                 color=COLOR_BLACK, linewidth=2.0)
    
    ax5.set_xlabel('Time [s]')
    ax5.set_ylabel(r'$F_{y,p} / W_p$ [-]')
    ax5.set_xlim([0, 5])
    ax5.set_ylim([-0.1, 0.1])
    ax5.set_xticks(np.arange(0, 5.5, 0.5))
    ax5.set_yticks([-0.1, -0.05, 0, 0.05, 0.1])
    ax5.grid(True, linestyle='-', alpha=0.3)
    ax5.set_axisbelow(True)
    
    # Add horizontal zero line
    ax5.axhline(y=0, color=COLOR_GRAY_DARK, linewidth=0.8, linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    save_figure(fig5, "fig5_normalized_lateral_footforce")
    plt.close(fig5)
    
    # Single Pedestrian DFT (separate simulation with N_p=1)
    print("\n" + "="*60)
    print("  Running simulation for single pedestrian DFT")
    print("="*60)
    
    sim_single = run_simulation(t_end=2000.0, A_y=0.1, f_bridge=1.3, N_p=1)
    
    t_single = sim_single['t']
    F_lateral_single = sim_single['F_lateral']
    f_lateral_p_single = sim_single['f_lateral_p']
    
    t_start_dft_single = 10.0
    idx_dft_single = t_single >= t_start_dft_single
    
    F_lateral_dft_single = F_lateral_single[idx_dft_single, :]
    t_dft_single = t_single[idx_dft_single]
    
    # Compute DFT for single pedestrian
    dt_single = np.mean(np.diff(t_dft_single))
    Fs_single = 1 / dt_single
    N_samples_single = F_lateral_dft_single.shape[0]
    
    Y_single = np.fft.fft(F_lateral_dft_single, axis=0)
    P2_single = np.abs(Y_single)
    P1_single = P2_single[:N_samples_single // 2 + 1, :]
    
    f_vec_single = Fs_single * np.arange(N_samples_single // 2 + 1) / N_samples_single
    
    idx_f_display_single = f_vec_single <= f_max_display
    
    # Figure 6: DFT of Single Pedestrian (Linear Scale)
    print("Generating Figure 6: Single Pedestrian DFT (linear)...")
    
    fig6, ax6 = plt.subplots(figsize=(12, 7))
    
    ax6.plot(f_vec_single[idx_f_display_single], P1_single[idx_f_display_single, 0], 
             color=COLOR_BLUE_PAPER, linewidth=1.5)
    
    # Add labels for single pedestrian
    harmonic_labels_single = [
        (1.3, 4.5e6, r'$f_{\ell}$'),       # 1st harmonic
        (3.3, 3.2e6, r'$3f_{\ell}$'),      # 3rd harmonic
        (5.3, 0.35e6, r'$5f_{\ell}$'),
        (7.3, 0.25e6, r'$7f_{\ell}$'),
        (9.3, 0.15e6, r'$9f_{\ell}$'),
    ]
    
    for x_pos, y_pos, label in harmonic_labels_single:
        if x_pos <= f_max_display:
            ax6.text(x_pos, y_pos, label, 
                    ha='left', va='bottom', fontsize=14, fontweight='bold',
                    color=COLOR_BLACK)
    
    ax6.set_xlabel('Frequency [Hz]')
    ax6.set_ylabel(r'$|DFT(F_{y,p})|$')
    ax6.set_xlim([0, f_max_display])
    ax6.set_ylim(bottom=0)
    ax6.grid(True, linestyle='-', alpha=0.3)
    ax6.set_axisbelow(True)
    
    plt.tight_layout()
    save_figure(fig6, "fig6_dft_single_pedestrian")
    plt.close(fig6)
    
    # FIG7 
    print("Generating Figure 7: Single Pedestrian DFT (semilogy)...")
    
    fig7, ax7 = plt.subplots(figsize=(12, 7))
    
    ax7.semilogy(f_vec_single[idx_f_display_single], P1_single[idx_f_display_single, 0], 
                 color=COLOR_BLUE_PAPER, linewidth=1.5)
    
    # Add labels
    harmonic_labels_single_log = [
        (1.3, 5.0e6, r'$f_{\ell}$'),       # 1st harmonic
        (3.3, 4.0e6, r'$3f_{\ell}$'),      # 3rd harmonic
        (5.3, 5.0e5, r'$5f_{\ell}$'),
        (7.3, 3.0e5, r'$7f_{\ell}$'),
        (9.3, 2.0e5, r'$9f_{\ell}$'),
    ]
    
    for x_pos, y_pos, label in harmonic_labels_single_log:
        if x_pos <= f_max_display:
            ax7.text(x_pos, y_pos, label, 
                    ha='left', va='bottom', fontsize=14, fontweight='bold',
                    color=COLOR_BLACK)
    
    ax7.set_xlabel('Frequency [Hz]')
    ax7.set_ylabel(r'$|DFT(F_{y,p})|$')
    ax7.set_xlim([0, f_max_display])
    ax7.grid(True, linestyle='-', alpha=0.3, which='both')
    ax7.set_axisbelow(True)
    
    plt.tight_layout()
    save_figure(fig7, "fig7_dft_single_pedestrian_log")
    plt.close(fig7)
    
    # Figure 8 - Combined DFT
    print("Generating Figure 8: Combined DFT + Phase Plot...")
    
    fig8, (ax8a, ax8b) = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, i in enumerate(sort_indices):
        color = cmap(idx / (N_p_plot - 1) if N_p_plot > 1 else 0.5)
        ax8a.plot(f_vec[idx_f_display], P1[idx_f_display, i], 
                 color=color, linewidth=0.8, alpha=0.7)
    
    # Add harmonic labels
    for x_pos, y_pos, label in harmonic_labels:
        ax8a.text(x_pos, y_pos, label, 
                ha='left', va='bottom', fontsize=14, fontweight='bold',
                color=COLOR_BLACK)
    
    ax8a.set_xlabel('Frequency [Hz]')
    ax8a.set_ylabel(r'$|DFT(F_{y,p})|$')
    ax8a.set_xlim([0, f_max_display])
    ax8a.set_ylim(bottom=0)
    ax8a.grid(True, linestyle='-', alpha=0.3)
    ax8a.set_axisbelow(True)
    ax8a.set_title('(a)', loc='left', fontweight='bold', fontsize=14)
    
    # Right subplot: Phase Plot
    for i in range(N_p):
        ax8b.plot(v_p_fig[:, i], v_p_dot_fig[:, i], 
                 color=COLOR_RED_PAPER, linewidth=1.8)
    
    ax8b.set_xlabel(r'$v_p$ [m]')
    ax8b.set_ylabel(r'$\dot{v}_p$ [m/s]')
    ax8b.grid(True, linestyle='-', alpha=0.3)
    ax8b.set_axisbelow(True)
    
    ax8b.axhline(y=0, color=COLOR_GRAY_DARK, linewidth=0.8, linestyle='-', alpha=0.5)
    ax8b.axvline(x=0, color=COLOR_GRAY_DARK, linewidth=0.8, linestyle='-', alpha=0.5)
    
    max_disp = np.max(np.abs(v_p_fig)) * 1.1
    max_vel = np.max(np.abs(v_p_dot_fig)) * 1.1
    ax8b.set_xlim([-max_disp, max_disp])
    ax8b.set_ylim([-max_vel, max_vel])
    ax8b.set_title('(b)', loc='left', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    save_figure(fig8, "fig8_combined_dft_phase")
    plt.close(fig8)
    
    
    # Figure 9: DFT (multi-pedestrian) - Same proportions as combine figure
    print("\nGenerating Figure 9: DFT (proportional for paper)...")
    
    fig9, ax9 = plt.subplots(figsize=(7, 6))
    
    for idx, i in enumerate(sort_indices):
        color = cmap(idx / (N_p_plot - 1) if N_p_plot > 1 else 0.5)
        ax9.plot(f_vec[idx_f_display], P1[idx_f_display, i], 
                 color=color, linewidth=0.8, alpha=0.7)
    
    # Add harmonic labels
    for x_pos, y_pos, label in harmonic_labels:
        ax9.text(x_pos, y_pos, label, 
                ha='left', va='bottom', fontsize=14, fontweight='bold',
                color=COLOR_BLACK)
    
    ax9.set_xlabel('Frequency [Hz]')
    ax9.set_ylabel(r'$|DFT(F_{y,p})|$')
    ax9.set_xlim([0, f_max_display])
    ax9.set_ylim(bottom=0)
    ax9.grid(True, linestyle='-', alpha=0.3)
    ax9.set_axisbelow(True)
    
    plt.tight_layout()
    save_figure(fig9, "fig9_dft_proportional")
    plt.close(fig9)
    
    # 
    # Figure 10: Phase Plot - Same proportions as combined figure
    print("Generating Figure 10: Phase Plot (proportional for paper)...")
    
    fig10, ax10 = plt.subplots(figsize=(7, 6))
    
    for i in range(N_p):
        ax10.plot(v_p_fig[:, i], v_p_dot_fig[:, i], 
                 color=COLOR_RED_PAPER, linewidth=1.8)
    
    ax10.set_xlabel(r'$v_p$ [m]')
    ax10.set_ylabel(r'$\dot{v}_p$ [m/s]')
    ax10.grid(True, linestyle='-', alpha=0.3)
    ax10.set_axisbelow(True)
    
    # Add origin lines
    ax10.axhline(y=0, color=COLOR_GRAY_DARK, linewidth=0.8, linestyle='-', alpha=0.5)
    ax10.axvline(x=0, color=COLOR_GRAY_DARK, linewidth=0.8, linestyle='-', alpha=0.5)
    
    # Set symmetric limits for phase plot
    max_disp = np.max(np.abs(v_p_fig)) * 1.1
    max_vel = np.max(np.abs(v_p_dot_fig)) * 1.1
    ax10.set_xlim([-max_disp, max_disp])
    ax10.set_ylim([-max_vel, max_vel])
    
    plt.tight_layout()
    save_figure(fig10, "fig10_phase_proportional")
    plt.close(fig10)



##### Main


if __name__ == '__main__':
    np.random.seed(42)
    generate_figures()
