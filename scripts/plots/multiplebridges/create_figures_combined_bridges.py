import os
import numpy as np
import scipy.io
import scipy.stats as stats
import scipy.optimize as optimize
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import h5py

# --- Configuration ---
BRIDGES_TO_LOAD = [1, 2, 3, 4, 5]
DELTA_VALUES = [0.15, 0.21]  # m/s^2

# --- Aesthetics (Matching SURFACE_curves.py) ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 1.5

# Colors
COLOR_GRAY_SIM = '#4d5358' # Dark grey for scatter
COLOR_BLACK = '#000000'

# Bridge Colors (matching MATLAB hex codes)
COLORS_BRIDGES = [
    '#0f62fe', # Bridge 1 (Blue)
    '#da1e28', # Bridge 2 (Red)
    '#198038', # Bridge 3 (Green)
    '#8a3ffc', # Bridge 4 (Purple)
    '#E37400', # Bridge 5 (Orange)
]

# --- Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Expecting: separate repo structure or similar to MATLAB logic relative to script
# If script is in scripts/plots/multiplebridges/, repo root is 3 levels up
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
RESULTS_DIR = os.path.join(REPO_ROOT, 'results', 'main_analysis')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'assets', 'figures', 'png', 'combined_bridges')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Helper Functions ---

def calculate_mom_fragility(cleaned_data, delta):
    fragility = []
    for vals in cleaned_data:
        if len(vals) > 0:
            log_vals = np.log(vals)
            mu_ln = np.mean(log_vals)
            sigma_ln = np.std(log_vals)
            prob_exceed = 1 - norm.cdf(np.log(delta), loc=mu_ln, scale=sigma_ln)
            fragility.append(prob_exceed)
        else:
            fragility.append(np.nan)
    return np.array(fragility)

def save_figure(fig, name):
    """Saves figure to png and pdf"""
    png_path = os.path.join(OUTPUT_DIR, f"{name}.png")
    # pdf_path = os.path.join(OUTPUT_DIR, f"{name}.pdf") # Optional
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    # fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"Saved {name}")

def load_data():
    data_store = {}
    
    default_P_ped = np.array([5, 10, 20, 30, 40, 50, 75, 100, 110, 120, 125, 130, 140, 150, 175, 200])

    for b in BRIDGES_TO_LOAD:
        bridge_name = f'bridge{b}'
        file_path = os.path.join(RESULTS_DIR, bridge_name, 'results_2000s.mat')
        
        if not os.path.exists(file_path):
            print(f"Warning: Data for {bridge_name} not found at {file_path}")
            continue
            
        print(f"Loading {bridge_name}...")
        
        P_ped = None
        rm = None
        
        try:
            # Try loading with scipy.io (older .mat files)
            mat_data = scipy.io.loadmat(file_path)
            if 'P_ped' in mat_data:
                P_ped = mat_data['P_ped'].flatten()
            if 'resultsMatrix' in mat_data:
                rm = mat_data['resultsMatrix']
                
        except NotImplementedError:
             # Fallback to h5py for v7.3 .mat files
             with h5py.File(file_path, 'r') as f:
                if 'P_ped' in f:
                    # h5py reads as (N, 1) usually or needs transpose
                    # Depending on how it was saved, usually we get a dataset
                    P_ped = np.array(f['P_ped']).flatten()
                    
                if 'resultsMatrix' in f:
                    # MATLAB saves column-major, HDF5 reads as row-major (transposed view)
                    # So a MATLAB (Sims x Peds) becomes (Peds x Sims) in h5py default read usually
                    # But we need to be careful. Let's read it and check shape.
                    rm = np.array(f['resultsMatrix'])
                    # We will check dimensions below to fix orientation
        
        if P_ped is None:
            print(f"P_ped missing in {bridge_name}. Using default vector.")
            P_ped = default_P_ped.copy()
            
        if rm is None:
             raise ValueError(f"resultsMatrix missing in {bridge_name}")

        # Fix dimensions
        # Expected: Sims (many) x Peds (small number, len(P_ped))
        r, c = rm.shape
        n_peds = len(P_ped)
        
        # Heuristic: If one dimension matches n_peds, that is the columns (usually).
        # We want shape (Sims, Peds)
        if c == n_peds and r != n_peds:
            pass # Already (Sims x Peds)
        elif r == n_peds and c != n_peds:
            rm = rm.T # Transpose to get (Sims x Peds)
        elif r == n_peds and c == n_peds:
             # Square? ambiguous, assume standard (Sims x Peds) unless user logic says otherwise
             pass
        else:
             print(f"Dimension mismatch in {bridge_name}: {r}x{c} vs {n_peds}. Proceeding with caution.")

        # Filtering Logic
        if b == 2:
            # Exclude 175 and 200 using boolean mask
            mask = ~np.isin(P_ped, [175, 200])
            P_ped = P_ped[mask]
            rm = rm[:, mask]
            print(f"  [Filter] Bridge 2: Removed 175, 200. Max P: {P_ped.max()}")

        # Limit to 50 simulations max for paper figures
        if rm.shape[0] > 50:
             print(f"  [Limit] {bridge_name}: Truncating from {rm.shape[0]} to 50 simulations.")
             rm = rm[:50, :]

        # Process stats
        medians = []
        p16s = []
        p84s = []
        fragility = []
        cleaned_data = []

        for i in range(len(P_ped)):
            col = rm[:, i]
            vals = col[~np.isnan(col)]
            cleaned_data.append(vals)

            if len(vals) > 0:
                medians.append(np.median(vals))
                p16s.append(np.percentile(vals, 16))
                p84s.append(np.percentile(vals, 84))
                
                # Standard Lognormal Fragility calculation moved to calculate_mom_fragility
                # to support multiple delta values on demand.
            else:
                medians.append(np.nan)
                p16s.append(np.nan)
                p84s.append(np.nan)

        data_store[bridge_name] = {
            'P_ped': P_ped,
            'cleaned_data': cleaned_data,
            'medians': np.array(medians),
            'p16s': np.array(p16s),
            'p84s': np.array(p84s)
        }
        
    return data_store

def plot_fig1_scatter(data_store):
    print("Generating Figure 1: Combined Scatter...")
    # 3x2 Grid
    fig, axes = plt.subplots(3, 2, figsize=(12, 14), constrained_layout=True)
    axes_flat = axes.flatten()

    # Subplots 1-5: Individual Bridges
    for b in range(1, 6):
        ax = axes_flat[b-1]
        bridge_name = f'bridge{b}'
        
        if bridge_name not in data_store:
            ax.axis('off')
            continue
            
        data = data_store[bridge_name]
        P_ped = data['P_ped']
        
        # Scatter points
        for i, vals in enumerate(data['cleaned_data']):
            x_vals = np.full(len(vals), P_ped[i])
            ax.scatter(x_vals, vals, c='gray', s=10, alpha=0.2, edgecolors='none')
            
        # Lines (Median, P16, P84)
        # Prepend 0 for start at origin
        x_plot = np.concatenate(([0], P_ped))
        y_med = np.concatenate(([0], data['medians']))
        y_p16 = np.concatenate(([0], data['p16s']))
        y_p84 = np.concatenate(([0], data['p84s']))
        
        ax.plot(x_plot, y_med, color=COLORS_BRIDGES[1], linewidth=2.5, label='Median') # Red
        ax.plot(x_plot, y_p16, color='black', linestyle='--', linewidth=1.5, label='16th and 84th percentiles')
        ax.plot(x_plot, y_p84, color='black', linestyle='--', linewidth=1.5)
        
        ax.set_title(f'Bridge {b}')
        ax.set_xlim(0, 200)
        
        if b == 4:
            ax.set_ylim(0, 1.0)
        else:
            ax.set_ylim(0, 0.6)
            
        ax.grid(True)
        
        if b >= 5: ax.set_xlabel('Crowd Size (P)')
        if (b-1) % 2 == 0: ax.set_ylabel('Peak Acceleration (m/s$^2$)')
        
        if b == 1:
            ax.legend(loc='upper left')

    # Subplot 6: Comparison of Medians
    ax6 = axes_flat[5]
    for b in range(1, 6):
        bridge_name = f'bridge{b}'
        if bridge_name not in data_store: continue
        
        data = data_store[bridge_name]
        x_plot = np.concatenate(([0], data['P_ped']))
        y_med = np.concatenate(([0], data['medians']))
        
        ax6.plot(x_plot, y_med, '-o', color=COLORS_BRIDGES[b-1], linewidth=2, markersize=4,
                 label=f'Bridge {b}')
                 
    ax6.set_title('Medians')
    ax6.grid(True)
    ax6.set_xlim(0, 200)
    ax6.set_ylim(bottom=0)
    ax6.set_xlabel('Crowd Size (P)')
    ax6.legend(loc='upper left')
    
    save_figure(fig, 'fig1_combined_scatter_medians')

def plot_fragility_comparison(data_store, bridge_indices, filename_suffix, delta):
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for b in bridge_indices:
        bridge_name = f'bridge{b}'
        if bridge_name not in data_store: continue
        
        d = data_store[bridge_name]
        # Calculate fragility for this specific delta
        fragility = calculate_mom_fragility(d['cleaned_data'], delta)
        
        ax.plot(d['P_ped'], fragility, '-o', color=COLORS_BRIDGES[b-1], linewidth=2.5,
                label=f'Bridge {b}')
                
    # ax.set_title(f'Fragility Curves ($\\delta_{{max}} = {delta}$ m/s$^2$)')
    ax.set_xlabel('Crowd Size (P)')
    ax.set_ylabel(r'$\mathbb{P}(\Delta > ' + str(delta) + r' \mid \Pi = P)$')
    ax.grid(True)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    save_figure(fig, filename_suffix)

def plot_fig4_mle_ls(data_store, delta, filename_suffix):
    print(f"Generating Figure 4: Bridge 3 MLE vs LS (delta={delta})...")
    b_idx = 3
    bridge_name = f'bridge{b_idx}'
    if bridge_name not in data_store: return
    
    d = data_store[bridge_name]
    x_j = d['P_ped']
    # Explicitly compute counts and exceedances for MLE
    n_j = []
    z_j = []
    
    for vals in d['cleaned_data']:
        n = len(vals)
        z = np.sum(vals > delta)
        n_j.append(n)
        z_j.append(z)
        
    n_j = np.array(n_j)
    z_j = np.array(z_j)
    fractions_obs = z_j / n_j
    
    # Define likelihood and LS functions
    # Params: [mu_ln, sigma_ln]
    
    def loglik(params):
        mu, sigma = params
        if sigma <= 0: return 1e10
        # p = Phi( (ln(x) - mu) / sigma )
        # Avoid log(0) issues by adding eps inside logs if needed, but x_j are > 0
        denom = sigma
        numer = np.log(x_j) - mu
        arg = numer / denom
        p = norm.cdf(arg)
        
        # Clip p to avoid log(0)
        p = np.clip(p, 1e-9, 1 - 1e-9)
        
        # Negative Log Likelihood
        L = np.sum(z_j * np.log(p) + (n_j - z_j) * np.log(1 - p))
        return -L

    def least_squares(params):
        mu, sigma = params
        if sigma <= 0: return 1e10
        p_est = norm.cdf((np.log(x_j) - mu) / sigma)
        return np.sum((fractions_obs - p_est)**2)

    # Initial guess
    initial_guess = [np.log(120), 0.2]
    
    # Optimize MLE
    res_mle = optimize.minimize(loglik, initial_guess, method='Nelder-Mead')
    mu_mle, sigma_mle = res_mle.x
    
    # Optimize LS
    res_ls = optimize.minimize(least_squares, initial_guess, method='Nelder-Mead')
    mu_ls, sigma_ls = res_ls.x
    
    print(f"Bridge 3 MLE: mu={mu_mle:.4f}, sigma={sigma_mle:.4f}")
    print(f"Bridge 3 LS:  mu={mu_ls:.4f}, sigma={sigma_ls:.4f}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Monte Carlo
    ax.plot(x_j, fractions_obs, 'o', color='#a2191f', markersize=8,
            label='Monte Carlo')
            
    # Smooth curves
    x_smooth = np.linspace(x_j.min(), x_j.max(), 200)
    
    p_mle_smooth = norm.cdf((np.log(x_smooth) - mu_mle) / sigma_mle)
    ax.plot(x_smooth, p_mle_smooth, '-', color='#da1e28', linewidth=3,
            label=fr'MLE: $\mu_{{\ln}}={mu_mle:.2f}, \sigma_{{\ln}}={sigma_mle:.2f}$')
            
    # LS Fit removed from plot
            
    # ax.set_title('Bridge 3: MLE vs Least Squares Fit')
    ax.set_xlabel('Crowd Size (P)')
    ax.set_ylabel(r'$\mathbb{P}(\Delta > ' + str(delta) + r' \mid \Pi = P)$')
    ax.grid(True)
    ax.legend(loc='upper left')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    save_figure(fig, filename_suffix)
    
def plot_mle_curves(data_store, target_bridges, filename_suffix, delta):
    print(f"Generating MLE Curves {target_bridges} (delta={delta})...")
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for b_idx in target_bridges:
        bridge_name = f'bridge{b_idx}'
        if bridge_name not in data_store: continue
        
        d = data_store[bridge_name]
        x_j = d['P_ped']
        
        n_j = []
        z_j = []
        for vals in d['cleaned_data']:
            n_j.append(len(vals))
            z_j.append(np.sum(vals > delta))
        n_j = np.array(n_j)
        z_j = np.array(z_j)
        
        def loglik(params):
            mu, sigma = params
            if sigma <= 0: return 1e10
            p = norm.cdf((np.log(x_j) - mu) / sigma)
            p = np.clip(p, 1e-9, 1 - 1e-9)
            L = np.sum(z_j * np.log(p) + (n_j - z_j) * np.log(1 - p))
            return -L
            
        # Initial guess logic similar to MATLAB
        # Find first non-zero z
        idx_nz = np.where(z_j > 0)[0]
        if len(idx_nz) > 0:
            start_p = x_j[idx_nz[0]]
            # small offset guess
            mu_guess = np.log(start_p + 20)
        else:
            mu_guess = np.log(100)
            
        initial_guess = [mu_guess, 0.3]
        res = optimize.minimize(loglik, initial_guess, method='Nelder-Mead')
        mu, sigma = res.x
        
        # Plot
        x_smooth = np.linspace(0, 200, 300)
        p_smooth = norm.cdf((np.log(x_smooth) - mu) / sigma)
        
        ax.plot(x_smooth, p_smooth, '-', color=COLORS_BRIDGES[b_idx-1], linewidth=3,
                label=f'Bridge {b_idx}')
                
    
    # ax.set_title('Fragility Curves (MLE Fitted)')
    ax.set_xlabel('Crowd Size (P)')
    ax.set_ylabel(r'$\mathbb{P}(\Delta > ' + str(delta) + r' \mid \Pi = P)$')
    ax.grid(True)
    ax.legend(loc='lower right')
    ax.set_xlim(left=0, right=200)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    save_figure(fig, filename_suffix)

# --- Main ---
def main():
    data = load_data()
    
    # Figure 1
    plot_fig1_scatter(data)
    
    for delta in DELTA_VALUES:
        suffix = str(delta).replace('.', 'p')
        print(f"\n--- Processing for Delta = {delta} ---")
        
        # Figure 2: B1, B2, B3
        plot_fragility_comparison(data, [1, 2, 3], f'fig2_fragility_B123_d{suffix}', delta)
        
        # Figure 3: B2, B4, B5
        plot_fragility_comparison(data, [2, 4, 5], f'fig3_fragility_B245_d{suffix}', delta)
        
        # Figure 4: Bridge 3 Fits
        plot_fig4_mle_ls(data, delta, f'fig4_bridge3_mle_vs_ls_d{suffix}')
        
        # Figure 5: MLE Curves only
        plot_mle_curves(data, [1, 2, 3], f'fig5_bridges123_mle_curves_d{suffix}', delta)
        plot_mle_curves(data, [2, 4, 5], f'fig5_bridges245_mle_curves_d{suffix}', delta)
    
    print(f"All figures saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
