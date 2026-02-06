import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib import cm
import matplotlib.ticker as mtick
import h5py

# --- Configuration ---
DELTA_VALUES = [0.15, 0.21]  # m/s^2
P_TARGET = 100
DELTA_MAX_RANGE = np.arange(0.01, 0.51, 0.01)  # 0.01 to 0.50 inclusive
SURFACE_COLORMAP = "RdYlBu_r"

# --- Aesthetics ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 1.5

# Colors
COLOR_GRAY_LIGHT = '#c1c7cd'
COLOR_GRAY_BIN = '#a2a9b0'
COLOR_BLACK = '#000000'
COLOR_BLUE_PAPER = '#0f62fe'
COLOR_GRAY_DARK = '#333333'
COLOR_RED_PAPER = '#da1e28'

# Colorblind-friendly palette
COLORS_CB_PALETTE = ['#0f62fe', '#da1e28', '#697077']

# --- Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
OUTPUT_BASE = os.path.join(REPO_ROOT, 'assets', 'figures')
OUTPUT_PNG = os.path.join(OUTPUT_BASE, 'png')
OUTPUT_PDF = os.path.join(OUTPUT_BASE, 'pdf')

for d in [OUTPUT_PNG, OUTPUT_PDF]:
    os.makedirs(d, exist_ok=True)

def save_figure(fig, name):
    """Saves figure to png and pdf"""
    png_path = os.path.join(OUTPUT_PNG, f"{name}.png")
    pdf_path = os.path.join(OUTPUT_PDF, f"{name}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"Saved {name}")
    plt.close(fig) # Close figure to free memory

def process_bridge(bridge_num):
    print(f"\n=========================================")
    print(f"  PROCESSING BRIDGE {bridge_num}")
    print(f"=========================================\n")
    
    output_bridge = f"bridge{bridge_num}"
    data_path = os.path.join(REPO_ROOT, 'results', 'main_analysis', output_bridge, 'results_2000s.mat')

    if not os.path.exists(data_path):
        print(f"Warning: Data file not found at: {data_path}. Skipping Bridge {bridge_num}.")
        return

    print(f"Loading data from {data_path}")
    
    try:
        mat_data = scipy.io.loadmat(data_path)
        results_matrix = mat_data['resultsMatrix']
        p_ped = mat_data['P_ped'].flatten()
    except NotImplementedError:
        print("  MATLAB v7.3 file detected. Using h5py.")
        with h5py.File(data_path, 'r') as f:
            p_ped = np.array(f['P_ped']).flatten()
            # h5py reads MATLAB matrices transposed
            results_matrix = np.array(f['resultsMatrix'])

    # Validate and Fix Dimensions
    # We expect (Sims, Peds)
    n_peds = len(p_ped)
    r, c = results_matrix.shape
    
    if c == n_peds and r != n_peds:
        pass # OK
    elif r == n_peds and c != n_peds:
        print(f"  Transposing results matrix from {r}x{c} to {c}x{r}")
        results_matrix = results_matrix.T
    else:
        print(f"  Warning: Ambiguous dimensions {r}x{c} vs P={n_peds}. Assuming current orientation is correct if (Sims, P).")

    # --- BRIDGE 2 SPECIAL FILTERING ---
    if bridge_num == 2:
        print("  Applying Bridge 2 constraints: Removing P=175 and P=200 due to data issues.")
        # Identify indices to keep
        # Check against floating point or integer values depending on what p_ped contains
        # Using np.isclose just in case, or straight comparison if integers
        keep_indices = ~np.isin(p_ped, [175, 200])
        
        removed_count = np.sum(~keep_indices)
        if removed_count > 0:
            p_ped = p_ped[keep_indices]
            results_matrix = results_matrix[:, keep_indices]
            print(f"  Removed {removed_count} crowd size columns. Remaining P_ped: {p_ped}")
        else:
            print("  Warning: Targeted P values (175, 200) not found in Bridge 2 data.")

    # Limit to 50 simulations max for paper figures
    if results_matrix.shape[0] > 50:
        print(f"Limiting analysis to first 50 simulations (Original: {results_matrix.shape[0]})")
        results_matrix = results_matrix[:50, :]

    # --- Figure 1: Boxplot ---
    print("Generating Figure 1: Boxplot...")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    data_for_boxplot = []
    labels = []
    for i in range(results_matrix.shape[1]):
        col = results_matrix[:, i]
        col = col[~np.isnan(col)]
        data_for_boxplot.append(col)
        labels.append(str(int(p_ped[i])))

    boxprops = dict(facecolor=COLOR_GRAY_LIGHT, color='black')
    medianprops = dict(color=COLOR_RED_PAPER, linewidth=1.5)
    flierprops = dict(marker='+', markeredgecolor='black')

    ax1.boxplot(data_for_boxplot, labels=labels, patch_artist=True,
                boxprops=boxprops, medianprops=medianprops, flierprops=flierprops)

    ax1.set_xlabel('Crowd Size (P)')
    ax1.set_ylabel('Peak Lateral Acceleration (m/s$^2$)')
    ax1.yaxis.grid(True, linestyle='-', alpha=0.3)
    ax1.set_axisbelow(True)

    save_figure(fig1, f"boxplot_peak_accel_{output_bridge}")

    # --- Figure 2: Scatter (Raw Results) ---
    print("Generating Figure 2: Scatter...")
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    medians = []
    p16s = []
    p84s = []

    for i, p_val in enumerate(p_ped):
        col = results_matrix[:, i]
        col = col[~np.isnan(col)]
        
        # Jitter/Scatter
        x_vals = np.full_like(col, i + 1)
        ax2.scatter(x_vals, col, color='k', s=20, alpha=0.15, edgecolors='none')
        
        medians.append(np.median(col))
        p16s.append(np.percentile(col, 16))
        p84s.append(np.percentile(col, 84))

    # Prepend 0
    medians = [0] + medians
    p16s = [0] + p16s
    p84s = [0] + p84s
    x_axis = np.concatenate(([0], np.arange(1, len(p_ped) + 1)))

    ax2.plot(x_axis, medians, color=COLOR_RED_PAPER, linewidth=2.5, label='Median')
    ax2.plot(x_axis, p16s, 'k--', linewidth=1.5, label='16th/84th Percentiles')
    ax2.plot(x_axis, p84s, 'k--', linewidth=1.5)

    ax2.set_xticks(x_axis)
    xtick_labels = ['0'] + [str(int(p)) for p in p_ped]
    ax2.set_xticklabels(xtick_labels)
    ax2.set_xlabel('Crowd Size (P)')
    ax2.set_ylabel('Peak Lateral Acceleration (m/s$^2$)')
    ax2.set_xlim(left=0)
    ax2.set_ylim(0, 0.5)
    ax2.yaxis.grid(True)
    ax2.xaxis.grid(False)
    ax2.legend(loc='upper right')

    save_figure(fig2, f"scatter_raw_results_{output_bridge}")

    # --- Figure 3: Histogram for Target P ---
    print("Generating Figure 3: Histogram...")
    # Find nearest P
    idx = (np.abs(p_ped - P_TARGET)).argmin()
    p_actual = p_ped[idx]
    data_col = results_matrix[:, idx]
    data_col = data_col[~np.isnan(data_col)]

    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.hist(data_col, bins=20, density=True, facecolor=COLOR_GRAY_BIN, alpha=0.7, 
             edgecolor='none', label=f'Monte Carlo (P = {int(p_actual)})')

    # Lognormal Fit
    fit_params = scipy.stats.lognorm.fit(data_col, floc=0) 
    s_fit, loc_fit, scale_fit = fit_params
    mu_ln_manual = np.mean(np.log(data_col))
    sigma_ln_manual = np.std(np.log(data_col))

    x_min = max(0.001, np.min(data_col) * 0.8)
    x_max = np.max(data_col) * 1.2
    x_fit = np.linspace(x_min, x_max, 200)

    p_fit = (1 / (x_fit * sigma_ln_manual * np.sqrt(2 * np.pi))) * np.exp(- (np.log(x_fit) - mu_ln_manual)**2 / (2 * sigma_ln_manual**2))

    ax3.plot(x_fit, p_fit, color=COLOR_BLACK, linewidth=2.5, 
             label=f'Lognormal Fit ($\\mu_{{ln}}={mu_ln_manual:.2f}$, $\\sigma_{{ln}}={sigma_ln_manual:.2f}$)')

    ax3.set_xlabel('Peak Lateral Acceleration (m/s^2)')
    ax3.set_ylabel('Probability Density')
    ax3.legend(loc='upper right')
    ax3.grid(True)

    save_figure(fig3, f"histogram_distribution_P{int(P_TARGET)}_{output_bridge}")

    # --- Figure 4: Fragility Curves ---
    print("Generating Figure 4: Fragility Curves...")
    fig_comb, ax_comb = plt.subplots(figsize=(10, 7))
    delta_strs = []

    for k, delta_val in enumerate(DELTA_VALUES):
        d_str = f"{int(round(delta_val * 100)):03d}"
        delta_strs.append(d_str)
        
        p_exceed_ln = []
        p_exceed_emp = []
        
        for i in range(len(p_ped)):
            data = results_matrix[:, i]
            data = data[~np.isnan(data)]
            
            # Empirical
            emp_prob = np.sum(data > delta_val) / len(data)
            p_exceed_emp.append(emp_prob)
            
            # Analytic
            mu = np.mean(np.log(data))
            sigma = np.std(np.log(data))
            ln_prob = 1 - norm.cdf(np.log(delta_val), loc=mu, scale=sigma)
            p_exceed_ln.append(ln_prob)
            
        # Individual plot - REMOVED as per user request
        # fig_ind, ax_ind = plt.subplots(figsize=(10, 6))
        # ax_ind.plot(p_ped, p_exceed_ln, '-', color=COLOR_RED_PAPER, linewidth=3, marker='o', markersize=10, label='Lognormal Fit')
        # ax_ind.plot(p_ped, p_exceed_emp, 's--', color=COLOR_GRAY_DARK, linewidth=2, marker='x', markersize=10, label='Monte Carlo')
        
        # ax_ind.set_xlabel('Crowd Size (P)')
        # ax_ind.set_ylabel(f'P($\\Delta > {delta_val:.2f} m/s^2 | \\Pi = P$)')
        # ax_ind.legend(loc='lower right')
        # ax_ind.grid(True)
        # ax_ind.set_ylim(0, 1.05)
        # ax_ind.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        # save_figure(fig_ind, f"fragility_analysis_{output_bridge}_{d_str}")
        
        # Combined plot
        c = COLORS_CB_PALETTE[min(k, len(COLORS_CB_PALETTE) - 1)]
        ax_comb.plot(p_ped, p_exceed_ln, '-', color=c, linewidth=2.5, marker='o', markersize=7, 
                     label=f'Lognormal Fit $\\delta_{{max}} = {delta_val:.2f}$ m/s$^2$')
        ax_comb.plot(p_ped, p_exceed_emp, '--', color=c, linewidth=1.5, marker='x', markersize=7, 
                     label=f'Monte Carlo')

    ax_comb.set_xlabel('Crowd Size (P)')
    ax_comb.set_ylabel(r'$\mathbb{P}(\Delta \geq \delta_{max} \mid \Pi = P)$')
    ax_comb.legend(loc='best', fontsize=11)
    ax_comb.grid(True)
    ax_comb.set_ylim(0, 1.05)
    ax_comb.set_xlim(left=0) # Start x-axis from 0
    ax_comb.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    combined_suffix = "_".join(delta_strs)
    save_figure(fig_comb, f"fragility_analysis_{output_bridge}_combined{combined_suffix}")

    # --- Figure 5: Fragility Surface ---
    print("Generating Figure 5: Fragility Surface...")
    if len(p_ped) > 1:
        X, Y = np.meshgrid(p_ped, DELTA_MAX_RANGE)
        Z = np.zeros_like(X, dtype=float)

        for i in range(len(p_ped)):
            data = results_matrix[:, i]
            data = data[~np.isnan(data)]
            mu = np.mean(np.log(data))
            sigma = np.std(np.log(data))
            col_deltas = Y[:, i]
            Z[:, i] = 1 - norm.cdf(np.log(col_deltas), loc=mu, scale=sigma)

        Z = np.clip(Z, 0, 1)

        fig_surf = plt.figure(figsize=(10, 10))
        ax_surf = fig_surf.add_subplot(111, projection='3d')
        surf = ax_surf.plot_surface(X, Y, Z, cmap=SURFACE_COLORMAP, shade=True, antialiased=True)
        ax_surf.view_init(elev=30, azim=65)
        fig_surf.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.95)
        ax_surf.dist = 11
        ax_surf.set_xlabel('Crowd Size (P)', labelpad=10)
        ax_surf.set_ylabel('Peak Lateral Accel. $\\delta_{max}$ (m/s$^2$)', labelpad=10)
        ax_surf.set_zlabel(r'$\mathbb{P}(\Delta \geq \delta_{max} \mid \Pi = P)$', labelpad=10)
        # Colorbar at bottom, matching figure width
        cbar_ax = fig_surf.add_axes([0.15, 0.05, 0.7, 0.025])
        cbar = fig_surf.colorbar(surf, cax=cbar_ax, orientation='horizontal')
        cbar.set_label(r'$\mathbb{P}(\Delta \geq \delta_{max} \mid \Pi = P)$')
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(size=0)
        save_figure(fig_surf, f"fragility_surface_{output_bridge}")

        # --- Figure 5c: Interpolated ---
        print("Generating Figure 5c: Fragility Surface (interpolated)...")
        from scipy.interpolate import RegularGridInterpolator
        p_ped_fine = np.linspace(p_ped.min(), p_ped.max(), 200)
        delta_fine = np.linspace(DELTA_MAX_RANGE.min(), DELTA_MAX_RANGE.max(), 200)
        interpolator = RegularGridInterpolator((DELTA_MAX_RANGE, p_ped), Z, method='linear', bounds_error=False, fill_value=None)
        
        X_fine, Y_fine = np.meshgrid(p_ped_fine, delta_fine)
        points = np.stack([Y_fine.ravel(), X_fine.ravel()], axis=-1)
        Z_fine = interpolator(points).reshape(X_fine.shape)
        Z_fine = np.clip(Z_fine, 0, 1)

        fig_surf_interp = plt.figure(figsize=(10, 10))
        ax_surf_interp = fig_surf_interp.add_subplot(111, projection='3d')
        
        cmap = plt.get_cmap(SURFACE_COLORMAP)
        norm_colors = plt.Normalize(0, 1)
        # Facecolors for finer shading
        Z_faces_interp = (Z_fine[:-1, :-1] + Z_fine[1:, :-1] + Z_fine[:-1, 1:] + Z_fine[1:, 1:]) / 4
        facecolors_interp = cmap(norm_colors(Z_faces_interp))
        
        ax_surf_interp.plot_surface(X_fine, Y_fine, Z_fine, facecolors=facecolors_interp, shade=False, antialiased=True)
        ax_surf_interp.view_init(elev=30, azim=65)
        fig_surf_interp.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.95)
        ax_surf_interp.dist = 11
        ax_surf_interp.set_xlabel('Crowd Size (P)', labelpad=10)
        ax_surf_interp.set_ylabel('Peak Lateral Accel. $\\delta_{max}$ (m/s$^2$)', labelpad=10)
        ax_surf_interp.set_zlabel(r'$\mathbb{P}(\Delta \geq \delta_{max} \mid \Pi = P)$', labelpad=10)
        sm_interp = cm.ScalarMappable(cmap=cmap, norm=norm_colors)
        sm_interp.set_array([])
        # Colorbar at bottom, matching figure width
        cbar_ax_interp = fig_surf_interp.add_axes([0.15, 0.05, 0.7, 0.025])
        cbar_interp = fig_surf_interp.colorbar(sm_interp, cax=cbar_ax_interp, orientation='horizontal')
        cbar_interp.set_label(r'$\mathbb{P}(\Delta \geq \delta_{max} \mid \Pi = P)$')
        cbar_interp.outline.set_visible(False)
        cbar_interp.ax.tick_params(size=0)
        save_figure(fig_surf_interp, f"fragility_surface_{output_bridge}_interpolated")

        # --- Figure 6: Contour Plot ---
        print("Generating Figure 6: Contour Plot...")
        fig_cont, ax_cont = plt.subplots(figsize=(10, 8))
        contour_levels = np.linspace(0.0, 1.0, 11)
        contour = ax_cont.contourf(X, Y, Z, contour_levels, cmap=SURFACE_COLORMAP)
        ax_cont.contour(X, Y, Z, contour_levels[1:-1], colors='w', linewidths=0.5)
        # Colorbar at bottom, matching figure width
        fig_cont.subplots_adjust(bottom=0.18)
        cbar_ax_cont = fig_cont.add_axes([0.125, 0.06, 0.775, 0.025])
        cbar_cont = fig_cont.colorbar(contour, cax=cbar_ax_cont, orientation='horizontal')
        cbar_cont.set_label(r'$\mathbb{P}(\Delta \geq \delta_{max} \mid \Pi = P)$')
        cbar_cont.outline.set_visible(False)
        cbar_cont.ax.tick_params(size=0)
        ax_cont.set_xlabel('Crowd Size (P)')
        ax_cont.set_ylabel('Peak Lateral Accel. $\\delta_{max}$ (m/s$^2$)')
        ax_cont.set_xticks([5, 50, 100, 150, 200])
        ax_cont.grid(True)
        
        # Reference lines
        from scipy.interpolate import interp1d
        delta_target = 0.21
        prob_target = 0.10
        delta_idx = (np.abs(DELTA_MAX_RANGE - delta_target)).argmin()
        probs_at_delta = Z[delta_idx, :]
        
        if probs_at_delta[0] < prob_target < probs_at_delta[-1]:
            interp_func = interp1d(probs_at_delta, p_ped, kind='linear', fill_value='extrapolate')
            P_at_target = float(interp_func(prob_target))
        else:
             nearest_idx = (np.abs(probs_at_delta - prob_target)).argmin()
             P_at_target = p_ped[nearest_idx]
             
        P_at_target = np.clip(P_at_target, p_ped.min(), p_ped.max())
        line_color = 'white'
        ax_cont.hlines(y=delta_target, xmin=p_ped.min(), xmax=P_at_target, 
                       colors=line_color, linestyles='dashed', linewidth=2.5)
        ax_cont.vlines(x=P_at_target, ymin=DELTA_MAX_RANGE.min(), ymax=delta_target,
                       colors=line_color, linestyles='dashed', linewidth=2.5)
        ax_cont.plot(P_at_target, delta_target, 'o', color=line_color, markersize=7, zorder=5)
        
        ax_cont.text(p_ped.min() + 3, delta_target + 0.012, f'$\\delta_{{max}} = {delta_target}$', 
                     fontsize=16, color=line_color, va='bottom', ha='left', weight='bold')
        ax_cont.text(P_at_target + 2, DELTA_MAX_RANGE.min() + 0.015, f'P = {P_at_target:.0f}', 
                     fontsize=16, color=line_color, va='bottom', ha='left', weight='bold')
        save_figure(fig_cont, f"fragility_contours_{output_bridge}")

    # --- Figure 7: Fragility Convergence (Subplots) --- REMOVED
    # print("Generating Figure 7: Fragility Convergence (Subplots)...")
    # ... code removed ...
    # save_figure(fig_conv, f"fragility_convergence_subplots_{output_bridge}")

    # --- Figure 8: Fragility Convergence (Combined) ---
    print("Generating Figure 8: Fragility Convergence (Combined)...")
    # Define variables needed for Figure 8 that were previously in Figure 7 block
    delta_convergence = 0.21
    sim_counts = [10, 30, 50]
    fig_conv_comb, ax_conv_comb = plt.subplots(figsize=(10, 7))
    conv_colors = ['#9ec5fe', '#0f62fe', '#001d6c']

    for idx, n_sims in enumerate(sim_counts):
        if n_sims > results_matrix.shape[0]:
            n_sims_curr = results_matrix.shape[0]
        else:
            n_sims_curr = n_sims
        subset_data = results_matrix[:n_sims_curr, :]
        p_exceed_ln_sub = []
        for i in range(len(p_ped)):
            col_data = subset_data[:, i]
            col_data = col_data[~np.isnan(col_data)]
            col_data_clean = col_data[col_data > 0]
            if len(col_data_clean) < 2:
                 mu = np.mean(np.log(col_data_clean)) if len(col_data_clean) > 0 else 0
                 sigma = 0.001
            else:
                mu = np.mean(np.log(col_data_clean))
                sigma = np.std(np.log(col_data_clean))
            ln_prob = 1 - norm.cdf(np.log(delta_convergence), loc=mu, scale=sigma)
            p_exceed_ln_sub.append(ln_prob)
        ax_conv_comb.plot(p_ped, p_exceed_ln_sub, '-', color=conv_colors[idx], linewidth=3, 
                          label=f'N = {n_sims_curr} Simulations')

    ax_conv_comb.set_xlabel('Crowd Size (P)')
    ax_conv_comb.set_ylabel(r'$\mathbb{P}(\Delta \geq \delta_{max} \mid \Pi = P)$')
    ax_conv_comb.legend(loc='best', fontsize=12)
    ax_conv_comb.grid(True)
    ax_conv_comb.set_ylim(0, 1.05)
    ax_conv_comb.set_xlim(left=0) # Start x-axis from 0
    ax_conv_comb.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    save_figure(fig_conv_comb, f"fragility_convergence_combined_{output_bridge}")

    # --- Figure 9: Logarithmic Standard Deviation ---
    print("Generating Figure 9: Logarithmic Standard Deviation...")
    fig_log_sigma, ax_log_sigma = plt.subplots(figsize=(10, 6))

    sigmas_ln = []
    plot_p = []

    for i in range(len(p_ped)):
        col = results_matrix[:, i]
        # Filter NaNs
        col = col[~np.isnan(col)]
        # Filter <= 0 for log
        col = col[col > 0]
        
        if len(col) > 1:
            # standard deviation of the log(data)
            sigma_val = np.std(np.log(col))
            sigmas_ln.append(sigma_val)
            plot_p.append(p_ped[i])
        else:
            # Not enough data points
            pass

    ax_log_sigma.plot(plot_p, sigmas_ln, '-o', color=COLOR_BLUE_PAPER, linewidth=2.5, markersize=8)

    ax_log_sigma.set_xlabel('Crowd Size (P)')
    ax_log_sigma.set_ylabel('Logarithmic Standard Deviation ($\\zeta$)')
    ax_log_sigma.grid(True)
    ax_log_sigma.set_xlim(left=0)
    ax_log_sigma.set_ylim(bottom=0)

    save_figure(fig_log_sigma, f"log_standard_deviation_{output_bridge}")

    # --- Figure 10: Standard Deviation ---
    print("Generating Figure 10: Standard Deviation...")
    fig_sigma, ax_sigma = plt.subplots(figsize=(10, 6))

    sigmas = []
    plot_p_sigma = []

    for i in range(len(p_ped)):
        col = results_matrix[:, i]
        # Filter NaNs
        col = col[~np.isnan(col)]
        
        if len(col) > 1:
            # standard deviation of the data
            sigma_val = np.std(col)
            sigmas.append(sigma_val)
            plot_p_sigma.append(p_ped[i])
        else:
            # Not enough data points
            pass

    ax_sigma.plot(plot_p_sigma, sigmas, '-o', color=COLOR_BLUE_PAPER, linewidth=2.5, markersize=8)

    ax_sigma.set_xlabel('Crowd Size (P)')
    ax_sigma.set_ylabel('Standard Deviation ($\\sigma$)')
    ax_sigma.grid(True)
    ax_sigma.set_xlim(left=0)
    ax_sigma.set_ylim(bottom=0)

    save_figure(fig_sigma, f"standard_deviation_{output_bridge}")

    # --- Store Data for Combined Plots ---
    GLOBAL_DATA_STORE[bridge_num] = {
        'p_ped': p_ped,
        'results_matrix': results_matrix,
        'X': X if 'X' in locals() else None,
        'Y': Y if 'Y' in locals() else None,
        'Z': Z if 'Z' in locals() else None,
        'sigmas': sigmas,
        'plot_p_sigma': plot_p_sigma,
        'sigmas_ln': sigmas_ln,
        'plot_p_ln': plot_p,
        'delta_max_range': DELTA_MAX_RANGE
    }

def plot_combined_figures():
    print("\nGenerating Combined Figures...")
    
    # --- 1. Combined Contours (3x2 Grid) ---
    print("Generating Combined Contours...")
    fig_cont_comb, axes = plt.subplots(3, 2, figsize=(12, 14))
    axes_flat = axes.flatten()
    
    contour_levels = np.linspace(0.0, 1.0, 11)
    
    for b_idx in range(1, 6):
        ax = axes_flat[b_idx-1]
        if b_idx not in GLOBAL_DATA_STORE:
            ax.axis('off')
            continue
            
        data = GLOBAL_DATA_STORE[b_idx]
        X, Y, Z = data['X'], data['Y'], data['Z']
        
        contour = ax.contourf(X, Y, Z, contour_levels, cmap=SURFACE_COLORMAP)
        
        ax.set_title(f'Bridge {b_idx}', fontsize=14, fontweight='bold')
        ax.set_xticks([5, 50, 100, 150, 200])
        ax.grid(True, alpha=0.3)
        
        # Add x-label only on bottom row (bridges 5 and the empty 6th)
        if b_idx >= 5:
            ax.set_xlabel('Crowd Size (P)')
        # Add y-label only on left column (bridges 1, 3, 5)
        if (b_idx-1) % 2 == 0:
            ax.set_ylabel('Peak Accel. $\\delta_{max}$ (m/s$^2$)')
    
    # Use the 6th subplot for a professional colorbar
    ax_cbar = axes_flat[5]
    ax_cbar.axis('off')
    
    # Create a horizontal colorbar in the 6th subplot space
    # Get the position of the 6th subplot and place a horizontal colorbar there
    plt.tight_layout()
    pos = ax_cbar.get_position()
    
    # Create a centered horizontal colorbar within this space
    cbar_width = pos.width * 0.8
    cbar_height = 0.02
    cbar_left = pos.x0 + (pos.width - cbar_width) / 2
    cbar_bottom = pos.y0 + pos.height * 0.45
    
    cbar_ax = fig_cont_comb.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
    cbar = fig_cont_comb.colorbar(contour, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(r'$\mathbb{P}(\Delta \geq \delta_{max} \mid \Pi = P)$', fontsize=13)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(size=0, labelsize=11)
    
    save_figure(fig_cont_comb, "combined_contours_all_bridges")
    
    # --- 2. Combined Surfaces (3x2 Grid) ---
    print("Generating Combined Surfaces...")
    fig_surf_comb = plt.figure(figsize=(11, 12))
    
    # Reduce spacing between subplots for a more compact layout
    fig_surf_comb.subplots_adjust(left=0.0, right=1.0, bottom=0.06, top=0.96, wspace=-0.1, hspace=0.08)
    
    for b_idx in range(1, 6):
        if b_idx not in GLOBAL_DATA_STORE: continue
        
        # 3x2 grid, index b_idx
        ax = fig_surf_comb.add_subplot(3, 2, b_idx, projection='3d')
        
        data = GLOBAL_DATA_STORE[b_idx]
        X, Y, Z = data['X'], data['Y'], data['Z']
        
        surf = ax.plot_surface(X, Y, Z, cmap=SURFACE_COLORMAP, shade=True, antialiased=True)
        ax.view_init(elev=30, azim=65)
        ax.dist = 9  # Zoom in more to fill the space
        
        ax.set_title(f'Bridge {b_idx}', fontsize=14, fontweight='bold', pad=0)
        ax.set_xlabel('Crowd Size (P)', labelpad=0, fontsize=10)
        ax.set_ylabel('$\\delta_{max}$ (m/s$^2$)', labelpad=0, fontsize=10)
        ax.set_zlabel('Prob.', labelpad=0, fontsize=10)
        ax.tick_params(pad=0, labelsize=9)
    
    # Add colorbar in the 6th subplot position (bottom-right)
    # First add a dummy 2D axis for positioning reference
    ax_cbar_ref = fig_surf_comb.add_subplot(3, 2, 6)
    ax_cbar_ref.axis('off')
    
    pos = ax_cbar_ref.get_position()
    
    # Create a horizontal colorbar centered in the 6th subplot space
    cbar_width = pos.width * 0.7
    cbar_height = 0.02
    cbar_left = pos.x0 + (pos.width - cbar_width) / 2
    cbar_bottom = pos.y0 + pos.height * 0.45
    
    cbar_ax = fig_surf_comb.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
    cbar = fig_surf_comb.colorbar(surf, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(r'$\mathbb{P}(\Delta \geq \delta_{max} \mid \Pi = P)$', fontsize=12)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(size=0, labelsize=10)
    
    save_figure(fig_surf_comb, "combined_surfaces_all_bridges")
    
    # --- 3. Combined Standard Deviations (Single Plot) ---
    print("Generating Combined Standard Deviations...")
    fig_sd, ax_sd = plt.subplots(figsize=(10, 7))
    
    for b_idx in range(1, 6):
        if b_idx not in GLOBAL_DATA_STORE: continue
        
        data = GLOBAL_DATA_STORE[b_idx]
        p_vals = data['plot_p_sigma']
        sigmas = data['sigmas']
        
        color = COLORS_BRIDGES[b_idx-1]
        
        ax_sd.plot(p_vals, sigmas, '-o', color=color, linewidth=2.5, markersize=6, 
                   label=f'Bridge {b_idx}')
                   
    ax_sd.set_xlabel('Crowd Size (P)')
    ax_sd.set_ylabel('Standard Deviation ($\\sigma$)')
    ax_sd.grid(True)
    ax_sd.set_xlim(left=0)
    ax_sd.set_ylim(bottom=0)
    ax_sd.legend(loc='upper right')
    
    save_figure(fig_sd, "combined_standard_deviations")
    
    # --- 4. Combined Logarithmic Standard Deviations (Single Plot) ---
    print("Generating Combined Logarithmic Standard Deviations...")
    fig_lsd, ax_lsd = plt.subplots(figsize=(10, 7))
    
    for b_idx in range(1, 6):
        if b_idx not in GLOBAL_DATA_STORE: continue
        
        data = GLOBAL_DATA_STORE[b_idx]
        p_vals = data['plot_p_ln']
        sigmas_ln = data['sigmas_ln']
        
        color = COLORS_BRIDGES[b_idx-1]
        
        ax_lsd.plot(p_vals, sigmas_ln, '-o', color=color, linewidth=2.5, markersize=6, 
                   label=f'Bridge {b_idx}')
                   
    ax_lsd.set_xlabel('Crowd Size (P)')
    ax_lsd.set_ylabel('Logarithmic Standard Deviation ($\\zeta$)')
    ax_lsd.grid(True)
    ax_lsd.set_xlim(left=0)
    ax_lsd.set_ylim(bottom=0)
    ax_lsd.legend(loc='upper right')
    
    save_figure(fig_lsd, "combined_log_standard_deviations")


# --- Main Loop ---
if __name__ == "__main__":
    # Bridge Colors (matching MATLAB hex codes)
    COLORS_BRIDGES = [
        '#0f62fe', # Bridge 1 (Blue)
        '#da1e28', # Bridge 2 (Red)
        '#198038', # Bridge 3 (Green)
        '#8a3ffc', # Bridge 4 (Purple)
        '#ff832b', # Bridge 5 (Orange)
    ]
    
    GLOBAL_DATA_STORE = {}
    
    BRIDGES_TO_PROCESS = [1, 2, 3, 4, 5]
    for b in BRIDGES_TO_PROCESS:
        process_bridge(b)
        
    plot_combined_figures()

    print("\nAll bridges processed and combined figures generated successfully.")
