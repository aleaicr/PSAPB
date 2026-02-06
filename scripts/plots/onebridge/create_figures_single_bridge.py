# init
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.stats import norm
import os
import seaborn as sns

# Optional: Uncomment if running in a headless environment
# import matplotlib
# matplotlib.use('Agg')

# Set professional style for paper-ready figures
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams.update({
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
    "savefig.dpi": 300,
})

# Inputs
bridge_num = 3 # Bridge number to analyze
delta = 0.21 # m/s^2 (Fragility threshold)
P_target = 100 # Choose a pedestrian load to analyze in histogram
output_bridge = f'bridge{bridge_num}'

# Load data
data_path = f"../../../results/main_analysis/bridge{bridge_num}/results_2000s.mat"

# Path setup
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
repo_root = os.path.dirname(script_dir)
output_base = os.path.join(repo_root, "assets", "figures")
output_png = os.path.join(output_base, "png")
output_fig = os.path.join(output_base, "fig")
output_pdf = os.path.join(output_base, "pdf")

# Create directories if they don't exist
os.makedirs(output_png, exist_ok=True)
os.makedirs(output_fig, exist_ok=True)
os.makedirs(output_pdf, exist_ok=True)

def save_figure(name):
    plt.tight_layout()
    # Save PNG
    png_path = os.path.join(output_png, f"{name}.png")
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    # Save PDF
    pdf_path = os.path.join(output_pdf, f"{name}.pdf")
    plt.savefig(pdf_path, bbox_inches='tight')
    # Save fig
    fig_path = os.path.join(output_fig, f"{name}.fig")
    plt.savefig(fig_path, bbox_inches='tight')
    print(f"Saved figure: {name} (.png, .fig and .pdf)")


if not os.path.exists(data_path):
    print(f"Error: File not found at {data_path}")
    exit(1)

print(f"Loading data from {data_path}")

try:
    mat_contents = sio.loadmat(data_path)
    resultsMatrix = mat_contents['resultsMatrix']
    P_ped = mat_contents['P_ped'].flatten()
except Exception as e:
    print(f"Error loading data: {e}")
    raise

# Limit to 50 simulations max for paper figures
if resultsMatrix.shape[0] > 50:
    print(f"Limiting analysis to first 50 simulations (Original: {resultsMatrix.shape[0]})")
    resultsMatrix = resultsMatrix[:50, :]

# Pre-processing: Clean data by removing NaNs per column
cleaned_data = []
valid_P_ped = []
for i in range(resultsMatrix.shape[1]):
    col = resultsMatrix[:, i]
    cleaned_col = col[~np.isnan(col)]
    if len(cleaned_col) > 0:
        cleaned_data.append(cleaned_col)
        valid_P_ped.append(P_ped[i])

if not cleaned_data:
    print("Error: No data points found after removing NaNs.")
    exit(1)

valid_P_ped = np.array(valid_P_ped)

# --- Figure 1: Boxplot ---
plt.figure(figsize=(10, 6))
# bp = plt.boxplot(cleaned_data, labels=valid_P_ped, patch_artist=True)
# Alternatively, using seaborn for even better boxplots:
sns.boxplot(data=cleaned_data, palette="Blues_r")
plt.xticks(range(len(valid_P_ped)), valid_P_ped)

plt.xlabel('Crowd Size (P)')
plt.ylabel('Peak Lateral Accel. ($m/s^2$)')
plt.title(f'Bridge {bridge_num}: Peak Lateral Acceleration Distribution')
save_figure(f"boxplot_peak_accel_{output_bridge}")

# --- Figure 2: Scatter Plot (Individual points) ---
plt.figure(figsize=(10, 6))
for i, (p_val, data) in enumerate(zip(valid_P_ped, cleaned_data)):
    plt.scatter([i] * len(data), data, color='black', alpha=0.15, s=20, edgecolors='none')

plt.xticks(range(len(valid_P_ped)), valid_P_ped)
plt.xlabel('Crowd Size (P)')
plt.ylabel('Peak Lateral Accel. ($m/s^2$)')
plt.title(f'Bridge {bridge_num}: Simulation Raw Results')
save_figure(f"scatter_raw_results_{output_bridge}")

# --- Figure 3: Histogram for a selected Crowd Size ---
if P_target not in valid_P_ped:
    P_target = valid_P_ped[np.argmin(np.abs(valid_P_ped - P_target))]

index = np.where(valid_P_ped == P_target)[0][0]
data_col = cleaned_data[index]

plt.figure(figsize=(8, 6))
sns.histplot(data_col, bins=20, kde=False, color='#1f77b4', stat="density", alpha=0.6, label='Empirical Data')

# Fit Lognormal Distribution
mu_ln = np.mean(np.log(data_col))
sigma_ln = np.std(np.log(data_col), ddof=1)
x_fit = np.linspace(max(0.001, min(data_col)*0.8), max(data_col)*1.2, 200)
p_fit = (1 / (x_fit * sigma_ln * np.sqrt(2 * np.pi))) * np.exp(- (np.log(x_fit) - mu_ln)**2 / (2 * sigma_ln**2))

plt.plot(x_fit, p_fit, 'r-', lw=2.5, label=f'Lognormal Fit\n($\mu_{{ln}}$={mu_ln:.2f}, $\sigma_{{ln}}$={sigma_ln:.2f})')

plt.xlabel('Peak Lateral Accel. ($m/s^2$)')
plt.ylabel('Probability Density')
plt.title(f'Bridge {bridge_num}: Acceleration PDF for P = {P_target}')
plt.legend()
save_figure(f"histogram_distribution_P{P_target}_{output_bridge}")

# --- Figure 4: Fragility curves ---
p_exceed_ln = []
p_exceed_emp = []

for data in cleaned_data:
    m = np.mean(np.log(data))
    s = np.std(np.log(data), ddof=1)
    p_ln = 1 - norm.cdf(np.log(delta), loc=m, scale=s)
    p_exceed_ln.append(p_ln)
    p_emp = np.sum(data > delta) / len(data)
    p_exceed_emp.append(p_emp)

plt.figure(figsize=(10, 6))
plt.plot(valid_P_ped, p_exceed_ln, '-o', color='#d62728', lw=2.5, markersize=8, label='Lognormal Fit')
plt.plot(valid_P_ped, p_exceed_emp, 's--', color='#1f77b4', lw=2, markersize=8, label='Empirical Observations')

plt.xlabel('Crowd Size (P)')
plt.ylabel(f'Probability of Accel. > {delta} $m/s^2$')
plt.title(f'Bridge {bridge_num}: Seismic Fragility Curves')
plt.yticks(np.arange(0, 1.1, 0.1), [f'{int(x*100)}%' for x in np.arange(0, 1.1, 0.1)])
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
save_figure(f"fragility_analysis_{output_bridge}")

plt.show()
print("All figures generated successfully.")
