"""
Detailed Experiment: α in 0.1 steps, noise rate in 1% steps

Configuration:
- α values: 11 values [0.0, -0.1, -0.2, ..., -1.0] (0.1 steps)
- noise_rates: 31 values [0%, 1%, 2%, ..., 30%] (1% steps)

Total runs: 11 × 31 = 341 runs
Estimated time on RTX 3090 Ti: ~7-8 minutes
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os

#==============================================================================
# Setup
#==============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

SAVE_DIR = 'experiment_detailed'
os.makedirs(SAVE_DIR, exist_ok=True)

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


#==============================================================================
# Network
#==============================================================================
class RescaledLeakyReLU(nn.Module):
    """Rescaled Leaky ReLU: sigma_alpha(z) / sqrt(1 + alpha^2)"""
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.scale = 1.0 / np.sqrt(1 + alpha**2)
    
    def forward(self, x):
        return self.scale * torch.where(x >= 0, x, self.alpha * x)


class OverparameterizedDNN(nn.Module):
    """Network with fixed A, B and trainable W (matching the thesis)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, alpha):
        super().__init__()
        self.alpha = alpha
        # Fixed input/output matrices
        self.register_buffer('A', torch.randn(hidden_dim, input_dim) / np.sqrt(hidden_dim))
        self.register_buffer('B', torch.randn(output_dim, hidden_dim) / np.sqrt(output_dim))
        # Trainable hidden weights (He initialization)
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim))
            for _ in range(num_layers)
        ])
        self.activation = RescaledLeakyReLU(alpha)
    
    def forward(self, x):
        h = torch.mm(x, self.A.t())
        for W in self.weights:
            h = self.activation(torch.mm(h, W.t()))
        return torch.mm(h, self.B.t())


#==============================================================================
# Data
#==============================================================================
def load_mnist(n_train, n_test, seed=42):
    from torchvision import datasets, transforms
    
    set_seed(seed)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    def extract(dataset, n):
        indices = np.random.choice(len(dataset), min(n, len(dataset)), replace=False)
        X = torch.stack([dataset[i][0] for i in indices]).view(len(indices), -1)
        Y = torch.tensor([dataset[i][1] for i in indices])
        X = X / (X.norm(dim=1, keepdim=True) + 1e-8)
        return X, Y
    
    X_train, Y_train = extract(train_data, n_train)
    X_test, Y_test = extract(test_data, n_test)
    
    return X_train, Y_train, X_test, Y_test


def inject_label_noise(Y, noise_rate, num_classes=10):
    """
    Inject symmetric label noise.
    
    For each sample with probability noise_rate:
      Replace true label y with a uniformly random incorrect label y' != y
    
    This satisfies Assumption 2.4 (sparse label corruption) with rate rho = noise_rate.
    """
    Y_noisy = Y.clone()
    n_corrupt = int(len(Y) * noise_rate)
    corrupted_mask = torch.zeros(len(Y), dtype=torch.bool)
    
    if n_corrupt > 0:
        corrupt_idx = np.random.choice(len(Y), n_corrupt, replace=False)
        corrupted_mask[corrupt_idx] = True
        for i in corrupt_idx:
            new_label = np.random.randint(0, num_classes)
            while new_label == Y[i].item():
                new_label = np.random.randint(0, num_classes)
            Y_noisy[i] = new_label
    
    return Y_noisy, corrupted_mask


def to_regression_targets(Y, num_classes=10):
    """One-hot encoding scaled to [-1, 1]"""
    n = len(Y)
    targets = -torch.ones(n, num_classes)
    targets[torch.arange(n), Y] = 1.0
    return targets


#==============================================================================
# Training with Full Recording
#==============================================================================
def train_full_recording(model, X_train, Y_train_noisy, Y_train_clean,
                         X_test, Y_test_noisy, Y_test_clean,
                         epochs, lr, record_every=1):
    model = model.to(device)
    X_tr = X_train.to(device)
    Y_tr_noisy = Y_train_noisy.to(device)
    Y_tr_clean = Y_train_clean.to(device)
    X_te = X_test.to(device)
    Y_te_noisy = Y_test_noisy.to(device)
    Y_te_clean = Y_test_clean.to(device)
    
    n_train = len(X_train)
    n_test = len(X_test)
    
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='sum')
    
    history = {
        'epoch': [],
        'L_nu': [],
        'L_clean': [],
        'R_nu': [],
        'R_clean': [],
        'L_nu_per_sample': [],
        'R_clean_per_sample': [],
    }
    
    model.eval()
    with torch.no_grad():
        L0_nu = criterion(model(X_tr), Y_tr_noisy).item()
        L0_clean = criterion(model(X_tr), Y_tr_clean).item()
    history['L0_nu'] = L0_nu
    history['L0_clean'] = L0_clean
    
    for epoch in range(epochs + 1):
        if epoch % record_every == 0:
            model.eval()
            with torch.no_grad():
                out_tr = model(X_tr)
                out_te = model(X_te)
                
                L_nu = criterion(out_tr, Y_tr_noisy).item()
                L_clean = criterion(out_tr, Y_tr_clean).item()
                R_nu = criterion(out_te, Y_te_noisy).item()
                R_clean = criterion(out_te, Y_te_clean).item()
                
                if np.isnan(L_nu) or L_nu > 1e10:
                    history['diverged'] = True
                    history['diverged_epoch'] = epoch
                    break
            
            history['epoch'].append(epoch)
            history['L_nu'].append(L_nu)
            history['L_clean'].append(L_clean)
            history['R_nu'].append(R_nu)
            history['R_clean'].append(R_clean)
            history['L_nu_per_sample'].append(L_nu / n_train)
            history['R_clean_per_sample'].append(R_clean / n_test)
        
        if epoch == epochs:
            break
        
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tr)
        loss = criterion(outputs, Y_tr_noisy)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
    
    history['diverged'] = history.get('diverged', False)
    
    return history


#==============================================================================
# Analysis Functions
#==============================================================================
def verify_theorem_3_1(history, alpha):
    L_nu = np.array(history['L_nu'])
    epochs = np.array(history['epoch'])
    L0 = history['L0_nu']
    
    if L0 <= 0 or len(L_nu) < 10:
        return None
    
    ratios = L_nu / L0
    valid = (ratios > 1e-10) & (ratios < 1.0)
    
    if np.sum(valid) < 5:
        return None
    
    log_ratios = np.log(ratios[valid])
    t_valid = epochs[valid]
    
    slope, intercept = np.polyfit(t_valid, log_ratios, 1)
    gamma_empirical = np.exp(slope)
    
    theoretical_factor = (1 - alpha)**2 / (1 + alpha**2)
    
    predicted = slope * t_valid + intercept
    ss_res = np.sum((log_ratios - predicted)**2)
    ss_tot = np.sum((log_ratios - np.mean(log_ratios))**2)
    r_squared = 1 - ss_res / (ss_tot + 1e-10)
    
    return {
        'gamma_empirical': gamma_empirical,
        'theoretical_factor': theoretical_factor,
        'r_squared': r_squared,
        'slope': slope,
        'valid_epochs': int(np.sum(valid))
    }


def find_optimal_stopping(history):
    R_clean = np.array(history['R_clean_per_sample'])
    epochs = np.array(history['epoch'])
    
    if len(R_clean) == 0:
        return None
    
    best_idx = np.argmin(R_clean)
    T_star = epochs[best_idx]
    min_R_clean = R_clean[best_idx]
    
    L_nu_at_Tstar = history['L_nu_per_sample'][best_idx]
    
    final_R_clean = R_clean[-1]
    final_L_nu = history['L_nu_per_sample'][-1]
    
    return {
        'T_star': int(T_star),
        'min_R_clean': min_R_clean,
        'L_nu_at_Tstar': L_nu_at_Tstar,
        'final_R_clean': final_R_clean,
        'final_L_nu': final_L_nu,
        'improvement': (final_R_clean - min_R_clean) / final_R_clean * 100
    }


#==============================================================================
# Main Experiment
#==============================================================================
def run_detailed_experiment(config):
    print("\n" + "="*70)
    print("Detailed Experiment: α (0.1 step) × ρ (1% step)")
    print("="*70)
    print(f"Config: {config['n_train']} train, {config['epochs']} epochs, lr={config['lr']}")
    print(f"α values: {len(config['alpha_values'])} values from {config['alpha_values'][0]} to {config['alpha_values'][-1]}")
    print(f"Noise rates: {len(config['noise_rates'])} values from {config['noise_rates'][0]*100:.0f}% to {config['noise_rates'][-1]*100:.0f}%")
    
    all_results = {
        'config': config,
        'experiments': []
    }
    
    total_runs = len(config['noise_rates']) * len(config['alpha_values'])
    pbar = tqdm(total=total_runs, desc="Total progress")
    
    # Check for checkpoint
    checkpoint_path = os.path.join(SAVE_DIR, 'checkpoint.json')
    completed_noise_rates = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
            all_results['experiments'] = checkpoint.get('experiments', [])
            completed_noise_rates = {exp['noise_rate'] for exp in all_results['experiments']}
            print(f"Resuming from checkpoint. Completed: {len(completed_noise_rates)} noise rates")
            pbar.update(len(completed_noise_rates) * len(config['alpha_values']))
    
    for noise_rate in config['noise_rates']:
        if noise_rate in completed_noise_rates:
            continue
            
        # Load data once per noise rate
        set_seed(config['seed'])
        X_train, Y_train_clean, X_test, Y_test_clean = load_mnist(
            config['n_train'], config['n_test'], config['seed']
        )
        
        # Inject noise
        Y_train_noisy, corrupted_mask = inject_label_noise(
            Y_train_clean, noise_rate, config['num_classes']
        )
        Y_test_noisy, _ = inject_label_noise(
            Y_test_clean, noise_rate, config['num_classes']
        )
        
        # Convert to regression targets
        Y_tr_noisy_reg = to_regression_targets(Y_train_noisy, config['num_classes'])
        Y_tr_clean_reg = to_regression_targets(Y_train_clean, config['num_classes'])
        Y_te_noisy_reg = to_regression_targets(Y_test_noisy, config['num_classes'])
        Y_te_clean_reg = to_regression_targets(Y_test_clean, config['num_classes'])
        
        noise_results = {
            'noise_rate': noise_rate,
            'n_corrupted': int(corrupted_mask.sum()),
            'alpha_results': {}
        }
        
        for alpha in config['alpha_values']:
            set_seed(config['seed'])
            
            model = OverparameterizedDNN(
                config['input_dim'],
                config['hidden_dim'],
                config['num_classes'],
                config['num_layers'],
                alpha
            )
            
            history = train_full_recording(
                model,
                X_train, Y_tr_noisy_reg, Y_tr_clean_reg,
                X_test, Y_te_noisy_reg, Y_te_clean_reg,
                config['epochs'],
                config['lr'],
                record_every=config['record_every']
            )
            
            thm31 = verify_theorem_3_1(history, alpha)
            thm34 = find_optimal_stopping(history)
            
            noise_results['alpha_results'][alpha] = {
                'history': {
                    'epoch': history['epoch'],
                    'L_nu_per_sample': history['L_nu_per_sample'],
                    'R_clean_per_sample': history['R_clean_per_sample'],
                },
                'theorem_3_1': thm31,
                'theorem_3_4': thm34,
                'diverged': history['diverged']
            }
            
            pbar.update(1)
        
        all_results['experiments'].append(noise_results)
        
        # Save checkpoint every noise rate
        save_results(all_results, 'checkpoint.json')
        
        # Print progress
        if (len(all_results['experiments'])) % 5 == 0:
            print(f"\n  Completed {len(all_results['experiments'])}/{len(config['noise_rates'])} noise rates")
    
    pbar.close()
    
    return all_results


#==============================================================================
# Save/Load
#==============================================================================
def save_results(results, filename):
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj
    
    path = os.path.join(SAVE_DIR, filename)
    with open(path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"  Saved: {path}")


#==============================================================================
# Visualization
#==============================================================================
def plot_detailed_results(results):
    """Generate heatmap and analysis plots"""
    noise_rates = sorted([exp['noise_rate'] for exp in results['experiments']])
    alpha_values = sorted([float(a) for a in results['experiments'][0]['alpha_results'].keys()])
    
    # Create matrix of R(T*) values
    R_matrix = np.zeros((len(noise_rates), len(alpha_values)))
    
    for i, exp in enumerate(sorted(results['experiments'], key=lambda x: x['noise_rate'])):
        for j, alpha in enumerate(alpha_values):
            alpha_str = str(alpha)
            res = exp['alpha_results'].get(alpha_str, {})
            t34 = res.get('theorem_3_4', {})
            R_matrix[i, j] = t34.get('min_R_clean', np.nan) if t34 else np.nan
    
    # Figure 1: Detailed Heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Full heatmap
    ax = axes[0]
    im = ax.imshow(R_matrix, aspect='auto', cmap='RdYlGn_r', origin='lower')
    
    # Show fewer ticks for readability
    xtick_step = max(1, len(alpha_values) // 10)
    ytick_step = max(1, len(noise_rates) // 10)
    
    ax.set_xticks(range(0, len(alpha_values), xtick_step))
    ax.set_xticklabels([f'{alpha_values[i]:.1f}' for i in range(0, len(alpha_values), xtick_step)])
    ax.set_yticks(range(0, len(noise_rates), ytick_step))
    ax.set_yticklabels([f'{noise_rates[i]*100:.0f}%' for i in range(0, len(noise_rates), ytick_step)])
    
    ax.set_xlabel('α')
    ax.set_ylabel('Noise Rate ρ')
    ax.set_title('Test Loss R(T*) - Detailed Heatmap\n(Lighter = Better)')
    plt.colorbar(im, ax=ax)
    
    # Mark best α for each noise rate (safe for all-NaN rows)
    for i in range(len(noise_rates)):
        row = R_matrix[i, :]
        if np.all(np.isnan(row)):
            continue   # この noise rate は全 α で失敗 → 描かない
        best_j = np.nanargmin(row)
        ax.scatter([best_j], [i], marker='*', s=100, c='blue',
                  edgecolors='white', linewidths=1)
    
    # Right: Best α vs noise rate
    ax = axes[1]
    best_alphas = []
    for i in range(len(noise_rates)):
        row = R_matrix[i, :]
        if np.all(np.isnan(row)):
            best_alphas.append(np.nan)
        else:
            best_alphas.append(alpha_values[np.nanargmin(row)])

    ax.plot([r*100 for r in noise_rates], best_alphas, 'bo-', markersize=4, linewidth=1)
    ax.set_xlabel('Noise Rate ρ (%)')
    ax.set_ylabel('Optimal α')
    ax.set_title('Optimal α vs Noise Rate (1% resolution)')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='ReLU (α=0)')
    ax.axhline(y=-1, color='green', linestyle=':', alpha=0.5, label='Absolute (α=-1)')
    ax.legend()
    ax.set_ylim(-1.1, 0.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'detailed_heatmap.png'), dpi=150)
    plt.close()
    
    # Figure 2: α=0 vs α=-1 comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    R_alpha0 = []
    R_alpha_neg1 = []
    
    for exp in sorted(results['experiments'], key=lambda x: x['noise_rate']):
        r0 = exp['alpha_results'].get('0.0', {}).get('theorem_3_4', {})
        r1 = exp['alpha_results'].get('-1.0', {}).get('theorem_3_4', {})
        R_alpha0.append(r0.get('min_R_clean', np.nan) if r0 else np.nan)
        R_alpha_neg1.append(r1.get('min_R_clean', np.nan) if r1 else np.nan)
    
    ax.plot([r*100 for r in noise_rates], R_alpha0, 'b-o', label='α=0 (ReLU)', markersize=4)
    ax.plot([r*100 for r in noise_rates], R_alpha_neg1, 'r-s', label='α=-1 (Absolute)', markersize=4)
    
    # Mark where α=-1 becomes better
    diff = np.array(R_alpha0) - np.array(R_alpha_neg1)
    crossover_idx = np.where(diff > 0)[0]
    if len(crossover_idx) > 0:
        crossover_rho = noise_rates[crossover_idx[0]] * 100
        ax.axvline(x=crossover_rho, color='gray', linestyle='--', alpha=0.7, 
                   label=f'Crossover at ρ≈{crossover_rho:.0f}%')
    
    ax.set_xlabel('Noise Rate ρ (%)')
    ax.set_ylabel('Test Loss R(T*)')
    ax.set_title('α=0 vs α=-1: Which is Better?')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'alpha_comparison.png'), dpi=150)
    plt.close()
    
    # Figure 3: Difference plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    diff = np.array(R_alpha0) - np.array(R_alpha_neg1)
    colors = ['blue' if d < 0 else 'red' for d in diff]
    ax.bar([r*100 for r in noise_rates], diff, color=colors, width=0.8, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Noise Rate ρ (%)')
    ax.set_ylabel('R(α=0) - R(α=-1)')
    ax.set_title('Difference: Positive = α=-1 Better, Negative = α=0 Better')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'difference_plot.png'), dpi=150)
    plt.close()
    
    print(f"\nPlots saved to {SAVE_DIR}/")


def print_summary(results):
    print("\n" + "="*70)
    print("SUMMARY: Best α for each noise rate")
    print("="*70)
    
    print(f"\n{'ρ':<8} {'Best α':<8} {'R(T*)':<12} {'R(α=0)':<12} {'R(α=-1)':<12} {'Winner':<10}")
    print("-"*70)
    
    for exp in sorted(results['experiments'], key=lambda x: x['noise_rate']):
        noise_rate = exp['noise_rate']
        
        best_alpha = None
        best_loss = float('inf')
        
        for alpha, data in exp['alpha_results'].items():
            if data['theorem_3_4'] and data['theorem_3_4']['min_R_clean'] < best_loss:
                best_loss = data['theorem_3_4']['min_R_clean']
                best_alpha = float(alpha)
        
        R_0 = exp['alpha_results'].get('0.0', {}).get('theorem_3_4', {}).get('min_R_clean', np.nan)
        R_neg1 = exp['alpha_results'].get('-1.0', {}).get('theorem_3_4', {}).get('min_R_clean', np.nan)
        
        if not np.isnan(R_0) and not np.isnan(R_neg1):
            winner = 'α=-1' if R_0 > R_neg1 else 'α=0'
        else:
            winner = 'N/A'
        
        print(f"{noise_rate*100:5.0f}%   {best_alpha:6.2f}   {best_loss:<12.4f} {R_0:<12.4f} {R_neg1:<12.4f} {winner:<10}")


def find_crossover_point(results):
    """Find the noise rate where α=-1 becomes better than α=0"""
    print("\n" + "="*70)
    print("CROSSOVER ANALYSIS")
    print("="*70)
    
    crossover_points = []
    
    sorted_exps = sorted(results['experiments'], key=lambda x: x['noise_rate'])
    
    for i, exp in enumerate(sorted_exps):
        R_0 = exp['alpha_results'].get('0.0', {}).get('theorem_3_4', {}).get('min_R_clean', None)
        R_neg1 = exp['alpha_results'].get('-1.0', {}).get('theorem_3_4', {}).get('min_R_clean', None)
        
        if R_0 and R_neg1:
            diff = R_0 - R_neg1  # Positive means α=-1 is better
            
            if i > 0:
                prev_exp = sorted_exps[i-1]
                prev_R_0 = prev_exp['alpha_results'].get('0.0', {}).get('theorem_3_4', {}).get('min_R_clean', None)
                prev_R_neg1 = prev_exp['alpha_results'].get('-1.0', {}).get('theorem_3_4', {}).get('min_R_clean', None)
                
                if prev_R_0 and prev_R_neg1:
                    prev_diff = prev_R_0 - prev_R_neg1
                    
                    # Check for sign change
                    if prev_diff < 0 and diff > 0:
                        # Linear interpolation
                        rho_prev = prev_exp['noise_rate']
                        rho_curr = exp['noise_rate']
                        crossover_rho = rho_prev + (rho_curr - rho_prev) * (-prev_diff) / (diff - prev_diff)
                        crossover_points.append(crossover_rho)
                        print(f"Crossover between ρ={rho_prev*100:.0f}% and ρ={rho_curr*100:.0f}%")
                        print(f"  Estimated crossover point: ρ ≈ {crossover_rho*100:.1f}%")
    
    if crossover_points:
        print(f"\nMain crossover: α=-1 becomes better than α=0 at ρ ≈ {crossover_points[0]*100:.1f}%")
    else:
        print("\nNo clear crossover found in the tested range.")
    
    return crossover_points


#==============================================================================
# Main
#==============================================================================
def main():
    # Detailed configuration: α in 0.1 steps, noise rate in 1% steps
    config = {
        'input_dim': 784,
        'num_classes': 10,
        'hidden_dim': 500,
        'num_layers': 3,
        'n_train': 1000,
        'n_test': 500,
        'epochs': 500,
        'lr': 0.005,
        'seed': 42,
        'record_every': 5,
        # α: 0.1 steps from 0.0 to -1.0
        'alpha_values': [round(x * 0.1, 1) for x in range(0, -11, -1)],  # [0.0, -0.1, ..., -1.0]
        # Noise rate: 1% steps from 0% to 30%
        'noise_rates': [round(x * 0.01, 2) for x in range(0, 31)],  # [0.0, 0.01, ..., 0.30]
    }
    
    total_runs = len(config['alpha_values']) * len(config['noise_rates'])
    print(f"Detailed experiment: {total_runs} runs")
    print(f"α values: {config['alpha_values']}")
    print(f"Noise rates: {[f'{r*100:.0f}%' for r in config['noise_rates'][:5]]} ... {[f'{r*100:.0f}%' for r in config['noise_rates'][-3:]]}")
    print(f"Estimated time on RTX 3090 Ti: ~{total_runs * 1.4 / 60:.1f} minutes")
    
    results = run_detailed_experiment(config)
    
    save_results(results, 'detailed_results.json')
    
    plot_detailed_results(results)
    
    print_summary(results)
    
    find_crossover_point(results)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
