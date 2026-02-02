"""
Experiment: α from -1.0 to +0.9

Configuration:
- α values: 20 values [-1.0, -0.9, ..., 0.0, 0.1, ..., 0.9] (0.1 steps)
- noise_rates: 31 values [0%, 1%, 2%, ..., 30%] (1% steps)

Total runs: 20 × 31 = 620 runs

Output:
- Figure 1: Heatmap + Optimal α curve
- Figure 2: Comparison of α=-1, -0.5, 0, 0.3, 0.5, 0.7
- Figure 3: Difference plot (red/blue bar chart)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os

# ================================================================================
# Setup
# ================================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

SAVE_DIR = 'experiment_extended'
os.makedirs(SAVE_DIR, exist_ok=True)

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ================================================================================
# Network
# ================================================================================
class RescaledLeakyReLU(nn.Module):
    """Rescaled Leaky ReLU: σ_α(z) / sqrt(1 + α²)"""
    def __init__(self, alpha):
        super().__init__()
        if alpha >= 1.0:
            raise ValueError(f"α must be < 1, got {alpha}")
        self.alpha = alpha
        self.scale = 1.0 / np.sqrt(1 + alpha**2)
    
    def forward(self, x):
        return self.scale * torch.where(x >= 0, x, self.alpha * x)


class OverparameterizedDNN(nn.Module):
    """Network with fixed A, B and trainable W"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, alpha):
        super().__init__()
        self.alpha = alpha
        self.register_buffer('A', torch.randn(hidden_dim, input_dim) / np.sqrt(hidden_dim))
        self.register_buffer('B', torch.randn(output_dim, hidden_dim) / np.sqrt(output_dim))
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

# ================================================================================
# Data
# ================================================================================
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
    """Symmetric label noise injection"""
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

# ================================================================================
# Training
# ================================================================================
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

# ================================================================================
# Analysis
# ================================================================================
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
        'min_R_clean': float(min_R_clean),
        'L_nu_at_Tstar': float(L_nu_at_Tstar),
        'final_R_clean': float(final_R_clean),
        'final_L_nu': float(final_L_nu),
        'improvement': float((final_R_clean - min_R_clean) / final_R_clean * 100)
    }

# ================================================================================
# Save/Load
# ================================================================================
def convert_for_json(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(i) for i in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def save_results(results, filename):
    path = os.path.join(SAVE_DIR, filename)
    with open(path, 'w') as f:
        json.dump(convert_for_json(results), f, indent=2)
    print(f"  Saved: {path}")

# ================================================================================
# Visualization (3 Figures, NaN-safe)
# ================================================================================
def safe_get_R(exp, alpha_str):
    """Safely get R value, returning np.nan if not available"""
    res = exp['alpha_results'].get(alpha_str, {})
    t34 = res.get('theorem_3_4')
    if t34 and t34.get('min_R_clean') is not None:
        return t34['min_R_clean']
    return np.nan


def plot_results(results):
    """Generate 3 figures (NaN-safe version)"""
    noise_rates = sorted([exp['noise_rate'] for exp in results['experiments']])
    alpha_values = sorted([float(a) for a in results['experiments'][0]['alpha_results'].keys()])
    
    # Build R_matrix
    R_matrix = np.zeros((len(noise_rates), len(alpha_values)))
    
    for i, exp in enumerate(sorted(results['experiments'], key=lambda x: x['noise_rate'])):
        for j, alpha in enumerate(alpha_values):
            alpha_str = str(alpha)
            R_matrix[i, j] = safe_get_R(exp, alpha_str)
    
    # ========== Figure 1: Heatmap + Optimal α ==========
    print("\n" + "="*60)
    print("Figure 1: Heatmap and Optimal α")
    print("="*60)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    ax = axes[0]
    im = ax.imshow(R_matrix, aspect='auto', cmap='RdYlGn_r', origin='lower')
    
    xtick_step = max(1, len(alpha_values) // 10)
    ytick_step = max(1, len(noise_rates) // 10)
    
    ax.set_xticks(range(0, len(alpha_values), xtick_step))
    ax.set_xticklabels([f'{alpha_values[i]:.1f}' for i in range(0, len(alpha_values), xtick_step)])
    ax.set_yticks(range(0, len(noise_rates), ytick_step))
    ax.set_yticklabels([f'{noise_rates[i]*100:.0f}%' for i in range(0, len(noise_rates), ytick_step)])
    
    ax.set_xlabel('α (negative to positive)')
    ax.set_ylabel('Noise Rate ρ')
    ax.set_title('Test Loss R(T*) - Extended Range\n(Lighter = Better)')
    plt.colorbar(im, ax=ax)
    
    # Mark best α (NaN-safe)
    for i in range(len(noise_rates)):
        row = R_matrix[i, :]
        if not np.all(np.isnan(row)):
            best_j = np.nanargmin(row)
            ax.scatter([best_j], [i], marker='*', s=100, c='blue', edgecolors='white', linewidths=1)
    
    # Right: Best α vs noise rate (NaN-safe)
    ax = axes[1]
    best_alphas = []
    valid_noise_rates = []
    for i in range(len(noise_rates)):
        row = R_matrix[i, :]
        if not np.all(np.isnan(row)):
            best_alphas.append(alpha_values[np.nanargmin(row)])
            valid_noise_rates.append(noise_rates[i])
    
    ax.plot([r*100 for r in valid_noise_rates], best_alphas, 'bo-', markersize=4, linewidth=1)
    ax.set_xlabel('Noise Rate ρ (%)')
    ax.set_ylabel('Optimal α')
    ax.set_title('Optimal α vs Noise Rate (Extended Range)')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='ReLU (α=0)')
    ax.axhline(y=-1, color='green', linestyle=':', alpha=0.5, label='Absolute (α=-1)')
    ax.axhline(y=0.5, color='orange', linestyle='-.', alpha=0.5, label='α=0.5')
    ax.legend()
    ax.set_ylim(-1.2, 1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'extended_heatmap.png'), dpi=150)
    plt.show()
    
    # ========== Figure 2: α comparison ==========
    print("\n" + "="*60)
    print("Figure 2: Comparison of Negative, Zero, and Positive α")
    print("="*60)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    alpha_to_plot = [-1.0, -0.5, 0.0, 0.3, 0.5, 0.7]
    colors = ['green', 'blue', 'red', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'D', 'v', '<']
    
    for alpha, color, marker in zip(alpha_to_plot, colors, markers):
        R_values = []
        for exp in sorted(results['experiments'], key=lambda x: x['noise_rate']):
            R_values.append(safe_get_R(exp, str(alpha)))
        
        if not all(np.isnan(v) for v in R_values):
            ax.plot([r*100 for r in noise_rates], R_values, 
                    color=color, marker=marker, linestyle='-',
                    label=f'α={alpha}', markersize=4, linewidth=1.5)
    
    ax.set_xlabel('Noise Rate ρ (%)')
    ax.set_ylabel('Test Loss R(T*)')
    ax.set_title('Comparison: Negative, Zero, and Positive α')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'alpha_comparison_extended.png'), dpi=150)
    plt.show()
    
    # ========== Figure 3: Difference plot (red/blue bars) ==========
    print("\n" + "="*60)
    print("Figure 3: Difference Plot (α=0 vs α=-1)")
    print("="*60)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    diffs = []
    valid_rhos = []
    for exp in sorted(results['experiments'], key=lambda x: x['noise_rate']):
        R_0 = safe_get_R(exp, '0.0')
        R_neg1 = safe_get_R(exp, '-1.0')
        if not np.isnan(R_0) and not np.isnan(R_neg1):
            diffs.append(R_0 - R_neg1)
            valid_rhos.append(exp['noise_rate'] * 100)
    
    colors = ['red' if d > 0 else 'blue' for d in diffs]
    
    ax.bar(valid_rhos, diffs, color=colors, width=0.8, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Noise Rate ρ (%)')
    ax.set_ylabel('R(α=0) - R(α=-1)')
    ax.set_title('Difference: Positive (Red) = α=-1 Better, Negative (Blue) = α=0 Better')
    ax.grid(True, alpha=0.3, axis='y')
    
    ax.bar([], [], color='red', alpha=0.7, label='α=-1 is better')
    ax.bar([], [], color='blue', alpha=0.7, label='α=0 is better')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'difference_plot_extended.png'), dpi=150)
    plt.show()
    
    print(f"\nPlots saved to {SAVE_DIR}/")
    
    return diffs


def print_summary(results, diffs):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("EXTENDED SUMMARY: Including α > 0")
    print("="*60)
    
    print(f"\n{'ρ':<8} {'Best α':<10} {'R*(best)':<12} {'R*(α=-1)':<12} {'R*(α=0)':<12} {'R*(α=0.5)':<12}")
    print("-"*80)
    
    for exp in sorted(results['experiments'], key=lambda x: x['noise_rate']):
        noise_rate = exp['noise_rate']
        if noise_rate not in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
            continue
        
        best_alpha = None
        best_loss = float('inf')
        
        for alpha, data in exp['alpha_results'].items():
            t34 = data.get('theorem_3_4')
            if t34 and t34.get('min_R_clean') is not None:
                if t34['min_R_clean'] < best_loss:
                    best_loss = t34['min_R_clean']
                    best_alpha = float(alpha)
        
        R_neg1 = safe_get_R(exp, '-1.0')
        R_0 = safe_get_R(exp, '0.0')
        R_05 = safe_get_R(exp, '0.5')
        
        if best_alpha is not None:
            print(f"{noise_rate*100:5.0f}%   {best_alpha:8.1f}   {best_loss:<12.4f} {R_neg1:<12.4f} {R_0:<12.4f} {R_05:<12.4f}")
    
    # α region wins
    print("\n" + "="*60)
    print("ANALYSIS: Best α distribution")
    print("="*60)
    
    wins = {'negative': 0, 'zero': 0, 'positive': 0}
    for exp in results['experiments']:
        best_alpha = None
        best_loss = float('inf')
        for alpha, data in exp['alpha_results'].items():
            t34 = data.get('theorem_3_4')
            if t34 and t34.get('min_R_clean') is not None:
                if t34['min_R_clean'] < best_loss:
                    best_loss = t34['min_R_clean']
                    best_alpha = float(alpha)
        
        if best_alpha is not None:
            if best_alpha < 0:
                wins['negative'] += 1
            elif best_alpha == 0:
                wins['zero'] += 1
            else:
                wins['positive'] += 1
    
    total = sum(wins.values())
    if total > 0:
        print(f"\n  α < 0  (negative): {wins['negative']:3d} / {total} ({wins['negative']/total*100:.1f}%)")
        print(f"  α = 0  (ReLU):     {wins['zero']:3d} / {total} ({wins['zero']/total*100:.1f}%)")
        print(f"  α > 0  (positive): {wins['positive']:3d} / {total} ({wins['positive']/total*100:.1f}%)")
    
    # Difference statistics
    if diffs:
        print("\n" + "="*60)
        print("ANALYSIS: R(α=0) - R(α=-1) statistics")
        print("="*60)
        diffs_array = np.array(diffs)
        print(f"\n  Mean:   {np.mean(diffs_array):+.4f}")
        print(f"  Std:    {np.std(diffs_array):.4f}")
        print(f"  Min:    {np.min(diffs_array):+.4f} (α=0 most advantageous)")
        print(f"  Max:    {np.max(diffs_array):+.4f} (α=-1 most advantageous)")
        print(f"  α=-1 wins: {np.sum(diffs_array > 0)} / {len(diffs_array)} ({np.sum(diffs_array > 0)/len(diffs_array)*100:.1f}%)")

# ================================================================================
# Main Experiment
# ================================================================================
def run_experiment(config):
    print("\n" + "="*70)
    print("Extended Experiment: α from -1.0 to +0.9")
    print("="*70)
    print(f"Config: {config['n_train']} train, {config['epochs']} epochs, lr={config['lr']}")
    print(f"α values: {len(config['alpha_values'])} values from {config['alpha_values'][0]} to {config['alpha_values'][-1]}")
    print(f"Noise rates: {len(config['noise_rates'])} values")
    
    all_results = {
        'config': config,
        'experiments': []
    }
    
    total_runs = len(config['noise_rates']) * len(config['alpha_values'])
    pbar = tqdm(total=total_runs, desc="Total progress")
    
    # Checkpoint loading
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
        
        set_seed(config['seed'])
        X_train, Y_train_clean, X_test, Y_test_clean = load_mnist(
            config['n_train'], config['n_test'], config['seed']
        )
        
        Y_train_noisy, corrupted_mask = inject_label_noise(
            Y_train_clean, noise_rate, config['num_classes']
        )
        Y_test_noisy, _ = inject_label_noise(
            Y_test_clean, noise_rate, config['num_classes']
        )
        
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
            
            noise_results['alpha_results'][str(alpha)] = {
                'theorem_3_1': thm31,
                'theorem_3_4': thm34,
                'diverged': history['diverged']
            }
            
            pbar.update(1)
        
        all_results['experiments'].append(noise_results)
        save_results(all_results, 'checkpoint.json')
    
    pbar.close()
    
    return all_results

# ================================================================================
# Main
# ================================================================================
def main():
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
        # α: from -1.0 to +0.9 in 0.1 steps (20 values, excluding 1.0)
        'alpha_values': [round(x * 0.1, 1) for x in range(-10, 10)],
        # Noise rate: 1% steps from 0% to 30%
        'noise_rates': [round(x * 0.01, 2) for x in range(0, 31)],
    }
    
    total_runs = len(config['alpha_values']) * len(config['noise_rates'])
    print(f"Extended experiment: {total_runs} runs")
    print(f"α values ({len(config['alpha_values'])}): {config['alpha_values']}")
    print(f"Noise rates: {len(config['noise_rates'])} values (0% to 30%)")
    print(f"Estimated time: ~{total_runs * 1.4 / 60:.1f} minutes")
    
    # Run experiment
    results = run_experiment(config)
    
    # Save final results
    save_results(results, 'extended_results.json')
    
    # Generate plots
    diffs = plot_results(results)
    
    # Print summary
    print_summary(results, diffs)
    
    print("\n" + "="*60)
    print("COMPLETED!")
    print("="*60)


if __name__ == '__main__':
    main()
