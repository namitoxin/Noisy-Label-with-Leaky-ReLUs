"""
Leaky ReLU Experiment: α from -1.0 to +0.9

Configuration:
- α values: 20 values [-1.0, -0.9, ..., 0.0, 0.1, ..., 0.9]
- noise_rates: 31 values [0%, 1%, 2%, ..., 30%]
- Total runs: 620

Seed is fixed (42), so results are fully reproducible.

Output files (in 'results/' folder):
- heatmap.png
- comparison.png
- difference.png
- results.json
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm
import json
import os

# ================================================================================
# Setup
# ================================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

SAVE_DIR = 'results'
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
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.scale = 1.0 / np.sqrt(1 + alpha**2)
    
    def forward(self, x):
        return self.scale * torch.where(x >= 0, x, self.alpha * x)


class DNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, alpha):
        super().__init__()
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
    
    return extract(train_data, n_train), extract(test_data, n_test)


def inject_noise(Y, rate, num_classes=10):
    Y_noisy = Y.clone()
    n_corrupt = int(len(Y) * rate)
    mask = torch.zeros(len(Y), dtype=torch.bool)
    
    if n_corrupt > 0:
        idx = np.random.choice(len(Y), n_corrupt, replace=False)
        mask[idx] = True
        for i in idx:
            new_label = np.random.randint(0, num_classes)
            while new_label == Y[i].item():
                new_label = np.random.randint(0, num_classes)
            Y_noisy[i] = new_label
    
    return Y_noisy, mask


def to_onehot(Y, num_classes=10):
    n = len(Y)
    targets = -torch.ones(n, num_classes)
    targets[torch.arange(n), Y] = 1.0
    return targets

# ================================================================================
# Training
# ================================================================================
def train(model, X_train, Y_train_noisy, Y_train_clean, X_test, Y_test_clean,
          epochs, lr, record_every=5):
    model = model.to(device)
    X_tr = X_train.to(device)
    Y_tr_noisy = Y_train_noisy.to(device)
    Y_tr_clean = Y_train_clean.to(device)
    X_te = X_test.to(device)
    Y_te_clean = Y_test_clean.to(device)
    
    n_train, n_test = len(X_train), len(X_test)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='sum')
    
    history = {'epoch': [], 'L_nu': [], 'R_clean': []}
    
    model.eval()
    with torch.no_grad():
        history['L0'] = criterion(model(X_tr), Y_tr_noisy).item()
    
    for epoch in range(epochs + 1):
        if epoch % record_every == 0:
            model.eval()
            with torch.no_grad():
                L_nu = criterion(model(X_tr), Y_tr_noisy).item()
                R_clean = criterion(model(X_te), Y_te_clean).item()
                
                if np.isnan(L_nu) or L_nu > 1e10:
                    history['diverged'] = True
                    break
            
            history['epoch'].append(epoch)
            history['L_nu'].append(L_nu / n_train)
            history['R_clean'].append(R_clean / n_test)
        
        if epoch == epochs:
            break
        
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_tr), Y_tr_noisy)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
    
    history['diverged'] = history.get('diverged', False)
    return history

# ================================================================================
# Analysis
# ================================================================================
def analyze_convergence(history, alpha):
    L = np.array(history['L_nu'])
    t = np.array(history['epoch'])
    L0 = history['L0']
    
    if L0 <= 0 or len(L) < 10:
        return None
    
    ratios = L / (L0 / len(L) * 1000)  # per-sample normalization
    valid = (ratios > 1e-10) & (ratios < 1.0)
    
    if np.sum(valid) < 5:
        return None
    
    log_ratios = np.log(ratios[valid])
    t_valid = t[valid]
    slope, _ = np.polyfit(t_valid, log_ratios, 1)
    
    return {
        'gamma': float(np.exp(slope)),
        'theory_factor': float((1 - alpha)**2 / (1 + alpha**2))
    }


def find_best_epoch(history):
    R = np.array(history['R_clean'])
    if len(R) == 0:
        return None
    
    best_idx = np.argmin(R)
    return {
        'T_star': int(history['epoch'][best_idx]),
        'min_R': float(R[best_idx]),
        'L_at_T': float(history['L_nu'][best_idx])
    }

# ================================================================================
# Save/Load
# ================================================================================
def to_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): to_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_json(i) for i in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def save(data, filename):
    path = os.path.join(SAVE_DIR, filename)
    with open(path, 'w') as f:
        json.dump(to_json(data), f, indent=2)
    print(f"Saved: {path}")

# ================================================================================
# Visualization
# ================================================================================
def get_R(exp, alpha_str):
    res = exp['alpha_results'].get(alpha_str, {})
    t = res.get('best_epoch')
    return t['min_R'] if t else np.nan


def plot_all(results):
    noise_rates = sorted([e['noise_rate'] for e in results['experiments']])
    alphas = sorted([float(a) for a in results['experiments'][0]['alpha_results'].keys()])
    
    # Build matrix
    R = np.zeros((len(noise_rates), len(alphas)))
    for i, exp in enumerate(sorted(results['experiments'], key=lambda x: x['noise_rate'])):
        for j, a in enumerate(alphas):
            R[i, j] = get_R(exp, str(a))
    
    # Figure 1: Heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    ax = axes[0]
    im = ax.imshow(R, aspect='auto', cmap='RdYlGn_r', origin='lower')
    ax.set_xticks(range(0, len(alphas), 2))
    ax.set_xticklabels([f'{alphas[i]:.1f}' for i in range(0, len(alphas), 2)])
    ax.set_yticks(range(0, len(noise_rates), 3))
    ax.set_yticklabels([f'{noise_rates[i]*100:.0f}%' for i in range(0, len(noise_rates), 3)])
    ax.set_xlabel('α')
    ax.set_ylabel('Noise Rate ρ')
    ax.set_title('Test Loss R(T*)\n(Lighter = Better)')
    plt.colorbar(im, ax=ax)
    
    for i in range(len(noise_rates)):
        if not np.all(np.isnan(R[i])):
            ax.scatter([np.nanargmin(R[i])], [i], marker='*', s=100, c='blue', edgecolors='white')
    
    ax = axes[1]
    best = [alphas[np.nanargmin(R[i])] if not np.all(np.isnan(R[i])) else np.nan for i in range(len(noise_rates))]
    ax.plot([r*100 for r in noise_rates], best, 'bo-', markersize=4)
    ax.set_xlabel('Noise Rate ρ (%)')
    ax.set_ylabel('Optimal α')
    ax.set_title('Optimal α vs Noise Rate')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='α=0')
    ax.axhline(y=-1, color='green', linestyle=':', alpha=0.5, label='α=-1')
    ax.legend()
    ax.set_ylim(-1.2, 1.0)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'heatmap.png'), dpi=150)
    plt.show()
    
    # Figure 2: Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    for a, c, m in [(-1.0,'green','o'), (-0.5,'blue','s'), (0.0,'red','^'), (0.3,'orange','D'), (0.5,'purple','v')]:
        vals = [get_R(e, str(a)) for e in sorted(results['experiments'], key=lambda x: x['noise_rate'])]
        ax.plot([r*100 for r in noise_rates], vals, color=c, marker=m, label=f'α={a}', markersize=4)
    ax.set_xlabel('Noise Rate ρ (%)')
    ax.set_ylabel('Test Loss R(T*)')
    ax.set_title('Comparison by α')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'comparison.png'), dpi=150)
    plt.show()
    
    # Figure 3: Difference
    fig, ax = plt.subplots(figsize=(12, 6))
    diffs, rhos = [], []
    for exp in sorted(results['experiments'], key=lambda x: x['noise_rate']):
        r0, r1 = get_R(exp, '0.0'), get_R(exp, '-1.0')
        if not np.isnan(r0) and not np.isnan(r1):
            diffs.append(r0 - r1)
            rhos.append(exp['noise_rate'] * 100)
    
    colors = ['red' if d > 0 else 'blue' for d in diffs]
    ax.bar(rhos, diffs, color=colors, width=0.8, alpha=0.7)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_xlabel('Noise Rate ρ (%)')
    ax.set_ylabel('R(α=0) - R(α=-1)')
    ax.set_title('Difference: Red = α=-1 better, Blue = α=0 better')
    ax.legend(handles=[Patch(facecolor='red', alpha=0.7, label='α=-1 better'),
                       Patch(facecolor='blue', alpha=0.7, label='α=0 better')])
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'difference.png'), dpi=150)
    plt.show()
    
    print(f"\nSaved to {SAVE_DIR}/")
    return diffs

# ================================================================================
# Main
# ================================================================================
def run(config):
    print(f"\nExperiment: {len(config['alphas'])} α × {len(config['noise_rates'])} ρ = {len(config['alphas'])*len(config['noise_rates'])} runs")
    
    results = {'config': config, 'experiments': []}
    
    # Checkpoint
    ckpt_path = os.path.join(SAVE_DIR, 'checkpoint.json')
    done = set()
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            ckpt = json.load(f)
            results['experiments'] = ckpt.get('experiments', [])
            done = {e['noise_rate'] for e in results['experiments']}
            print(f"Resuming: {len(done)} noise rates done")
    
    pbar = tqdm(total=len(config['noise_rates']) * len(config['alphas']))
    pbar.update(len(done) * len(config['alphas']))
    
    for rho in config['noise_rates']:
        if rho in done:
            continue
        
        set_seed(config['seed'])
        (X_tr, Y_tr), (X_te, Y_te) = load_mnist(config['n_train'], config['n_test'], config['seed'])
        Y_tr_noisy, mask = inject_noise(Y_tr, rho, config['n_classes'])
        
        Y_tr_noisy_oh = to_onehot(Y_tr_noisy, config['n_classes'])
        Y_tr_clean_oh = to_onehot(Y_tr, config['n_classes'])
        Y_te_clean_oh = to_onehot(Y_te, config['n_classes'])
        
        exp = {'noise_rate': rho, 'n_corrupted': int(mask.sum()), 'alpha_results': {}}
        
        for alpha in config['alphas']:
            set_seed(config['seed'])
            model = DNN(config['input_dim'], config['hidden_dim'], config['n_classes'], config['n_layers'], alpha)
            hist = train(model, X_tr, Y_tr_noisy_oh, Y_tr_clean_oh, X_te, Y_te_clean_oh,
                        config['epochs'], config['lr'], config['record_every'])
            
            exp['alpha_results'][str(alpha)] = {
                'convergence': analyze_convergence(hist, alpha),
                'best_epoch': find_best_epoch(hist),
                'diverged': hist['diverged']
            }
            pbar.update(1)
        
        results['experiments'].append(exp)
        save(results, 'checkpoint.json')
    
    pbar.close()
    return results


def main():
    config = {
        'input_dim': 784,
        'n_classes': 10,
        'hidden_dim': 500,
        'n_layers': 3,
        'n_train': 1000,
        'n_test': 500,
        'epochs': 500,
        'lr': 0.005,
        'seed': 42,
        'record_every': 5,
        'alphas': [round(x * 0.1, 1) for x in range(-10, 10)],
        'noise_rates': [round(x * 0.01, 2) for x in range(0, 31)],
    }
    
    results = run(config)
    save(results, 'results.json')
    diffs = plot_all(results)
    
    # Summary
    print("\n" + "="*60)
    wins = {'neg': 0, 'zero': 0, 'pos': 0}
    for exp in results['experiments']:
        best_a, best_r = None, float('inf')
        for a, d in exp['alpha_results'].items():
            if d['best_epoch'] and d['best_epoch']['min_R'] < best_r:
                best_r = d['best_epoch']['min_R']
                best_a = float(a)
        if best_a is not None:
            if best_a < 0: wins['neg'] += 1
            elif best_a == 0: wins['zero'] += 1
            else: wins['pos'] += 1
    
    total = sum(wins.values())
    print(f"α<0: {wins['neg']}/{total} ({wins['neg']/total*100:.1f}%)")
    print(f"α=0: {wins['zero']}/{total} ({wins['zero']/total*100:.1f}%)")
    print(f"α>0: {wins['pos']}/{total} ({wins['pos']/total*100:.1f}%)")
    print("="*60)


if __name__ == '__main__':
    main()
