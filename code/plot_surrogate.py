from neural_network import *
import os
import re
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data', 'CylindricalDrift')

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Params ────────────────────────────────────────────────────────────────────
Equation_name       = 'CylindricalDrift'
choose              = 10000
noise_level         = 50
noise_type          = 'Gaussian'
trail_num           = 'run1'
Activation_function = 'Rational'

# ── Load data ─────────────────────────────────────────────────────────────────
un = np.load(os.path.join(DATA_DIR, 'n_field.npy'))
vn = np.load(os.path.join(DATA_DIR, 'rho_field.npy'))
x  = np.load(os.path.join(DATA_DIR, 'x.npy'))
t  = np.load(os.path.join(DATA_DIR, 't.npy'))

# ── Build network ─────────────────────────────────────────────────────────────
Net = NN(Num_Hidden_Layers=5,
         Neurons_Per_Layer=50,
         Input_Dim=2,
         Output_Dim=2,
         Data_Type=torch.float32,
         Device=device,
         Activation_Function=Activation_function,
         Batch_Norm=False)

# ── Resolve best epoch ────────────────────────────────────────────────────────
model_save_dir  = os.path.join(PROJECT_ROOT, 'model_save', Equation_name,
                               f'{choose}_{noise_level}_{trail_num}({noise_type})')
best_epoch_path = os.path.join(model_save_dir, 'best_epoch.npy')
loss_file_path  = os.path.join(model_save_dir, 'loss.txt')

if os.path.exists(best_epoch_path):
    best_epoch = int(np.load(best_epoch_path)[0])
    print(f"Loaded best epoch from file: {best_epoch}")
elif os.path.exists(loss_file_path):
    print("best_epoch.npy not found — parsing loss.txt ...")
    best_epoch, best_loss = None, float('inf')
    with open(loss_file_path, 'r') as f:
        for line in f:
            m = re.search(r'iter_num:\s*(\d+).*loss_validate:\s*([0-9.]+)', line)
            if m:
                epoch    = int(m.group(1))
                val_loss = float(m.group(2))
                if val_loss < best_loss:
                    best_loss, best_epoch = val_loss, epoch
    if best_epoch is None:
        raise ValueError(f"No valid entries found in {loss_file_path}")
    print(f"Best epoch: {best_epoch}  (val_loss = {best_loss:.8f})")
    np.save(best_epoch_path, np.array([best_epoch]))
    print(f"Saved {best_epoch_path}")
else:
    raise FileNotFoundError(
        f"Neither best_epoch.npy nor loss.txt found in {model_save_dir}")

# ── Load normalisation stats ──────────────────────────────────────────────────
norm_stats_path = os.path.join(model_save_dir, 'norm_stats.npy')
if os.path.exists(norm_stats_path):
    n_mean, n_std, rho_mean, rho_std = np.load(norm_stats_path)
    print(f"Norm stats — n: mean={n_mean:.4e}, std={n_std:.4e} | "
          f"rho: mean={rho_mean:.4e}, std={rho_std:.4e}")
else:
    # Fall back to no normalisation (identity transform)
    print("Warning: norm_stats.npy not found — assuming unnormalised model.")
    n_mean,   n_std   = 0.0, 1.0
    rho_mean, rho_std = 0.0, 1.0

# ── Load best checkpoint ──────────────────────────────────────────────────────
best_model_path = os.path.join(model_save_dir, f"Net_{Activation_function}_{best_epoch}.pkl")
Net.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
print(f"Loaded checkpoint: {best_model_path}")

# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_surrogate_vs_truth(Net, un, vn, x, t, device, n_snapshots=4):
    Net.eval()
    t_indices = np.linspace(0, len(t) - 1, n_snapshots, dtype=int)
    fig, axs = plt.subplots(2, n_snapshots, figsize=(4 * n_snapshots, 6))

    with torch.no_grad():
        for col, ti in enumerate(t_indices):
            t_val    = t[ti]
            x_tensor = torch.tensor(
                np.column_stack([x, np.full_like(x, t_val)]),
                dtype=torch.float32, device=device)

            pred = Net(x_tensor).cpu().numpy()

            # ── Denormalise ───────────────────────────────────────
            n_pred   = pred[:, 0] * n_std   + n_mean
            rho_pred = pred[:, 1] * rho_std + rho_mean

            axs[0, col].plot(x, un[ti, :], label='truth',     color='black')
            axs[0, col].plot(x, n_pred,    label='surrogate', color='red', linestyle='--')
            axs[0, col].set_title(f't = {t_val:.1f} hr')
            axs[0, col].set_ylabel('n' if col == 0 else '')
            axs[0, col].legend(fontsize=7)

            axs[1, col].plot(x, vn[ti, :], label='truth',     color='black')
            axs[1, col].plot(x, rho_pred,  label='surrogate', color='blue', linestyle='--')
            axs[1, col].set_ylabel('rho' if col == 0 else '')
            axs[1, col].set_xlabel('x')
            axs[1, col].legend(fontsize=7)

    fig.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/surrogate_vs_truth.png', dpi=150, bbox_inches='tight')
    print('Saved plots/surrogate_vs_truth.png')
    plt.show()

plot_surrogate_vs_truth(Net, un, vn, x, t, device)