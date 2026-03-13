from neural_network import *
import os
import matplotlib.pyplot as plt


'''
Surrogate model training for the cylindrical drift-diffusion system.
    ∂n/∂t  = D∇²n + (D/x)∂n/∂x - v_s∂n/∂x - (v_s/x)n - β·ρ·n + α·n
    ∂ρ/∂t  = s·n
'''

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data', 'CylindricalDrift')

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ── Params ────────────────────────────────────────────────────────────────────
Equation_name       = 'CylindricalDrift'
noise_level         = 20
noise_type          = 'Gaussian'
trail_num           = 'run1'
Activation_function = 'Rational'
TRAIN_FRAC          = 0.8        # 80/20 split of full dataset

# ── Early stopping params ─────────────────────────────────────────────────────
PATIENCE      = 20     # checkpoints with no improvement
MIN_DELTA     = 1e-6   # minimum improvement to reset patience
OVERFIT_RATIO = 2.0    # stop if val_loss > OVERFIT_RATIO * train_loss

# ── Load data from FEniCS simulation ─────────────────────────────────────────
un = np.load(os.path.join(DATA_DIR, 'n_field.npy'))    # [nt, nx]
vn = np.load(os.path.join(DATA_DIR, 'rho_field.npy'))  # [nt, nx]
x  = np.load(os.path.join(DATA_DIR, 'x.npy'))          # [nx]
t  = np.load(os.path.join(DATA_DIR, 't.npy'))          # [nt]

print(f"Loaded data — x: {x.shape}, t: {t.shape}, n: {un.shape}, rho: {vn.shape}")

# ── Add noise (optional) ──────────────────────────────────────────────────────
if noise_level > 0:
    if noise_type == 'Gaussian':
        un += (noise_level / 100) * np.std(un) * np.random.randn(*un.shape)
        vn += (noise_level / 100) * np.std(vn) * np.random.randn(*vn.shape)
    elif noise_type == 'Uniform':
        un *= 1 + 0.01 * noise_level * np.random.uniform(-1, 1, un.shape)
        vn *= 1 + 0.01 * noise_level * np.random.uniform(-1, 1, vn.shape)

# ── Neural network: (x, t) -> (n, rho) ───────────────────────────────────────
torch.manual_seed(525)
if torch.cuda.is_available():
    torch.cuda.manual_seed(525)

Net = NN(Num_Hidden_Layers=5,
         Neurons_Per_Layer=50,
         Input_Dim=2,           # (x, t)
         Output_Dim=2,          # (n, rho)
         Data_Type=torch.float32,
         Device=device,
         Activation_Function=Activation_function,
         Batch_Norm=False)


# ── Surrogate training ────────────────────────────────────────────────────────
def train_surrogate_model(Net, un, vn):
    iter_num     = 50000
    total_points = un.size           # nt * nx = 36,000
    choose       = int(total_points * TRAIN_FRAC)
    choose_val   = total_points - choose
    print(f"Dataset — total: {total_points}, train: {choose}, validate: {choose_val}")

    model_save_dir = os.path.join(
        PROJECT_ROOT, 'model_save', Equation_name,
        f'{choose}_{noise_level}_{trail_num}({noise_type})')
    noise_data_dir = os.path.join(
        PROJECT_ROOT, 'noise_data_save', Equation_name,
        f'{choose}_{noise_level}({noise_type})')
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(noise_data_dir, exist_ok=True)

    # ── Persist / reload noisy data ───────────────────────────────
    noisy_n_path   = os.path.join(noise_data_dir, f'n_{noise_level}.npy')
    noisy_rho_path = os.path.join(noise_data_dir, f'rho_{noise_level}.npy')
    if not os.path.exists(noisy_n_path):
        np.save(noisy_n_path,   un)
        np.save(noisy_rho_path, vn)
    else:
        un = np.load(noisy_n_path)
        vn = np.load(noisy_rho_path)
        print('===load noisy data===')

    # ── Normalise using CLEAN data stats ──────────────────────────
    un_clean = np.load(os.path.join(DATA_DIR, 'n_field.npy'))
    vn_clean = np.load(os.path.join(DATA_DIR, 'rho_field.npy'))

    n_mean,   n_std   = un_clean.mean(), un_clean.std()
    rho_mean, rho_std = vn_clean.mean(), vn_clean.std()

    print(f"n   — clean std: {un_clean.std():.4e}, noisy std: {un.std():.4e}, "
          f"SNR: {un_clean.std() / abs(un.std() - un_clean.std() + 1e-12):.2f}")
    print(f"rho — clean std: {vn_clean.std():.4e}, noisy std: {vn.std():.4e}, "
          f"SNR: {vn_clean.std() / abs(vn.std() - vn_clean.std() + 1e-12):.2f}")

    un_norm = (un - n_mean)   / n_std
    vn_norm = (vn - rho_mean) / rho_std

    np.save(os.path.join(model_save_dir, 'norm_stats.npy'),
            np.array([n_mean, n_std, rho_mean, rho_std]))
    print(f"Norm stats (clean) — n: mean={n_mean:.4e}, std={n_std:.4e} | "
          f"rho: mean={rho_mean:.4e}, std={rho_std:.4e}")

    # ── Build full grid and shuffle once — guaranteed no overlap ──
    XX, TT = np.meshgrid(x, t)
    X_flat  = XX.flatten()
    T_flat  = TT.flatten()
    N_flat  = un_norm.flatten()
    R_flat  = vn_norm.flatten()

    np.random.seed(525)
    all_indices = np.random.permutation(total_points)
    train_idx   = all_indices[:choose]
    val_idx     = all_indices[choose:]

    database_choose   = torch.tensor(
        np.column_stack([X_flat[train_idx], T_flat[train_idx]]),
        dtype=torch.float32).to(device).requires_grad_(True)
    h_data_choose     = torch.tensor(
        np.column_stack([N_flat[train_idx], R_flat[train_idx]]),
        dtype=torch.float32).to(device)
    database_validate = torch.tensor(
        np.column_stack([X_flat[val_idx], T_flat[val_idx]]),
        dtype=torch.float32).to(device).requires_grad_(True)
    h_data_validate   = torch.tensor(
        np.column_stack([N_flat[val_idx], R_flat[val_idx]]),
        dtype=torch.float32).to(device)

    # ── Paths ─────────────────────────────────────────────────────
    origin_model_path = os.path.join(model_save_dir, f"Net_{Activation_function}_origin.pkl")
    loss_file_path    = os.path.join(model_save_dir, 'loss.txt')
    best_epoch_path   = os.path.join(model_save_dir, 'best_epoch.npy')
    torch.save(Net.state_dict(), origin_model_path)

    NN_optimizer = torch.optim.Adam(Net.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(
                       NN_optimizer, T_max=iter_num, eta_min=1e-6)
    MSELoss        = torch.nn.MSELoss()
    validate_error = []

    # ── Early stopping state ───────────────────────────────────────
    best_val_loss    = float('inf')
    patience_counter = 0
    stop_reason      = 'completed'

    print('=============== train Net =================')
    with open(loss_file_path, 'w') as log_file:
        for iter in range(iter_num):
            NN_optimizer.zero_grad()
            prediction = Net(database_choose)
            loss       = MSELoss(h_data_choose, prediction)
            loss.backward()
            NN_optimizer.step()
            scheduler.step()          # ← every iteration for cosine schedule

            if (iter + 1) % 500 == 0:
                with torch.no_grad():
                    prediction_validate = Net(database_validate)
                    loss_validate = MSELoss(h_data_validate, prediction_validate).item()

                loss_val   = loss.item()
                current_lr = NN_optimizer.param_groups[0]['lr']
                validate_error.append(loss_validate)

                iter_model_path = os.path.join(model_save_dir,
                                               f"Net_{Activation_function}_{iter + 1}.pkl")
                torch.save(Net.state_dict(), iter_model_path)

                log_line = ("iter_num: %d      loss: %.8f    loss_validate: %.8f    lr: %.2e\n"
                            % (iter + 1, loss_val, loss_validate, current_lr))
                print(log_line, end='')
                log_file.write(log_line)
                log_file.flush()

                # ── Track best & save incrementally ───────────────
                if loss_validate < best_val_loss - MIN_DELTA:
                    best_val_loss    = loss_validate
                    patience_counter = 0
                    np.save(best_epoch_path, np.array([iter + 1]))
                else:
                    patience_counter += 1

                # ── Early stopping checks ──────────────────────────
                if patience_counter >= PATIENCE:
                    stop_reason = f'patience ({PATIENCE} checkpoints without improvement)'
                    log_file.write(f'Early stop: {stop_reason}\n')
                    break

                if loss_validate > OVERFIT_RATIO * loss_val:
                    stop_reason = (f'overfitting (val/train ratio '
                                   f'{loss_validate / loss_val:.2f} > {OVERFIT_RATIO})')
                    log_file.write(f'Early stop: {stop_reason}\n')
                    break

    best_epoch = (validate_error.index(min(validate_error)) + 1) * 500
    print(f'Stopped: {stop_reason}')
    print(f'Best epoch: {best_epoch}  (val_loss = {min(validate_error):.8f})')
    np.save(best_epoch_path, np.array([best_epoch]))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    train_surrogate_model(Net, un, vn)