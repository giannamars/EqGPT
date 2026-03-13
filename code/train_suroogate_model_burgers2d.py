import scipy.io as scio
from neural_network import *
import os

'''
Surrogate model training for the 2D Burgers equation.
'''

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data')

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ── Params ────────────────────────────────────────────────────────────────────
Equation_name       = 'Burgers_2D'
choose              = 10000
noise_level         = 50
noise_type          = 'Gaussian'   # 'Gaussian' or 'Uniform'
trail_num           = 'PIS'
Activation_function = 'Rational'
choose_validate     = 5000

# ── Load data ─────────────────────────────────────────────────────────────────
data_path = os.path.join(DATA_DIR, Equation_name, 'Burgers2D.mat')
data      = scio.loadmat(data_path)
x  = np.squeeze(data["x"])
y  = np.squeeze(data["y"])
t  = np.squeeze(data["t"])
un = data["u"]

# ── Add noise ─────────────────────────────────────────────────────────────────
if noise_type == 'Gaussian':
    un += (noise_level / 100) * np.std(un) * np.random.randn(*un.shape)
elif noise_type == 'Uniform':
    un *= 1 + 0.01 * noise_level * np.random.uniform(-1, 1, un.shape)

# ── Neural network ────────────────────────────────────────────────────────────
torch.manual_seed(525)
if torch.cuda.is_available():
    torch.cuda.manual_seed(525)

Net = NN(Num_Hidden_Layers=5,
         Neurons_Per_Layer=50,
         Input_Dim=3,
         Output_Dim=1,
         Data_Type=torch.float32,
         Device=device,
         Activation_Function=Activation_function,
         Batch_Norm=False)


# ── Surrogate training ────────────────────────────────────────────────────────
def train_surrogate_model(Net, un):
    model_save_dir = os.path.join(
        PROJECT_ROOT, 'model_save', Equation_name,
        f'{choose}_{noise_level}_{trail_num}({noise_type})')
    noise_data_dir = os.path.join(
        PROJECT_ROOT, 'noise_data_save', Equation_name,
        f'{choose}_{noise_level}({noise_type})')
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(noise_data_dir, exist_ok=True)

    # ── Persist / reload noisy data ───────────────────────────────
    noisy_data_path = os.path.join(noise_data_dir, f'un_{noise_level}.npy')
    if not os.path.exists(noisy_data_path):
        np.save(noisy_data_path, un)
    else:
        un = np.load(noisy_data_path)
        print('===load noisy data===')

    # ── Random dataset ────────────────────────────────────────────
    iter_num = 50000
    h_data_choose, h_data_validate, database_choose, database_validate = \
        random_data_2D(choose, choose_validate, x, y, t, un)

    database_choose   = Variable(database_choose.to(device),   requires_grad=True)
    database_validate = Variable(database_validate.to(device), requires_grad=True)
    h_data_choose     = h_data_choose.to(device)
    h_data_validate   = h_data_validate.to(device)

    # ── Paths ─────────────────────────────────────────────────────
    origin_model_path = os.path.join(model_save_dir, f"Net_{Activation_function}_origin.pkl")
    loss_file_path    = os.path.join(model_save_dir, 'loss.txt')
    best_epoch_path   = os.path.join(model_save_dir, 'best_epoch.npy')
    torch.save(Net.state_dict(), origin_model_path)

    NN_optimizer   = torch.optim.Adam(Net.parameters())
    MSELoss        = torch.nn.MSELoss()
    validate_error = []

    print('=============== train Net =================')
    with open(loss_file_path, 'w') as log_file:
        for iter in range(iter_num):
            NN_optimizer.zero_grad()
            prediction = Net(database_choose)
            loss       = MSELoss(h_data_choose, prediction)
            loss.backward()
            NN_optimizer.step()

            if (iter + 1) % 500 == 0:
                with torch.no_grad():
                    prediction_validate = Net(database_validate)
                    loss_validate = MSELoss(h_data_validate, prediction_validate).item()

                validate_error.append(loss_validate)
                loss_val        = loss.item()
                iter_model_path = os.path.join(model_save_dir,
                                               f"Net_{Activation_function}_{iter + 1}.pkl")
                torch.save(Net.state_dict(), iter_model_path)

                log_line = ("iter_num: %d      loss: %.8f    loss_validate: %.8f\n"
                            % (iter + 1, loss_val, loss_validate))
                print(log_line, end='')
                log_file.write(log_line)
                log_file.flush()

    best_epoch = (validate_error.index(min(validate_error)) + 1) * 500
    print('Best epoch:', best_epoch)
    np.save(best_epoch_path, np.array([best_epoch]))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    train_surrogate_model(Net, un)