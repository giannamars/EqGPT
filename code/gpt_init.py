import scipy.io as scio
from neural_network import *
from train_gpt import *
import matplotlib.pyplot as plt
import os

'''
Surrogate model training for the 2D Burgers equation.
The surrogate model is used to generate meta data and calculate smooth derivatives.
'''

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data')

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ── Params ────────────────────────────────────────────────────────────────────
Equation_name        = 'Burgers_2D'
choose               = 10000
noise_level          = 50
noise_type           = 'Gaussian'   # 'Gaussian' or 'Uniform'
trail_num            = 'PIS'
Learning_Rate        = 0.001
Delete_equation_name = 'Burgers_2D'
Activation_function  = 'Rational'
choose_validate      = 5000
meta_data_num        = 10000

# ── Load data ─────────────────────────────────────────────────────────────────
data_path = os.path.join(DATA_DIR, Equation_name, 'Burgers2D.mat')
data      = scio.loadmat(data_path)
x  = np.squeeze(data["x"])
y  = np.squeeze(data["y"])
t  = np.squeeze(data["t"])
un = data["u"]
x_low, x_up = -0.8,  0.8
y_low, y_up = -0.8,  0.8
t_low, t_up =  0.0,  2.0
target = [[2], [0, 1]]
Left   = 'u_t'
epi    = 1e-6

x_num = x.shape[0]
y_num = y.shape[0]
t_num = t.shape[0]

# ── Add noise ─────────────────────────────────────────────────────────────────
if noise_type == 'Gaussian':
    un += (noise_level / 100) * np.std(un) * np.random.randn(*un.shape)
elif noise_type == 'Uniform':
    un *= 1 + 0.01 * noise_level * np.random.uniform(-1, 1, un.shape)

# ── Train GPT (keeping target equation unseen) ───────────────────────────────
gpt_model_path = os.path.join(PROJECT_ROOT, 'gpt_model', f'PDEGPT_{Equation_name}.pt')
if not os.path.exists(gpt_model_path):
    train_num_data = get_train_dataset(Equation_name=Delete_equation_name)
    batch_size     = 128
    epochs         = 100
    dataset        = MyDataSet(train_num_data)
    data_loader    = Data.DataLoader(dataset, batch_size=batch_size,
                                     collate_fn=dataset.padding_batch)
    model = GPT().to(device)
    train(model, data_loader, Equation_name=Equation_name)

# ── Neural network (inputs: x, y, t) ─────────────────────────────────────────
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


# ── Meta-data grid generation ─────────────────────────────────────────────────
def Generate_meta_data_2D(Net, Load_state, nx=20, ny=20, nt=20):
    model_load_path = os.path.join(
        PROJECT_ROOT, 'model_save', Equation_name,
        f'{choose}_{noise_level}_{trail_num}({noise_type})',
        f'{Load_state}.pkl')
    Net.load_state_dict(torch.load(model_load_path, map_location=device))
    Net.eval()

    xs = torch.linspace(x_low, x_up, nx)
    ys = torch.linspace(y_low, y_up, ny)
    ts = torch.linspace(t_low, t_up, nt)
    total = nx * ny * nt

    database = torch.zeros([total, 3])
    num = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nt):
                database[num, 0] = xs[i]
                database[num, 1] = ys[j]
                database[num, 2] = ts[k]
                num += 1

    database = Variable(database, requires_grad=True).to(device)
    return Net, database


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

    # ── Save initial weights & paths ──────────────────────────────
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


# ── Meta-data retrieval ───────────────────────────────────────────────────────
def get_meta(Net):
    best_epoch_path = os.path.join(
        PROJECT_ROOT, 'model_save', Equation_name,
        f'{choose}_{noise_level}_{trail_num}({noise_type})',
        'best_epoch.npy')
    best_epoch = np.load(best_epoch_path)[0]
    print("best_epoch:", best_epoch)

    Load_state   = f'Net_{Activation_function}_{best_epoch}'
    Net, database = Generate_meta_data_2D(Net, Load_state)
    return Net, database


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    train_surrogate_model(Net, un)