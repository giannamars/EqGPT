'''
Neural network definitions and data utilities for Burgers_2D surrogate model.
'''

import numpy as np
import torch
import random
import os
from torch.autograd import Variable
import torch.nn as nn
import math

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


# ── Activation function: Rational ─────────────────────────────────────────────
class Rational(torch.nn.Module):
    """
    Learnable rational activation function R(x) = N(x)/D(x) where:
        N(x) = a0 + a1*x + a2*x^2 + a3*x^3
        D(x) = b0 + b1*x + b2*x^2
    Coefficients are initialised to approximate ReLU and learned during training.
    Reference: Boulle et al., "Rational neural networks" arXiv:2004.01902 (2020).
    """
    def __init__(self,
                 Data_Type = torch.float32,
                 Device    = torch.device('cpu')):
        super(Rational, self).__init__()
        self.a = torch.nn.parameter.Parameter(
                    torch.tensor((1.1915, 1.5957, 0.5, .0218),
                                 dtype=Data_Type, device=Device))
        self.b = torch.nn.parameter.Parameter(
                    torch.tensor((2.3830, 0.0, 1.0),
                                 dtype=Data_Type, device=Device))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        a, b = self.a, self.b
        N_X = a[0] + X*(a[1] + X*(a[2] + a[3]*X))
        D_X = b[0] + X*(b[1] + b[2]*X)
        return N_X / D_X


# ── Surrogate network ─────────────────────────────────────────────────────────
class NN(torch.nn.Module):
    """
    MLP surrogate network mapping (x, y, t) -> u.
    For Burgers_2D: Input_Dim=3, Output_Dim=1, 5 hidden layers x 50 neurons,
    Rational activation, no batch norm.
    """
    def __init__(self,
                 Num_Hidden_Layers   : int          = 3,
                 Neurons_Per_Layer   : int          = 20,
                 Input_Dim           : int          = 1,
                 Output_Dim          : int          = 1,
                 Data_Type           : torch.dtype  = torch.float32,
                 Device              : torch.device = torch.device('cpu'),
                 Activation_Function : str          = "Rational",
                 Batch_Norm          : bool         = False):

        assert Num_Hidden_Layers > 0
        assert Neurons_Per_Layer > 0
        assert Input_Dim         > 0
        assert Output_Dim        > 0

        super(NN, self).__init__()

        self.Num_Hidden_Layers = Num_Hidden_Layers
        self.Batch_Norm        = Batch_Norm

        # ── Layers ────────────────────────────────────────────────
        self.Layers = torch.nn.ModuleList()

        # Input → first hidden
        self.Layers.append(torch.nn.Linear(Input_Dim, Neurons_Per_Layer)
                           .to(dtype=Data_Type, device=Device))
        # Hidden → hidden
        for _ in range(1, Num_Hidden_Layers):
            self.Layers.append(torch.nn.Linear(Neurons_Per_Layer, Neurons_Per_Layer)
                               .to(dtype=Data_Type, device=Device))
        # Last hidden → output
        self.Layers.append(torch.nn.Linear(Neurons_Per_Layer, Output_Dim)
                           .to(dtype=Data_Type, device=Device))

        # ── Weight initialisation (Xavier with gain=1.41 for Rational) ───
        for i in range(Num_Hidden_Layers + 1):
            torch.nn.init.xavier_normal_(self.Layers[i].weight, gain=1.41)
            torch.nn.init.zeros_(self.Layers[i].bias)

        # ── Activation functions ───────────────────────────────────
        self.Activation_Functions = torch.nn.ModuleList(
            [Rational(Data_Type=Data_Type, Device=Device)
             for _ in range(Num_Hidden_Layers)]
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        for i in range(self.Num_Hidden_Layers):
            X = self.Activation_Functions[i](self.Layers[i](X))
        return self.Layers[self.Num_Hidden_Layers](X)


def random_data_1D_coupled(choose, choose_validate, x, t, un, vn, random_seed=525):
    """
    Flattens the 2D fields un[t, x] and vn[t, x] into a list of
    (x, t) -> (n, rho) samples and splits randomly into training
    and validation sets.

    Inputs:
        x:   [nx]      spatial coordinates
        t:   [nt]      time coordinates
        un:  [nt, nx]  n field
        vn:  [nt, nx]  rho field

    Returns: h_data_choose, h_data_validate, database_choose, database_validate
             all as float32 tensors, with h_data of shape [N, 2] and
             database of shape [N, 2].
    """
    x_num, t_num = x.shape[0], t.shape[0]
    total = x_num * t_num

    random.seed(random_seed)
    np.random.seed(random_seed)

    h_data   = np.zeros([total, 2])   # (n, rho)
    database = np.zeros([total, 2])   # (x, t)
    num = 0
    for i in range(t_num):
        for j in range(x_num):
            database[num, 0] = x[j]
            database[num, 1] = t[i]
            h_data[num,   0] = un[i, j]   # n
            h_data[num,   1] = vn[i, j]   # rho
            num += 1

    # ── Shuffle database and h_data with the same permutation ─────
    state = np.random.get_state()
    np.random.shuffle(database)
    np.random.set_state(state)
    np.random.shuffle(h_data)

    def to_tensor(arr):
        return torch.from_numpy(arr.astype(np.float32))

    return (to_tensor(h_data[0:choose]),
            to_tensor(h_data[choose:choose + choose_validate]),
            to_tensor(database[0:choose]),
            to_tensor(database[choose:choose + choose_validate]))


def random_data_2D(choose,choose_validate,x,y,t,un,random_seed=525):
    x_num=x.shape[0]
    y_num=y.shape[0]
    t_num=t.shape[0]
    total=x_num*y_num*t_num
    random.seed(random_seed)
    data=np.zeros(3)
    h_data=np.zeros([total,1])
    database=np.zeros([total,3])
    num=0


    for j in range(x_num):
        for k in range(y_num):
            for i in range(t_num):
                data[0]=x[j]
                data[1]=y[k]
                data[2]=t[i]
                h_data[num]=un[i,k,j]
                database[num]=data
                num+=1

    state = np.random.get_state()
    np.random.shuffle(database)
    np.random.set_state(state)
    np.random.shuffle(h_data)
    h_data_choose = h_data[0:choose]
    database_choose = database[0:choose]
    h_data_validate = h_data[choose:choose + choose_validate]
    database_validate = database[choose:choose + choose_validate]

    h_data_choose = torch.from_numpy(h_data_choose.astype(np.float32))
    database_choose = torch.from_numpy(database_choose.astype(np.float32))
    h_data_validate = torch.from_numpy(h_data_validate.astype(np.float32))
    database_validate = torch.from_numpy(database_validate.astype(np.float32))
    return h_data_choose,h_data_validate,database_choose,database_validate