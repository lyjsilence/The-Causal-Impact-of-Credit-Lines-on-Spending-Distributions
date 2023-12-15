import numpy as np
import torch
import copy
import torch.nn as nn
from scipy.interpolate import BSpline

ACTIVATIONS = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "sigmoid": nn.Sigmoid(),
    "softsign": nn.Softsign(),
    "selu": nn.SELU(),
    "softmax": nn.Softmax(dim=1)
}


class LayerNorm(nn.Module):
    def __init__(self, hidden, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.hidden = hidden
        self.eps = eps
        self.alpha = nn.Parameter(torch.randn(hidden))
        self.beta = nn.Parameter(torch.randn(hidden))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return (x-mean) / std * self.alpha + self.beta


# Numerical MLP
class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden=[64, 64, 64], activation='relu', dropout=0.1, device='cpu'):
        super(MLP, self).__init__()
        self.activation = ACTIVATIONS[activation]
        self.dim = [dim_in] + dim_hidden + [dim_out]
        self.linears = nn.ModuleList([nn.Linear(self.dim[i-1], self.dim[i]) for i in range(1, len(self.dim))])
        self.layernorms = nn.ModuleList([LayerNorm(hidden) for hidden in dim_hidden])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(len(dim_hidden))])

    def forward(self, x):
        for i in range(len(self.dim)-2):
            x = self.linears[i](x)
            x = x + self.layernorms[i](x)
            x = self.activation(x)
            x = self.dropouts[i](x)
        x = self.linears[-1](x)
        return x

# Functional_MLP
class Functional_MLP(nn.Module):
    def __init__(self, dim_out, t=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], device='cpu'):
        super(Functional_MLP, self).__init__()
        self.dim_out = dim_out
        self.device = device

        spline_deg = 3
        self.num_basis = len(t) - spline_deg - 1
        self.basis_layer = []
        for i in range(self.num_basis):
            const_basis = np.zeros(self.num_basis)
            const_basis[i] = 1.0
            self.basis_layer.append(BSpline(np.array(t), const_basis, spline_deg))
        # the weights for each basis
        self.alpha = nn.Parameter(torch.randn(self.dim_out, self.num_basis))
        torch.nn.init.xavier_uniform_(self.alpha)

        self.t = torch.tensor(t).to(self.device)
    def forward(self, x):
        t = self.t.unsqueeze(1).cpu().detach().numpy()
        self.bases = [torch.tensor(basis(t).transpose(-1, -2)).to(torch.float32).to(self.device) for basis in self.basis_layer]
        y = 0
        for j in range(x.shape[1]):
            betas = torch.sum(torch.cat([self.alpha[j][k] * self.bases[k] for k in range(self.num_basis)]), dim=0, keepdim=True)
            y += x[:, j].unsqueeze(1).repeat([1, betas.shape[1]]) * betas
        return y

# Neural Functional Regression for arbitrary quantile
class NFR(nn.Module):
    def __init__(self, dim_in, dim_out, t=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dim_hidden=[64, 64, 64],
                 activation='relu', dropout=0.1, device='cpu'):
        super(NFR, self).__init__()
        '''
        dim_in: dimension of features
        dim_out: dimension of output from numerical MLP
        num_basis: number of basis used
        '''
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.device = device
        self.activation = ACTIVATIONS[activation]
        self.dropout = dropout

        # Numerical MLP for scalar input and representation learning
        self.MLP = MLP(dim_in, dim_out, dim_hidden, activation=activation, dropout=self.dropout)

        # Functional MLP for scalar input and functional output
        self.Functional_MLP = Functional_MLP(dim_out, t, device=device)


    def forward(self, X):
        X = self.MLP(X)
        X = self.Functional_MLP(X)
        return X


# Neural Functional Regression with representation learning for arbitrary quantile
class rep_NFR(nn.Module):
    def __init__(self, dim_in, dim_out, t=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dim_hidden_rep=[64, 64, 64],
                 dim_hidden_head=[32, 32], num_treatment=5, activation='relu', dropout=0.1, device='cpu'):
        super(rep_NFR, self).__init__()
        '''
        dim_in: dimension of features
        dim_out: dimension of output from numerical MLP
        num_basis: number of basis used
        '''
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_treatment = num_treatment
        self.t = t
        self.device = device
        self.activation = ACTIVATIONS[activation]
        self.dropout = dropout

        # Numerical MLP for scalar input and representation learning
        self.MLP = MLP(dim_in, dim_out, dim_hidden_rep, activation=activation, dropout=self.dropout)

        # headers
        headers = []
        for i in range(self.num_treatment):
            headers.append(nn.Sequential(
                               MLP(dim_out, dim_in, dim_hidden_head, activation=activation, dropout=self.dropout),
                               Functional_MLP(dim_in, t, device=device)
                           ))
        self.headers = nn.ModuleList(headers)

    def forward(self, X):
        X, D = X[:, :-1], X[:, -1]
        X = self.MLP(X)
        y = torch.zeros([X.shape[0], len(self.t)]).to(self.device)
        for i in range(self.num_treatment):
            y += self.headers[i](X) * ((D == i) * 1.0).unsqueeze(1)
        return y
