import os
import numpy as np
import pandas as pd
import random
import torch
import argparse
from sklearn.model_selection import KFold
from models import NFR, MLP, rep_NFR
from baseline_models import est_ECDF, Cross_validate, Regression, propensity_score
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils import sim_dataset_cdf, sim_dataset_reg
from training import train_regression, train_classification, train_classical
from torch.utils.data.sampler import SubsetRandomSampler
import warnings
warnings.filterwarnings('ignore')

from scipy.stats import beta


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


class DataGenerator:
    def __init__(self, args):
        self.sample_size = args.sample_size
        self.num_cov = args.num_cov
        self.num_obs = args.num_obs
        self.num_treatment = args.n_treat
        self.obs_dim = args.obs_dim
        self.save_path = args.data_save_path
        self.balance_level = args.balance
        self.phi = 0.2
        self.c = 0

        self.basis_list = [self.inv_beta_1, self.inv_beta_2, self.inv_beta_3, self.inv_beta_4, self.inv_beta_5]

    def inv_beta_1(self, alpha):
        return beta.ppf(alpha, 0.1, 0.1)

    def inv_beta_2(self, alpha):
        return beta.ppf(alpha, 0.2, 0.2)

    def inv_beta_3(self, alpha):
        return beta.ppf(alpha, 0.3, 0.3)

    def inv_beta_4(self, alpha):
        return beta.ppf(alpha, 0.4, 0.4)

    def inv_beta_5(self, alpha):
        return beta.ppf(alpha, 0.5, 0.5)

    def non_linear_x_fun(self, x):
        return np.exp(x / self.phi) / np.sum(np.exp(x / self.phi)).reshape(-1, 1)

    def inverse_cdf(self, alpha, i, ground=None):
        self.D_mean = self.D.mean()
        x = np.array([self.X[i, 2*j] * self.X[i, 2*j+1] for j in range(int(self.num_cov/2))])
        non_linear_x = self.non_linear_x_fun(x)
        basis = np.stack([self.basis_list[k](alpha) for k in range(len(self.basis_list))])
        sum_basis = np.matmul(non_linear_x, basis)
        if ground is not None:
            obs = self.c + (1-self.c) * (self.D_mean + np.sqrt(ground)) * sum_basis
        else:
            obs = self.c + (1-self.c) * (self.D_mean + np.sqrt(self.D[i])) * sum_basis
        return obs

    def create_D(self, X):
        if self.balance_level == 0:
            coef_0 = np.array([np.sqrt(1/10)] * 10).reshape(-1, 1) # mean=0, var=1
            coef_1 = np.array([np.sqrt(1/10)] * 10).reshape(-1, 1) # mean=0, var=1
            coef_2 = np.array([np.sqrt(1/10)] * 10).reshape(-1, 1) # mean=0, var=1
            coef_3 = np.array([np.sqrt(1/10)] * 10).reshape(-1, 1) # mean=0, var=1
            coef_4 = np.array([np.sqrt(1/10)] * 10).reshape(-1, 1) # mean=0, var=1
        elif self.balance_level == 1:
            coef_0 = np.array([0.289,0.289,0.450,0.450,0.260,0.260,0.250,0.250,0.289,0.289]).reshape(-1, 1) # mean=-0.4, var=1
            coef_1 = np.array([0.289,0.289,0.400,0.400,0.289,0.289,0.300,0.300,0.289,0.289]).reshape(-1, 1) # mean=-0.2, var=1
            coef_2 = np.array([np.sqrt(0.1)] * 10).reshape(-1, 1) # mean=0, var=1
            coef_3 = np.array([0.289,0.289,0.400,0.400,0.289,0.289,0.300,0.300,0.289,0.289]).reshape(-1, 1) # mean=0.2, var=1
            coef_4 = np.array([0.289,0.289,0.250,0.250,0.260,0.260,0.450,0.450,0.289,0.289]).reshape(-1, 1) # mean=0.4, var=1
        elif self.balance_level == 2:
            coef_0 = np.array([0.289,0.289,0.400,0.400,0.400,0.400,0.100,0.100,0.289,0.289]).reshape(-1, 1) # mean=-0.6, var=1
            coef_1 = np.array([0.289,0.289,0.450,0.450,0.260,0.260,0.250,0.250,0.289,0.289]).reshape(-1, 1) # mean=-0.4, var=1
            coef_2 = np.array([np.sqrt(0.1)] * 10).reshape(-1, 1) # mean=0, var=1
            coef_3 = np.array([0.289,0.289,0.250,0.250,0.260,0.260,0.450,0.450,0.289,0.289]).reshape(-1, 1) # mean=0.4, var=1
            coef_4 = np.array([0.289,0.289,0.100,0.100,0.400,0.400,0.400,0.400,0.289,0.289]).reshape(-1, 1) # mean=0.6, var=1
        elif self.balance_level == 3:
            coef_0 = np.array([0.289,0.289,0.500,0.500,0.270,0.270,0.100,0.100,0.289,0.289]).reshape(-1, 1) # mean=-0.8, var=1
            coef_1 = np.array([0.289,0.289,0.450,0.450,0.260,0.260,0.250,0.250,0.289,0.289]).reshape(-1, 1) # mean=-0.4, var=1
            coef_2 = np.array([np.sqrt(0.1)] * 10).reshape(-1, 1) # mean=0, var=1
            coef_3 = np.array([0.289,0.289,0.250,0.250,0.260,0.260,0.450,0.450,0.289,0.289]).reshape(-1, 1) # mean=0.4, var=1
            coef_4 = np.array([0.289,0.289,0.100,0.100,0.270,0.270,0.500,0.500,0.289,0.289]).reshape(-1, 1) # mean=0.8, var=1
        elif self.balance_level == 4:
            coef_0 = np.array([0.289,0.289,0.500,0.500,0.289,0.289,0.000,0.000,0.289,0.289]).reshape(-1, 1) # mean=-1, var=1
            coef_1 = np.array([0.289,0.289,0.400,0.400,0.400,0.400,0.100,0.100,0.289,0.289]).reshape(-1, 1) # mean=-0.6, var=1
            coef_2 = np.array([np.sqrt(0.1)] * 10).reshape(-1, 1) # mean=0, var=1
            coef_3 = np.array([0.289,0.289,0.100,0.100,0.400,0.400,0.400,0.400,0.289,0.289]).reshape(-1, 1) # mean=0.6, var=1
            coef_4 = np.array([0.289,0.289,0.000,0.000,0.289,0.289,0.500,0.500,0.289,0.289]).reshape(-1, 1) # mean=1, var=1
        elif self.balance_level == 5:
            coef_0 = np.array([0.350,0.350,0.450,0.450,0.380,0.380,0.100,0.100,0.150,0.150]).reshape(-1, 1) # mean=-1.5, var=1
            coef_1 = np.array([0.289,0.289,0.500,0.500,0.270,0.270,0.100,0.100,0.289,0.289]).reshape(-1, 1) # mean=-0.8, var=1
            coef_2 = np.array([np.sqrt(0.1)] * 10).reshape(-1, 1) # mean=0, var=1
            coef_3 = np.array([0.289,0.289,0.100,0.100,0.270,0.270,0.500,0.500,0.289,0.289]).reshape(-1, 1) # mean=0.8, var=1
            coef_4 = np.array([0.150,0.150,0.100,0.100,0.380,0.380,0.450,0.450,0.350,0.350]).reshape(-1, 1) # mean=1.5, var=1
        combine_z0 = np.dot(X, coef_0)
        combine_z1 = np.dot(X, coef_1)
        combine_z2 = np.dot(X, coef_2)
        combine_z3 = np.dot(X, coef_3)
        combine_z4 = np.dot(X, coef_4)
        total_sum = np.exp(combine_z0) + np.exp(combine_z1) + np.exp(combine_z2) + np.exp(combine_z3) + np.exp(combine_z4)
        true_prob_0 = np.exp(combine_z0) / total_sum
        true_prob_1 = np.exp(combine_z1) / total_sum
        true_prob_2 = np.exp(combine_z2) / total_sum
        true_prob_3 = np.exp(combine_z3) / total_sum
        true_prob_4 = np.exp(combine_z4) / total_sum

        prob = np.hstack((true_prob_0, true_prob_1, true_prob_2, true_prob_3, true_prob_4))
        D_onehot = np.array(list(map(lambda x: np.random.multinomial(n=1, pvals=x), prob)))
        D = np.squeeze(np.array(list(map(lambda x: np.argwhere(x == 1), D_onehot)))).reshape(-1, 1)
        return D

    def generate(self):
        # generate covariates by assuming all features follows normal distribution with specific mean and variance
        self.X = np.zeros([self.sample_size, self.num_cov])
        self.X[:, 0:2] = np.random.normal(loc=-2, scale=1, size=[self.sample_size, 2])
        self.X[:, 2:4] = np.random.normal(loc=-1, scale=1, size=[self.sample_size, 2])
        self.X[:, 4:6] = np.random.normal(loc=0, scale=1, size=[self.sample_size, 2])
        self.X[:, 6:8] = np.random.normal(loc=1, scale=1, size=[self.sample_size, 2])
        self.X[:, 8:10] = np.random.normal(loc=2, scale=1, size=[self.sample_size, 2])

        # generate treatment
        self.D = self.create_D(self.X)

        # generate observations by basis function
        y_list = []
        for idx in range(self.sample_size):
            alpha = np.random.uniform(low=0, high=1, size=self.num_obs)
            y_list.append(self.inverse_cdf(alpha, idx).squeeze() + np.random.normal(loc=0, scale=0.05, size=self.num_obs))
        self.Y = np.stack(y_list)

        return self.X, self.D, self.Y

    def plot(self, num_sample):
        sample = np.random.choice(self.sample_size, num_sample, replace=False)
        for idx in sample:
            alpha = np.linspace(0, 1, self.num_obs)
            obs_no = self.inverse_cdf(alpha, idx).squeeze()
            obs = obs_no + np.random.normal(loc=0, scale=0.05, size=self.num_obs)
            plt.plot(alpha, obs_no, label=f'instance {idx}')
            plt.scatter(alpha, obs, s=5, alpha=0.5)

        plt.xlabel('quantiles')
        plt.ylabel('$Y^{-1}$')
        plt.legend()
        plt.grid()
        plt.savefig('instances.pdf')
        plt.show()

    def groud_truth(self, quantiles):
        alpha = quantiles
        y_inv_mean_list = []

        for ground_D in range(self.num_treatment):
            y_inv_list = []
            for idx in range(self.sample_size):
                y_inv_list.append(self.inverse_cdf(alpha, idx, ground=ground_D).squeeze())

            y_inv_mean_list.append(np.mean(y_inv_list, axis=0).reshape(1, 9))

        return np.stack(y_inv_mean_list, axis=0).squeeze()



parser = argparse.ArgumentParser(description='Simulation experiment on causal function')

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--data_save_path', type=str, default='data')
parser.add_argument('--results_save_path', type=str, default='results')

parser.add_argument('--sample_size', type=int, default=5000, help='Number of samples')
parser.add_argument('--num_obs', type=int, default=100, help='Number of observations (y) for each sample')
parser.add_argument('--num_cov', type=int, default=10, help='Number of covariates')
parser.add_argument('--n_treat', type=int, default=5, help='Number of categories that a treatment can take')

parser.add_argument('--obs_dim', type=int, default=9)
parser.add_argument('--balance', type=int, default=3)

parser.add_argument('--act_fn', type=str, default="tanh", choices=["tanh", "softplus", "elu", "relu"])

parser.add_argument('--batch_size', type=int, default=128)

parser.add_argument('--lr', type=float, default=0.003)

parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=150)

parser.add_argument('--dropout', type=float, default=0.1)

parser.add_argument('--reg_met', type=str, default='Classical', choices=['MCP', 'lasso', 'ridge', 'elastic net', 'MLP', 'NFR', 'rep_NFR', 'Classical'])
parser.add_argument('--class_met', type=str, default='Random_forest', choices=['MLP', 'Random_forest', 'logistic_regression'])
parser.add_argument('--start_exp', type=int, default=0)
parser.add_argument('--end_exp', type=int, default=100)

args = parser.parse_args()


def compute_MAE(true, pred):
    return np.mean(np.abs(np.array(true) - np.array(pred)))


def compute_RE(true, pred):
    return np.mean(np.abs((np.array(true) - np.array(pred)) / np.array(true)))


class Estimator:
    def __init__(self, Y_inv, D, X, quantiles, K_fold, reg_met):
        self.K_fold = K_fold
        self.quantiles = quantiles
        self.Phi = self.B_spline(quantiles)
        self.N = X.shape[0]
        self.Y_inv = Y_inv
        self.X = X
        self.D = D.reshape(-1, 1)
        self.XD = np.hstack([X, self.D])
        self.num_treatment = len(np.unique(D))
        self.reg_met = reg_met

        self.args = args

        self.lambdas = [1, 0.1, 0.01, 0.001]
        self.alphas = [0.001, 0.01, 0.1, 1]
        self.l1_ratio = [.01, .1, .5, .9, .99]
        self.valid_loss_met = 'mean_square_loss'

        if self.reg_met == 'MCP':
            self.lambda_star \
                = Cross_validate(self.XD, self.Y_inv, self.Phi, self.valid_loss_met, lambdas=self.lambdas, reg_met=self.reg_met)
            self.alpha_star, self.l1_ratio_star = None, None

        elif self.reg_met == 'lasso' or reg_met == 'ridge':
            self.alpha_star \
                = Cross_validate(self.XD, self.Y_inv, self.Phi, self.valid_loss_met, alphas=self.alphas, reg_met=self.reg_met)
            self.l1_ratio_star, self.lambda_star = None, None

        elif self.reg_met == 'elastic net':
            self.alpha_star, self.l1_ratio_star \
                = Cross_validate(self.XD, self.Y_inv, self.Phi, self.valid_loss_met,  alphas=self.alphas,
                                 l1_ratio=self.l1_ratio, reg_met=self.reg_met)
            self.lambda_star = None

    def B_spline(self, t):
        from scipy.interpolate import BSpline

        spline_deg = 3
        num_est_basis = len(t) - spline_deg - 1
        Phi = np.zeros((len(t), num_est_basis))
        for k in range(num_est_basis):
            coeff_ = np.eye(num_est_basis)[k, :]
            fun = BSpline(t, coeff_, spline_deg)
            Phi[:, k] = fun(t)
        return Phi

    def DR_estimate(self, XD_train, Y_inv_train, XD_test, Y_inv_test):
        est_XD_test_list = []
        for i in range(self.num_treatment):
            est_XD_test = XD_test.copy()
            est_XD_test[:, -1] = i
            est_XD_test_list.append(est_XD_test)

        if self.reg_met in ['MCP', 'lasso', 'ridge', 'elastic net']:
            self.est_Y_inv_list \
                = Regression(XD_train, XD_test, est_XD_test_list, Y_inv_train, Y_inv_test,
                             self.Phi, self.lambda_star, self.alpha_star, self.l1_ratio_star, self.reg_met)

        elif self.reg_met in ['NFR', 'MLP', 'rep_NFR']:
            if self.reg_met == 'NFR':
                models = NFR(dim_in=args.num_cov+1, dim_out=args.num_cov+1, t=self.quantiles, dim_hidden=[64, 64],
                             activation=args.act_fn, device=device).to(device)
            elif self.reg_met == 'MLP':
                models = MLP(dim_in=args.num_cov+1, dim_out=len(quantiles), device=device).to(device)
            elif self.reg_met == 'rep_NFR':
                models = rep_NFR(dim_in=args.num_cov, dim_out=32, t=self.quantiles, dim_hidden_rep=[64, 64],
                                 dim_hidden_head=[32, 32], activation=args.act_fn, device=device).to(device)

            dl_train = DataLoader(dataset=sim_dataset_reg(XD_train, Y_inv_train), shuffle=True, batch_size=args.batch_size)
            dl_test = DataLoader(dataset=sim_dataset_reg(XD_test, Y_inv_test), shuffle=False, batch_size=len(XD_test))

            dl_est_test = []
            for i in range(self.num_treatment):
                dl_est_test.append(DataLoader(dataset=sim_dataset_reg(est_XD_test_list[i], Y_inv_test), shuffle=False, batch_size=len(est_XD_test_list[i])))

            self.est_Y_inv_list = train_regression(args, models, dl_train, dl_test, dl_est_test, device=device, verbose=True)

        elif self.reg_met in ['Classical']:
            self.est_Y_inv_list = [np.zeros([len(self.XD_test), 9]), np.zeros([len(self.XD_test), 9]),
                                   np.zeros([len(self.XD_test), 9]), np.zeros([len(self.XD_test), 9]),
                                   np.zeros([len(self.XD_test), 9])]
            for q in range(9):
                print('Estimate for quantile', q)
                Y_train_q, Y_test_q = Y_inv_train[:, q], Y_inv_test[:, q]
                models = MLP(dim_in=args.num_cov+1, dim_out=1, device=device).to(device)
                dl_train = DataLoader(dataset=sim_dataset_reg(XD_train, Y_train_q), shuffle=True, batch_size=args.batch_size)
                dl_test = DataLoader(dataset=sim_dataset_reg(XD_test, Y_test_q), shuffle=False, batch_size=len(XD_test))
                dl_est_test = []
                for i in range(self.num_treatment):
                    dl_est_test.append(DataLoader(dataset=sim_dataset_reg(est_XD_test_list[i], Y_test_q), shuffle=False, batch_size=len(est_XD_test_list[i])))
                args.epochs = 50
                est_Y_q = train_regression(args, models, dl_train, dl_test, dl_est_test, device=device, verbose=True)
                for j, est_Y in enumerate(est_Y_q):
                    self.est_Y_inv_list[j][:, q] = est_Y.squeeze()

        return [np.mean(est_Y_inv, axis=0) for est_Y_inv in self.est_Y_inv_list]

    def IPW_estimate(self, XD_train, XD_test, Y_inv_test):
        X_train, D_train = XD_train[:, :-1], XD_train[:, -1].reshape(-1, 1)
        X_test, D_test = XD_test[:, :-1], XD_test[:, -1].reshape(-1, 1)

        if args.class_met == 'MLP':
            from sklearn.model_selection import train_test_split
            train_idx, val_idx = train_test_split(range(len(X_train)), train_size=0.8, random_state=42)
            D_train, D_val = D_train[train_idx], D_train[val_idx]
            X_train, X_val = X_train[train_idx], X_train[val_idx]
            dl_train = DataLoader(np.hstack((X_train, D_train.reshape(-1, 1))), shuffle=True, batch_size=args.batch_size)
            dl_val = DataLoader(np.hstack((X_val, D_val.reshape(-1, 1))), shuffle=False, batch_size=len(X_val))
            dl_test = DataLoader(X_test, shuffle=False, batch_size=len(X_test))
            models = MLP(dim_in=X_train.shape[1], dim_out=self.num_treatment, dim_hidden=[64, 64, 64, 64], device=device).to(device)
            self.est_prob = train_classification(args, models, dl_train, dl_val, dl_test, device=device, verbose=True)
        else:
            self.est_prob = propensity_score(X_train, D_train, X_test)

        return [np.mean(Y_inv_test * (D_test == i) * 1.0 / np.expand_dims(self.est_prob[:, i], axis=1), axis=0) for i in range(self.num_treatment)]

    def DDL_estimate(self, XD_test, Y_inv_test):
        X_test, D_test = XD_test[:, :-1], XD_test[:, -1].reshape(-1, 1)

        return [np.mean(self.est_Y_inv_list[i] + (D_test == i) * 1.0 / np.expand_dims(self.est_prob[:, i], 1) * (Y_inv_test - self.est_Y_inv_list[i]), axis=0) for i in range(self.num_treatment)]

    def estimate(self):
        kf = KFold(n_splits=self.K_fold)
        est_DR_list = []
        est_IPW_list = []
        est_DDL_list = []
        weight_list = []
        iter = 0

        for train_idx, test_idx in kf.split(self.XD):
            print(f'\ncross fitting: {iter}')
            # training DR

            self.XD_train, self.XD_test = self.XD[train_idx], self.XD[test_idx]
            self.Y_inv_train, self.Y_inv_test = self.Y_inv[train_idx], self.Y_inv[test_idx]

            est_DR = self.DR_estimate(self.XD_train, self.Y_inv_train, self.XD_test, self.Y_inv_test)
            # est_DR_list.append(self.compute_ATE(est_DR))
            est_DR_list.append(est_DR)

            est_IPW = self.IPW_estimate(self.XD_train, self.XD_test, self.Y_inv_test)
            # est_IPW_list.append(self.compute_ATE(est_IPW))
            est_IPW_list.append(est_IPW)

            est_DDL = self.DDL_estimate(self.XD_test, self.Y_inv_test)
            # est_DDL_list.append(self.compute_ATE(est_DDL))
            est_DDL_list.append(est_DDL)

            weight_list.append(len(test_idx)/self.N)
            iter += 1

        # return np.average(est_DR_list, axis=0, weights=weight_list)
        return np.average(np.stack(est_DR_list), axis=0, weights=weight_list), \
               np.average(np.stack(est_IPW_list), axis=0, weights=weight_list), \
               np.average(np.stack(est_DDL_list), axis=0, weights=weight_list)

    def compute_ATE(self, est_list):
        ATE = np.ones([self.num_treatment, self.num_treatment, self.args.obs_dim])
        for i in range(len(est_list)):
            for j in range(len(est_list)):
                if i != j:
                    ATE[i, j, :] = est_list[i] - est_list[j]
        return ATE


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    set_seed(0)
    quantiles = np.linspace(0.1, 0.9, args.obs_dim)

    DGP = DataGenerator(args)
    X, D, Y = DGP.generate()

    # DGP.plot(num_sample=5)
    Ground_truth = DGP.groud_truth(quantiles)

    # estimate the distribution of observations (y)
    y_inv_sample_list, y_inv_pdf_list = [], []
    y_inv_list = []

    # estimate the empirical CDF
    for i in range(args.sample_size):
        y_inv = est_ECDF(Y[i, :].squeeze(), quantiles, args)
        y_inv_list.append(y_inv)

    y_inv_lambda = np.stack(y_inv_list)

    for exp_id in range(args.start_exp, args.end_exp):
        if not os.path.exists(os.path.join(args.results_save_path, 'Simulation', 'exp_' + str(exp_id))):
            os.makedirs(os.path.join(args.results_save_path, 'Simulation', 'exp_' + str(exp_id)))

        # set random seed for each experiment

        print(f'---------Exp_id: {exp_id}----------')

        save_path = os.path.join(args.results_save_path, 'Simulation', 'exp_' + str(exp_id))
        df_ground = pd.DataFrame(Ground_truth, columns=quantiles)
        df_ground.to_csv(os.path.join(save_path, 'Ground_DR.csv'))

        for reg_met in ['MCP', 'lasso', 'ridge', 'elastic net', 'Classical', 'NFR']:
            print(f'----Training: {reg_met}----')
            estimator = Estimator(Y_inv=y_inv_lambda, D=D, X=X, quantiles=quantiles, K_fold=5, reg_met=reg_met)
            DR_res, IPW_res, DDL_res = estimator.estimate()
            df_DR = pd.DataFrame(DR_res, columns=quantiles)
            df_IPW = pd.DataFrame(IPW_res, columns=quantiles)
            df_DML = pd.DataFrame(DDL_res, columns=quantiles)
            df_DR.to_csv(os.path.join(save_path, reg_met+'_DR.csv'))
            df_IPW.to_csv(os.path.join(save_path, reg_met+'_IPW.csv'))
            df_DML.to_csv(os.path.join(save_path, reg_met+'_DML.csv'))

