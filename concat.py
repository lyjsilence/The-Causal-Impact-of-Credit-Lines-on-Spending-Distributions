import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance


def concat_sim():
    df_ground, df_DR, df_IPW, df_DML, df_DR_e, df_IPW_e, df_DML_e = {}, {}, {}, {}, {}, {}, {}

    df_DR_res_q = pd.DataFrame(columns=['Classical', 'MCP', 'lasso', 'ridge', 'elastic net', 'NFR'])
    df_IPW_res_q = pd.DataFrame(columns=['Classical', 'MCP', 'lasso', 'ridge', 'elastic net', 'NFR'])
    df_DML_res_q = pd.DataFrame(columns=['Classical', 'MCP', 'lasso', 'ridge', 'elastic net', 'NFR'])

    df_DR_res_q_std = pd.DataFrame(columns=['Classical', 'MCP', 'lasso', 'ridge', 'elastic net', 'NFR'])
    df_IPW_res_q_std = pd.DataFrame(columns=['Classical', 'MCP', 'lasso', 'ridge', 'elastic net', 'NFR'])
    df_DML_res_q_std = pd.DataFrame(columns=['Classical', 'MCP', 'lasso', 'ridge', 'elastic net', 'NFR'])

    for model in ['Classical', 'MCP', 'lasso', 'ridge', 'elastic net', 'NFR']:
        df_DR[model], df_IPW[model], df_DML[model] = [], [], []
        df_DR_e[model], df_IPW_e[model], df_DML_e[model] = [], [], []
        df_ground[model] = []

        for exp_id in range(50):
            save_path = f"results/Simulation/exp_{str(exp_id)}"
            ground = pd.read_csv(os.path.join(save_path, 'Ground_DR.csv')).iloc[:, 1:]

            model_DR = pd.read_csv(os.path.join(save_path, f'{model}_DR.csv')).iloc[:, 1:]
            model_IPW = pd.read_csv(os.path.join(save_path, f'{model}_IPW.csv')).iloc[:, 1:]
            model_DML = pd.read_csv(os.path.join(save_path, f'{model}_DML.csv')).iloc[:, 1:]

            df_DR[model].append(np.array(model_DR))
            df_IPW[model].append(np.array(model_IPW))
            df_DML[model].append(np.array(model_DML))

            df_DR_e[model].append(np.abs(model_DR - ground))
            df_IPW_e[model].append(np.abs(model_IPW - ground))
            df_DML_e[model].append(np.abs(model_DML - ground))

        df_DR_res_q[model] = np.mean(np.mean(np.array(df_DR_e[model]), axis=0), axis=0)
        df_IPW_res_q[model] = np.mean(np.mean(np.array(df_IPW_e[model]), axis=0), axis=0)
        df_DML_res_q[model] = np.mean(np.mean(np.array(df_DML_e[model]), axis=0), axis=0)
        df_DR_res_q_std[model] = np.mean(np.std(np.array(df_DR_e[model]), axis=0), axis=0)
        df_IPW_res_q_std[model] = np.mean(np.std(np.array(df_IPW_e[model]), axis=0), axis=0)
        df_DML_res_q_std[model] = np.mean(np.std(np.array(df_DML_e[model]), axis=0), axis=0)

    df_DR_res_q.to_csv('results/Simulation/'+'mean_DR.csv')
    df_IPW_res_q.to_csv('results/Simulation/'+'mean_IPW.csv')
    df_DML_res_q.to_csv('results/Simulation/'+'mean_DML.csv')
    df_DR_res_q.to_csv('results/Simulation/'+'std_DR.csv')
    df_IPW_res_q.to_csv('results/Simulation/'+'std_IPW.csv')
    df_DML_res_q.to_csv('results/Simulation/'+'std_DML.csv')

if __name__ == '__main__':
    concat_sim()