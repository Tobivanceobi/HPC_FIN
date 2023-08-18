import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold

from src.datasetLoader import DatasetLoader
from src.finTrainer import FINTrainer
from src.helper.pickleLoader import load_object
from src.modelLoader import ModelLoader
from src.network.dynamicNN import DynamicNeuralNetwork
from src.network.finNN import FeatureImitationNetwork
from src.network.utiles import Data, EarlyStopping


def equalize_classes(targets, threshold=3):
    l = np.unique(np.array(targets), return_counts=True)
    tar_map = []
    for i in range(len(l[0])):
        if l[1][i] < threshold:
            c_up = i
            c_down = i
            con = True
            con2 = False
            while con:
                if c_up + 1 < len(l[0]):
                    c_up += 1
                    if l[1][c_up] >= threshold:
                        tar_map.append([l[0][i], l[0][c_up]])
                        con = False

                if c_down - 1 >= 0 and con:
                    c_down -= 1
                    if l[1][c_down] >= threshold:
                        tar_map.append([l[0][i], l[0][c_down]])
                        con = False

    tar_map_t = np.array(tar_map).transpose()
    y_new = []
    for age in targets:
        if age in tar_map_t[0]:
            id_age = list(tar_map_t[0]).index(age)
            y_new.append(tar_map_t[1][id_age])
        else:
            y_new.append(age)
    return y_new


def check_if_param_used(param, hpt_path):
    hpt_df = pd.read_csv(hpt_path, index_col=0)
    condition = (
            (hpt_df['learning_rate'] == param['learning_rate']) &
            (hpt_df['batch_size'] == param['batch_size']) &
            (hpt_df['hidden_sizes'] == str(param['hidden_sizes'])) &
            (hpt_df['activation'] == param['activation'])
    ).any()
    return condition.any()

fb = ['delta', 'theta', 'alpha', 'beta', 'whole_spec']
md = ['delta_quantile', 'theta_quantile', 'alpha_quantile', 'beta_quantile', 'whole_spec_quantile']

dl = DatasetLoader(40000, ['EC', 'EO'], fb)
dl.get_epoch_ids()
dl.get_x_data()
dl.get_y_data()

x_data = dl.x_data
y_data = dl.y_data
group = dl.group
print(y_data)
y_stf = [int(age*10) for age in y_data]
equalize_classes(y_data)
skf_vals = []
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=126)
train_index, test_index = next(iter(skf.split(x_data, y_stf, group)))
x_train, x_test = [x_data[i] for i in train_index], [x_data[i] for i in test_index]
y_train, y_test = [y_data[i] for i in train_index], [y_data[i] for i in test_index]
