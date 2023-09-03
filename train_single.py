import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, StratifiedKFold

from src.datasetLoader import DatasetLoader
from src.finTrainer import FINTrainer
from src.helper.pickleLoader import load_object
from src.modelLoader import ModelLoader
from src.network.dynamicNN import DynamicNeuralNetwork
from src.network.finNN import FeatureImitationNetwork
from src.network.utiles import Data, EarlyStopping


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

fb = ['delta', 'theta', 'alpha', 'beta', 'whole_spec']
md = ['delta_quantile', 'theta_quantile', 'alpha_quantile', 'beta_quantile', 'whole_spec_quantile']

dl = DatasetLoader(40000, ['EC'], fb)
dl.get_epoch_ids()
dl.get_x_data()
dl.get_y_data()

x_data = dl.x_data
y_data = dl.y_data

group = dl.group
sample_ids = dl.sample_ids

y_stf = [int(age) for age in y_data]
skf_vals = []
skf = StratifiedKFold(n_splits=4)
train_index, test_index = next(iter(skf.split(x_data, y_stf, group)))

x_train, x_test = [x_data[i] for i in train_index], [x_data[i] for i in test_index]
y_train, y_test = [y_data[i] for i in train_index], [y_data[i] for i in test_index]

train_data = Data(x_train, y_train, device)
test_data = Data(x_test, y_test, device)


ml = ModelLoader(md)  # , 'whole_spec_quantile'
ml.build_fin_models()
stage_1 = ml.fin_models
stage_2 = DynamicNeuralNetwork(
    input_size=len(fb * 100),
    output_size=1,
    hidden_sizes=[300, 100],
    activ='sigmoid',
    dropout_p=0.4
)

model = FeatureImitationNetwork(stage_1, stage_2, dl.data_order, device, freeze_fins=True)
model.to(device)

early_stopping = EarlyStopping(tolerance=5, min_delta=1)

fit_param = dict(
    optimizer='sgd',
    learning_rate=0.0001,
    batch_size=64,
    epochs=150,
    weight_decay=0.0001,
    momentum=0.1,
    activation='sigmoid',
    sched_ss=20,
    sched_g=0.9
)
eval_score = []
for phase in [1, 2]:
    if phase == 2:
        model.unfreeze_fins()

    fin_trainer = FINTrainer(
        model,
        fit_param,
        train_data,
        test_data,
        early_stop=early_stopping,
    )

    score = fin_trainer.fit_model()

    print(f'Phase {phase} - Evaluation MAE: ', score)
    eval_score.append(score)
print(eval_score)
print(min(fin_trainer.loss_log['train_loss']))

