import os
import sys

import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.tensorboard import SummaryWriter
import logging

from src.datasetLoader import DatasetLoader
from src.finTrainer import FINTrainer
from src.helper.pickleLoader import load_object
from src.modelLoader import ModelLoader
from src.network.dynamicNN import DynamicNeuralNetwork
from src.network.finNN import FeatureImitationNetwork
from src.network.utiles import Data, EarlyStopping

# Writer will output to ./runs/ directory by default
writer = SummaryWriter()




print(sys.argv[0])
print('-----------------')
print(sys.argv[1])

run = int(sys.argv[1])
num_nodes = int(sys.argv[2])
pid = int(sys.argv[1])
# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

# Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')
else:
    print('Using CPU :(')

fb = ['delta', 'theta', 'alpha', 'beta', 'whole_spec']
md = ['delta_quantile', 'theta_quantile', 'alpha_quantile', 'beta_quantile', 'whole_spec_quantile']

dl = DatasetLoader(40000, ['EC', 'EO'], fb)
dl.get_epoch_ids()
dl.get_x_data()
dl.get_y_data()

x_data = dl.x_data
y_data = dl.y_data
group = dl.group
sample_ids = dl.sample_ids

gkf = GroupKFold(n_splits=5)
train_index, test_index = next(iter(gkf.split(x_data, y_data, group)))

x_train, x_test = [x_data[i] for i in train_index], [x_data[i] for i in test_index]
y_train, y_test = [y_data[i] for i in train_index], [y_data[i] for i in test_index]

train_data = Data(x_train, y_train, device)
test_data = Data(x_test, y_test, device)

# Create a file to save our model scores with their parameters.
dir_path = r"/home/modelrep/sadiya/tobias_ettling/data/results/"
res_path = dir_path + f'hpt_{pid}.csv'

# Check to avoid overwrite
if not (os.path.isfile(res_path)):
    # Create dataframe with parameters/score columns
    df = pd.DataFrame(
        columns=['MAE score P1', 'MAE score P2', 'loss', 'learning_rate', 'batch_size', 'hidden_sizes', 'epochs',
                 'activation', 'optimizer', 'early stopping'])
    df.to_csv(res_path)

hp_space = load_object('/home/modelrep/sadiya/tobias_ettling/HPC_FIN/hptSpace')

for i in range(0, 2):
    if i % num_nodes == pid:
        hyper_param = hp_space[i]
        hyper_param['device'] = device
        hyper_param['input_size'] = len(fb * 100)

        ml = ModelLoader(md)  # , 'whole_spec_quantile'
        ml.build_fin_models()
        stage_1 = ml.fin_models
        stage_2 = DynamicNeuralNetwork(
            input_size=hyper_param['input_size'],
            output_size=hyper_param['output_size'],
            hidden_sizes=hyper_param['hidden_sizes'],
            activ=hyper_param['activation'],
            dropout_p=hyper_param['dropout_p']
        )

        model = FeatureImitationNetwork(stage_1, stage_2, dl.data_order, device, freeze_fins=True)
        model.to(device)

        early_stopping = EarlyStopping(tolerance=3, min_delta=0.2)

        fit_param = dict(
            learning_rate=hyper_param['learning_rate'],
            batch_size=hyper_param['batch_size'],
            epochs=hyper_param['epochs'],
            weight_decay=hyper_param['weight_decay'],
            momentum=hyper_param['momentum'],
            activation=hyper_param['activation'],
            sched_ss=hyper_param['sched_ss'],
            sched_g=hyper_param['sched_g']
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

        # Read in the results
        r = pd.read_csv(res_path, index_col=0)

        # Add new results
        r.loc[len(r.index)] = [eval_score[0], eval_score[1], min(fin_trainer.loss_log['train_loss']),
                               fit_param['learning_rate'], fit_param['batch_size'], str(fit_param['hidden_sizes']),
                               fit_param['epochs'],
                               fit_param['activation'], fit_param['optimizer'],
                               fin_trainer.early_stopping.early_stop]
        print(r.loc[len(r.index) - 1])
        r.to_csv(res_path)
