import os
import sys

import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from src.datasetLoader import DatasetLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fb = ['delta', 'theta', 'alpha', 'beta', 'whole_spec']
md = ['delta_quantile', 'theta_quantile', 'alpha_quantile', 'beta_quantile', 'whole_spec_quantile']

dl = DatasetLoader(40000, ['EC', 'EO'], fb)
dl.check_cache()

x_data = dl.x_data
y_data = dl.y_data
group = dl.group
sample_ids = dl.sample_ids


dir_path = r"/content/drive/MyDrive/FIN/brainage"
# Create a file to save our model scores with their parameters.
res_path = dir_path + '/hpt.csv'

# Check to avoid overwrite
if not (os.path.isfile(res_path)):
    # Create dataframe with parameters/score columns
    df = pd.DataFrame(
        columns=['MAE score', 'loss', 'learning_rate', 'batch_size', 'hidden_sizes', 'epochs',
                 'activation', 'optimizer', 'early stopping'])
    df.to_csv(res_path)

    # TODO: Parameter add device and input size
