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
print(f"Job {pid} started...")
print(f'Using device: {device}')

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    cuda = torch.device('cuda')  # Default CUDA device
    cuda0 = torch.device('cuda:0')
    cuda2 = torch.device('cuda:2')

    # Print GPU IDs
    for gpu_id in range(num_gpus):
        print(f"GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
else:
    print("No GPUs available")