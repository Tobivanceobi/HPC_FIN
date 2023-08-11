import numpy as np
import torch
from torch.utils.data import Dataset


class Data(Dataset):
    def __init__(self, x, y, device='cpu'):
        self.x = torch.tensor(np.float32(x), requires_grad=True).to(device)
        self.y = torch.tensor(np.float32(y), requires_grad=True).to(device)
        self.len = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter > self.tolerance:
                self.early_stop = True