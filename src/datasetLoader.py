import os

import numpy as np
import pandas as pd
from typing import List

from src.helper.pickleLoader import save_object, load_object


class DatasetLoader:
    TARGET_PATH = r'/home/modelrep/sadiya/tobias_ettling/data/train_subjects.csv'
    DATA_PATH = r'/home/modelrep/sadiya/tobias_ettling/data/training/'
    CACHE_PATH = r'/home/modelrep/sadiya/tobias_ettling/data/cache/'
    FREQ_BANDS = {
        'delta': [0.5, 4],
        'theta': [4, 8],
        'alpha': [8, 14],
        'beta': [14, 30],
        'whole_spec': [0.5, 30]
    }

    def __init__(
            self,
            num_samples: int,
            state: List[str],
            freq_bands: List[str],
            shuffle: bool = True,
            verbose: bool = True):
        self.num_samples = num_samples
        self.state = state
        self.__shuffle_samples = shuffle
        self.__verbose = verbose
        self.__freq_bands = freq_bands
        self.__traget_df = pd.read_csv(self.TARGET_PATH)

        self.x_data = np.array([])
        self.y_data = np.array([])
        self.sample_ids = np.array([])
        self.group = np.array([])

        self.data_order = []
        self.rejected_ids = np.array([])

        self.cache_data_fname = []

        for fn in os.listdir(self.DATA_PATH):
            for fb in freq_bands:
                if fb in fn:
                    self.cache_data_fname.append(fn.split('.pick')[0])

    def get_y_data(self):
        y_data_temp = []
        group_temp = []
        for samp_id in self.sample_ids:
            sample_group = int(samp_id.split('_')[1])
            sample_y = self.load_target_age(sample_group)
            y_data_temp.append(sample_y)
            group_temp.append(sample_group)

        self.y_data = np.array(y_data_temp)
        self.group = np.array(group_temp)

    def get_x_data(self):
        x_data_temp = []
        for d in range(len(self.cache_data_fname)):
            obj = load_object(self.DATA_PATH + self.cache_data_fname[d])
            freq_band = self.cache_data_fname[d].split('_1')[0]
            self.data_order.append(freq_band)
            print(freq_band)
            fb_x_data = []
            for i in range(len(self.sample_ids)):
                if self.sample_ids[i] in obj['sample_ids']:
                    samp_idx = np.where(obj['sample_ids'] == self.sample_ids[i])[0][0]
                    fb_x_data.append(obj['x_data'][samp_idx])
            x_data_temp.append(np.array(fb_x_data))
        x_data = []
        for n in range(len(self.sample_ids)):
            x_samp = []
            for data in x_data_temp:
                x_samp.append(data[n])
            x_data.append(np.array(x_samp))
        self.x_data = x_data

    def get_epoch_ids(self):
        rej_ids = []
        accepted_ids = []
        for d in self.cache_data_fname:
            obj = load_object(self.DATA_PATH + d)
            rej_ids.append(obj['rejected_ids'])
            accepted_ids.append(obj['sample_ids'])
        self.rejected_ids = np.unique(np.concatenate(rej_ids))
        accepted_ids = np.unique(np.concatenate(accepted_ids))
        accepted_ids = np.setdiff1d(accepted_ids, self.rejected_ids)
        accepted_ids = np.array([samp_id for samp_id in accepted_ids if int(samp_id.split('_')[1]) <= 1200])
        if self.num_samples < len(accepted_ids):
            self.num_samples = len(accepted_ids)
            idx_sample = np.random.choice(len(accepted_ids), self.num_samples, replace=False)
            self.sample_ids = accepted_ids[idx_sample]
        else:
            self.sample_ids = accepted_ids
        return self.rejected_ids, self.sample_ids

    def load_target_age(self, subj_id) -> float:
        """
        Load the age of the subject from the .csv file in the main data set
        :return: age of the subject as float.
        """
        targets = np.array(self.__traget_df[['id', 'age']])
        return targets[subj_id - 1][1]

    def get_cache_code(self):
        cache_code = str(self.num_samples) + '_'
        for s in self.state:
            cache_code += s + '_'
        for fb in self.__freq_bands:
            cache_code += fb + '_'
        return cache_code + 'set'
