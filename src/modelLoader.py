import os

import torch
from torch import nn

from src.helper.pickleLoader import load_object
from src.network.dynamicNN import DynamicNeuralNetwork


class ModelLoader:
    MODEL_PATH = r'/scratch/modelrep/sadiya/students/tobias/data/models/'

    def __init__(self, sel_models):
        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.selected_models = sel_models

        self.fin_models = nn.ModuleDict()
        self.model_dict = dict()
        self.get_aval_models()

        print('Following models where found:')
        print('')
        print(self.model_dict.keys())
        print('')

    def build_fin_models(self):
        for key in self.model_dict.keys():
            self.load_pretrained_model(self.model_dict[key])

    def load_pretrained_model(self, info):
        model_name = info['frequency_band'] + '_' + info['feature']
        model_param = load_object(info['model_param'])
        model = DynamicNeuralNetwork(
            model_param['input_size'],
            model_param['output_size'],
            model_param['hidden_sizes'],
            model_param['activation'],
            model_param['dropout_p']
        )
        model.load_state_dict(torch.load(info['model_path']))
        model.to(self.__device)
        self.fin_models[model_name] = model
        return model

    def get_aval_models(self):
        fb_paths = [x for x in next(os.walk(self.MODEL_PATH))[1]]
        for fb in fb_paths:
            feat_paths = [x for x in next(os.walk(self.MODEL_PATH + fb + '/'))[1]]
            for m in feat_paths:
                if self.check_model_files(fb, m):
                    info_dict = dict(
                        feature=m,
                        frequency_band=fb,
                        model_path=self.MODEL_PATH + fb + '/' + m + '/model.pth',
                        model_param=self.MODEL_PATH + fb + '/' + m + '/model_param',
                        scaler=self.MODEL_PATH + fb + '/' + m + '/scaler'
                    )
                    if fb + '_' + m in self.selected_models:
                        self.model_dict[fb + '_' + m] = info_dict
                else:
                    print(f'model files for fin: {m} - {fb}')
        return self.model_dict

    def check_model_files(self, fq_band, method):
        return os.path.isfile(self.MODEL_PATH + fq_band + '/' + method + '/model.pth') and \
            os.path.isfile(self.MODEL_PATH + fq_band + '/' + method + '/model_param.pickle') and \
            os.path.isfile(self.MODEL_PATH + fq_band + '/' + method + '/scaler.pickle')
