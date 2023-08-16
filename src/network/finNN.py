import torch
import torch.nn as nn


class FeatureImitationNetwork(nn.Module):
    def __init__(self, stage_1, stage_2, data_order, device='cpu', freeze_fins: bool = True):
        super(FeatureImitationNetwork, self).__init__()

        self.data_order = data_order

        self.freeze_fins = freeze_fins
        self.__device = device

        self.__stage_1 = stage_1
        self.stage_1_remove_output_layer()

        self.__stage_2 = stage_2

    def forward(self, x_in):
        x = torch.tensor([]).to(self.__device)
        for key in self.__stage_1.keys():
            fb = key.split('_')[0]
            if 'whole_spec' in key:
                fb = 'whole_spec'
            data_idx = self.data_order.index(fb)
            x_fin = self.__stage_1[key](x_in[data_idx])
            x = torch.cat((x, x_fin), dim=1)
        x = self.__stage_2(x)
        return x

    def unfreeze_fins(self):
        for key in self.__stage_1:
            model = self.__stage_1[key]
            # unfreeze the parameters of the pretrained model
            for param in model.parameters():
                param.requires_grad = True
            self.__stage_1[key] = model

    def stage_1_remove_output_layer(self):
        for key in self.__stage_1:
            model = self.__stage_1[key]
            model.model_arch = model.model_arch[:-1]
            if self.freeze_fins:
                # Freeze the parameters of the pretrained model
                for param in model.parameters():
                    param.requires_grad = False

            self.__stage_1[key] = model
