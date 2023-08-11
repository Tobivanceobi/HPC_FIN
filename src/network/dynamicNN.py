from collections import OrderedDict

import torch
import torch.nn as nn


class DynamicNeuralNetwork(nn.Module):
    """
    Creates a neural network model of linear layers
    """
    def __init__(self, input_size, output_size, hidden_sizes, activ, dropout_p=0.2):
        """
        Specify a linear model and dynamically specify the layers.

        :param input_size: The size of the input (first) layer.
        :param output_size: The size of the output (last) layer.
        :param hidden_sizes: A list of hidden layers as int for there size.
        """
        super(DynamicNeuralNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.activation = activ
        self.dropout_p = dropout_p

        self.model_arch = nn.ModuleList()

        self.create_input_layer()
        self.create_hidden_layers()
        self.create_output_layer()

    def forward(self, x):
        for layer in self.model_arch:
            x = layer(x)
        return x

    def get_activation_func(self):
        if self.activation == 'sigmoid':
            return nn.Sigmoid()
        elif self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'tanh':
            return nn.Tanh()
        else:
            return nn.ReLU

    def create_input_layer(self):
        input_layer = nn.Sequential(OrderedDict([
                ('in_linear', nn.Linear(self.input_size, self.hidden_sizes[0])),
                ('in_batch norm', nn.BatchNorm1d(self.hidden_sizes[0])),
                ('in_activation', self.get_activation_func()),
                ('in_dropout', nn.Dropout(p=self.dropout_p))
            ])
        )
        self.model_arch.append(input_layer)

    def create_hidden_layers(self):
        # Add hidden layers with dropout
        for i in range(len(self.hidden_sizes) - 1):
            layer = nn.Sequential(OrderedDict([
                (f'{i}_linear', nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i + 1])),
                (f'{i}_batch norm', nn.BatchNorm1d(self.hidden_sizes[i + 1])),
                (f'{i}_activation', self.get_activation_func()),
                (f'{i}_dropout', nn.Dropout(p=self.dropout_p))
            ])
            )
            self.model_arch.append(layer)

    def create_output_layer(self):
        self.model_arch.append(nn.Sequential(OrderedDict([
                ('out_linear', nn.Linear(self.hidden_sizes[-1], self.output_size)),
            ])
        ))


