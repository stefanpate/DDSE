import numpy as np
import torch
import torch.nn as nn


class Net(nn.Module):

    def __init__(self, n_inputs, n_outputs, hidden_widths, activation_fcn):
        '''
        Args:
            - n_inputs
            - n_outputs
            - hidden_widths: Iterable with number hidden units
            - activation_fcn: Function from torch.nn.functional (e.g., tanh)
            in each layer
        '''      
        super(Net, self).__init__()
        
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.hidden_widths = hidden_widths
        self.activation_fcn = activation_fcn
        self.n_layers = len(self.hidden_widths)
        self.layers = nn.ModuleList()
        
        # Create layers in a loop
        for i in range(len(self.hidden_widths) + 1):
            if i == 0:
                self.layers.append(nn.Linear(n_inputs, self.hidden_widths[0])) # Args to layers are in_features, out_features
            elif i == len(hidden_widths):
                self.layers.append(nn.Linear(self.hidden_widths[-1], n_outputs))
            else:
                self.layers.append(nn.Linear(self.hidden_widths[i-1], hidden_widths[i]))


    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation_fcn(layer(x))
        out = self.layers[-1](x)
        return out
