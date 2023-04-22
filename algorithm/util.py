import torch.nn as nn
import torch
import numpy as np

def init_layers(layers, std=np.sqrt(2), bias_const=0.0):
    for layer in layers:
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight.data, mode='fan_out', nonlinearity='relu')
        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.constant_(layer.weight.data, 1)
            nn.init.constant_(layer.bias.data, 0)
        elif isinstance(layer, nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, std)
            torch.nn.init.constant_(layer.bias, bias_const)

