"""
Created on : ------

@author: Ariel_Kantorovich
"""

import torch
import torch.nn as nn
import torch.nn.init as init

class Wireless_NN(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        """
        :param input_size: In general (2 * T_exploration) because [(Pn, In)_t=0, ...., (Pn, In)_t=T_explor]
        :param output_size: (2, ) because beta_n and alpha_n
        """
        super(Wireless_NN, self).__init__()
        self.fc0 = nn.Linear(input_size, 256)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_size)

        # Initialize weights
        self.init_weights()

    def forward(self, x):
        x = torch.relu(self.fc0(x))
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

