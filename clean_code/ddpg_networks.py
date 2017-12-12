import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, D_in, D_out):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(D_in, 500)
        self.linear2 = nn.Linear(500, 800)
        self.linear3 = nn.Linear(800, 448)
        self.head = nn.Linear(448, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return self.head(x)
