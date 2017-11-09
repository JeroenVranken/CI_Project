#!/usr/bin/env python
from __future__ import print_function
from itertools import count

import torch
import torch.autograd
import torch.nn.functional as F
from torch.autograd import Variable
from numpy import genfromtxt
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
import math, random

my_data = genfromtxt('aalborg.csv', delimiter=',')
#print(my_data[2])


POLY_DEGREE = 4
W_target = torch.randn(POLY_DEGREE, len(my_data)) * 5
b_target = torch.randn(len(my_data) - 3) * 5


def make_features(x):
    """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, POLY_DEGREE+1)], 1)


def f(x):
    """Approximated function."""
    return x.mm(W_target) + b_target[0]


def poly_desc(W, b):
    """Creates a string description of a polynomial."""
    result = 'y = '
    for i, w in enumerate(W):
        result += '{:+.2f} x^{} '.format(w, len(W) - i)
    result += '{:+.2f}'.format(b[0])
    return result


def get_batch(batch_size=1):
    """Builds a batch i.e. (x, f(x)) pair."""
    x = np.empty([batch_size, len(my_data.T) - 3])
    y = np.empty([batch_size, 3])
    for n in range(batch_size):
    	random_int = np.random.randint(1, len(my_data))
    	random_data = my_data[random_int]
    	x[n] = random_data[3:]
    	y[n] = random_data[:-(len(random_data.T) - 3)]
  
    x,y=torch.from_numpy(x),torch.from_numpy(y)
    x,y=x.type(torch.FloatTensor),y.type(torch.FloatTensor)
    x=torch.unsqueeze(x,1)
    #print(y)
    return Variable(x), Variable(y)


class SimpleRNN(torch.nn.Module):
    def __init__(self, hidden_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size

        self.inp = torch.nn.Linear(22, hidden_size)
        self.rnn = torch.nn.LSTM(hidden_size, hidden_size, 2, dropout=0.05)
        self.out = torch.nn.Linear(hidden_size, 3)

    def step(self, input, hidden=None):
        input = self.inp(input.view(1, -1)).unsqueeze(1)
        output, hidden = self.rnn(input, hidden)
        output = self.out(output.squeeze(1))
        return output, hidden

    def forward(self, inputs, hidden=None, force=True, steps=0):
        if force or steps == 0: steps = len(inputs)
        outputs = Variable(torch.zeros(steps, 3, 1))
        for i in range(steps):
            if force or i == 0:
                input = inputs[i]
            else:
                input = output
            output, hidden = self.step(input, hidden)
            outputs[i] = output
        return outputs, hidden

n_epochs = 100
n_iters = 50
hidden_size = 10

model = SimpleRNN(hidden_size)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

losses = np.zeros(n_epochs) # For plotting

for epoch in range(n_epochs):

    for iter in range(n_iters):
        inputs,targets = get_batch()

        # Use teacher forcing 50% of the time
        force = np.random.random() < 0.5
        outputs, hidden = model(inputs, None, force)

        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        losses[epoch] += loss.data[0]

    if epoch > 0:
        print(epoch, loss.data[0])
    # #Use some plotting library
    # if epoch % 10 == 0:
    #     torch.show_plot('inputs', _inputs, True)
    #     show_plot('outputs', outputs.data.view(-1), True)
    #     show_plot('losses', losses[:epoch] / n_iters)

    #     #Generate a test
    #     outputs, hidden = model(inputs, False, 50)
    #     show_plot('generated', outputs.data.view(-1), True)

# Online training
"""hidden = None

while True:
    inputs = get_latest_sample()
    outputs, hidden = model(inputs, hidden)

    optimizer.zero_grad()
    loss = criterion(outputs, inputs)
    loss.backward()
    optimizer.step()"""

torch.save(model.state_dict(), 'modelparameters.pkl')