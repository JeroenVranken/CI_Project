#!/usr/bin/env python
from __future__ import print_function
from itertools import count
from sklearn import preprocessing

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
from SimpleRNN import *

training_names = ['aalborg.csv', 'alpine-1.csv', 'f-speedway.csv']
sequence_size = 1
D_in = sequence_size * 25
D_out = 3

sequence_size = 1
hidden_size = 15
my_data = np.empty((1,D_in))

for filename in training_names:
    # Get data, split off first row, and normalize between [0,1]
    my_data = np.append(my_data, genfromtxt(filename, delimiter=','), axis=0)
print(my_data.shape)



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


def get_batch(sequence_size=1):
    """Builds a batch i.e. (x, f(x)) pair."""
    x = np.empty([sequence_size, len(my_data.T)])
    y = np.empty([1, 3])
    random_int = np.random.randint(0, len(my_data) - (sequence_size + 1))
    x = my_data[random_int:random_int + sequence_size].flatten()
    y = my_data[random_int + sequence_size + 1][0:3].flatten()

    # for n in range(sequence_size):
    #     random_data = my_data[n:n+sequence_size]
    #     print(random_data)
    #     x[n] = my_data[3:]
    #     y[n] = random_data[:-(len(random_data.T) - 3)]

    # print(x, x.shape)
    # print(y, y.shape)
    x,y=torch.from_numpy(x),torch.from_numpy(y)

    x,y=x.type(torch.FloatTensor),y.type(torch.FloatTensor)
    #print(y)
    return Variable(x), Variable(y)

# get_batch(10)

class SimpleLSTM(torch.nn.Module):
    def __init__(self, D_in, sequence_size, hidden_size, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(SimpleLSTM, self).__init__()
        self.lstm1 = torch.nn.LSTM(input_size=D_in, hidden_size=hidden_size, num_layers=sequence_size, bias=True, dropout=False)
        self.softmax1 = torch.nn.Linear(hidden_size * sequence_size * D_in, D_out)#hidden_size, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        h_n, c_n = self.lstm1(x, None)
        


        y_pred = self.softmax1(h_n.view(1, -1))
        #print(y_pred)
        return y_pred

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, H2, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.rnn = torch.nn.LSTM(H, H2, 1)
        self.linear2 = torch.nn.Linear(H2, D_out)
        self.tanh = torch.nn.Hardtanh()

    def forward(self, x, hidden=None):
        input = self.linear1(x.view(1, -1)).unsqueeze(1)
        output, hidden = self.rnn(input, hidden)
        output = self.linear2(output.squeeze(1))
        output2 = self.tanh(output)
        return output2, hidden


n_epochs = 100
n_iters = 100

model = TwoLayerNet(D_in, hidden_size, 100, D_out)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

losses = np.zeros(n_epochs) # For plotting
hidden = None
for epoch in range(n_epochs):

    for iter in range(n_iters):
        x, y = get_batch(sequence_size)      


        # Use teacher forcing 50% of the time
        # force = np.random.random() < 0.5
        outputs, hidden = model(x, None)

        optimizer.zero_grad()

        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        losses[epoch] += loss.data[0]

    if epoch > 0:
        print(epoch, loss.data[0])




# class SimpleRNN(torch.nn.Module):
#     def __init__(self, hidden_size):
#         super(SimpleRNN, self).__init__()
#         self.hidden_size = hidden_size

#         self.inp = torch.nn.Linear(22, hidden_size)
#         self.rnn = torch.nn.LSTM(hidden_size, hidden_size, 2, dropout=0.05)
#         self.out = torch.nn.Linear(hidden_size, 3)

#     def step(self, input, hidden=None):
#         input = self.inp(input.view(1, -1)).unsqueeze(1)
#         output, hidden = self.rnn(input, hidden)
#         output = self.out(output.squeeze(1))
#         return output, hidden

#     def forward(self, inputs, hidden=None, force=True, steps=0):
#         if force or steps == 0: steps = len(inputs)
#         outputs = Variable(torch.zeros(steps, 3, 1))
#         for i in range(steps):
#             if force or i == 0:
#                 input = inputs[i]
#             else:
#                 input = output
#             output, hidden = self.step(input, hidden)
#             outputs[i] = output
#         return outputs, hidden

# n_epochs = 100
# n_iters = 50
# # hidden_size = 10

# model = SimpleRNN(hidden_size)
# criterion = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# losses = np.zeros(n_epochs) # For plotting

# for epoch in range(n_epochs):

#     for iter in range(n_iters):
#         inputs,targets = get_batch(10)
#         print(inputs, targets)
#         break
#         # Use teacher forcing 50% of the time
#         force = np.random.random() < 0.5
#         outputs, hidden = model(inputs, None, force)

#         optimizer.zero_grad()
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()

#         losses[epoch] += loss.data[0]

#     if epoch > 0:
#         print(epoch, loss.data[0])
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

torch.save(model.state_dict(), 'SimpleLSTMparameters.pkl')