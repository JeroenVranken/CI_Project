import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from numpy import genfromtxt
import numpy as np
import torch.optim as optim
import sys


class JNetV1(torch.nn.Module):
    def __init__(self, D_in, h_layer_size, n_hidden_layers, D_out):
        super(JNetV1, self).__init__()
        self.D_in = D_in
        self.h_layer_size = h_layer_size
        self.n_hidden_layers = n_hidden_layers
        self.D_out = D_out
        self.batch_size = 1

        self.linear1 = torch.nn.Linear(D_in, h_layer_size)
        self.lstm = torch.nn.LSTM(h_layer_size, h_layer_size, n_hidden_layers)
        self.linear2 = torch.nn.Linear(h_layer_size, 10)
        self.linear3 = torch.nn.Linear(10, D_out)


    def init_hidden(self):
        hidden = (autograd.Variable(torch.zeros(self.n_hidden_layers, self.batch_size, self.h_layer_size)), autograd.Variable(torch.randn((self.n_hidden_layers, self.batch_size, self.h_layer_size))))

    def forward(self, inp, h_0):
        f1 = self.linear1(inp)
        lstm_out, h_0 = self.lstm(f1, h_0)
        f2 = self.linear2(lstm_out[-1])
        f3 = self.linear3(f2)
        return f3, h_0

class JNetV2(torch.nn.Module):
    def __init__(self):
        super(JNetV2, self).__init__()

        self.D_in = 25
        self.h_layer_size = 5
        self.n_hidden_layers = 3
        self.D_out = 3
        self.batch_size = 1

        self.linear1 = torch.nn.Linear(self.D_in, self.h_layer_size)
        self.lstm = torch.nn.LSTM(self.h_layer_size, self.h_layer_size, self.n_hidden_layers)
        self.linear2 = torch.nn.Linear(self.h_layer_size, self.D_out)


    def init_states(self):
        self.states = (autograd.Variable(torch.randn(self.n_hidden_layers, self.batch_size, self.h_layer_size)), autograd.Variable(torch.randn((self.n_hidden_layers, self.batch_size, self.h_layer_size))))

    def forward(self, inp, states):
        f1 = self.linear1(inp)
        lstm_out, self.states = self.lstm(f1, states)
        f2 = self.linear2(lstm_out[-1])
        return f2, self.states


# Initial linear net

class simpleNetV1(nn.Module):
    def __init__(self):
        super(simpleNetV1, self).__init__()
        self.D_in = 25
        self.h1_size = 100
        self.h2_size = 50
        self.h3_size = 20
        self.D_out = 3

        self.inp_h1 = nn.Linear(self.D_in, self.h1_size)
        self.h1_h2 = nn.Linear(self.h1_size, self.h2_size)
        self.h2_h3 = nn.Linear(self.h2_size, self.h3_size)
        self.out = nn.Linear(self.h3_size, self.D_out)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()
    
    def forward(self, x):
        h1 = self.inp_h1(x)
        h1_act = self.relu(h1)
        h2 = self.h1_h2(h1_act)
        h2_act = self.relu(h2)
        h3 = self.h2_h3(h2)
        h3_act = self.relu(h3)
        output = self.out(h3_act)
        out_relu= self.sigm(output[0:2])
        out_steer = self.tanh(output[2])
        out_activated = torch.cat((out_relu, out_steer), 0)

        return out_activated

# Same as simpleNetV1 but increased hidden layer size best epoch = 1

class simpleNetV2(nn.Module):
    def __init__(self):
        super(simpleNetV2, self).__init__()
        self.D_in = 25
        self.h1_size = 600
        self.h2_size = 300
        self.h3_size = 100
        self.D_out = 3

        self.inp_h1 = nn.Linear(self.D_in, self.h1_size)
        self.h1_h2 = nn.Linear(self.h1_size, self.h2_size)
        self.h2_h3 = nn.Linear(self.h2_size, self.h3_size)
        self.out = nn.Linear(self.h3_size, self.D_out)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()
    
    def forward(self, x):
        h1 = self.inp_h1(x)
        h1_act = self.relu(h1)
        h2 = self.h1_h2(h1_act)
        h2_act = self.relu(h2)
        h3 = self.h2_h3(h2)
        h3_act = self.relu(h3)
        output = self.out(h3_act)
        out_relu= self.sigm(output[0:2])
        out_steer = self.tanh(output[2])
        out_activated = torch.cat((out_relu, out_steer), 0)

        return out_activated

# Made simpleNetV2 deeper with extra hidden layer, best epoch = 3
# seems like the deeper net makes the steering less smooth (non-smoothness caused by training data)

class simpleNetV3(nn.Module):
    def __init__(self):
        super(simpleNetV3, self).__init__()
        self.D_in = 25
        self.h1_size = 600
        self.h2_size = 400
        self.h3_size = 200
        self.h4_size = 100
        self.D_out = 3

        self.inp_h1 = nn.Linear(self.D_in, self.h1_size)
        self.h1_h2 = nn.Linear(self.h1_size, self.h2_size)
        self.h2_h3 = nn.Linear(self.h2_size, self.h3_size)
        self.h3_h4 = nn.Linear(self.h3_size, self.h4_size)
        self.out = nn.Linear(self.h4_size, self.D_out)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()
    
    def forward(self, x):
        h1 = self.inp_h1(x)
        h1_act = self.relu(h1)
        h2 = self.h1_h2(h1_act)
        h2_act = self.relu(h2)
        h3 = self.h2_h3(h2)
        h3_act = self.relu(h3)
        h4 = self.h3_h4(h3)
        h4_act = self.relu(h4)
        output = self.out(h4_act)
        out_relu= self.sigm(output[0:2])
        out_steer = self.tanh(output[2])
        out_activated = torch.cat((out_relu, out_steer), 0)

        return out_activated


# Made simpleNetV1 deeper, and removed convolution
# best epoch = 6
class simpleNetV4(nn.Module):
    def __init__(self):
        super(simpleNetV4, self).__init__()
        self.D_in = 25
        self.h1_size = 100
        self.h2_size = 100
        self.h3_size = 100
        self.h4_size = 100
        self.h5_size = 100
        self.h6_size = 100
        self.D_out = 3

        self.inp_h1 = nn.Linear(self.D_in, self.h1_size)
        self.h1_h2 = nn.Linear(self.h1_size, self.h2_size)
        self.h2_h3 = nn.Linear(self.h2_size, self.h3_size)
        self.h3_h4 = nn.Linear(self.h3_size, self.h4_size)
        self.h4_h5 = nn.Linear(self.h4_size, self.h5_size)
        self.h5_h6 = nn.Linear(self.h5_size, self.h6_size)

        self.out = nn.Linear(self.h6_size, self.D_out)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()
    
    def forward(self, x):
        h1 = self.inp_h1(x)
        h1_act = self.relu(h1)
        h2 = self.h1_h2(h1_act)
        h2_act = self.relu(h2)
        h3 = self.h2_h3(h2)
        h3_act = self.relu(h3)
        h4 = self.h3_h4(h3)
        h4_act = self.relu(h4)
        h5 = self.h4_h5(h4)
        h5_act = self.relu(h5)
        h6 = self.h5_h6(h5)
        h6_act = self.relu(h6)
        output = self.out(h6_act)
        out_relu= self.sigm(output[0:2])
        out_steer = self.tanh(output[2])
        out_activated = torch.cat((out_relu, out_steer), 0)

        return out_activated

class simpleNetV5(nn.Module):
    def __init__(self):
        super(simpleNetV5, self).__init__()
        self.D_in = 25
        self.h1_size = 1000
        self.h2_size = 500
        self.h3_size = 100
        self.D_out = 3

        self.inp_h1 = nn.Linear(self.D_in, self.h1_size)
        self.h1_h2 = nn.Linear(self.h1_size, self.h2_size)
        self.h2_h3 = nn.Linear(self.h2_size, self.h3_size)
        self.out = nn.Linear(self.h3_size, self.D_out)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()
    
    def forward(self, x):
        h1 = self.inp_h1(x)
        h1_act = self.relu(h1)
        h2 = self.h1_h2(h1_act)
        h2_act = self.relu(h2)
        h3 = self.h2_h3(h2)
        h3_act = self.relu(h3)
        output = self.out(h3_act)
        out_relu= self.sigm(output[0:2])
        out_steer = self.tanh(output[2])
        out_activated = torch.cat((out_relu, out_steer), 0)

        return out_activated


# Todo: try different optimizers and loss functions
# So far all have been trained with nn.MSEloss and optimizer = SGD, learning rate = 0.01
