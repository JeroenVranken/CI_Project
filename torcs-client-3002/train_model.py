from sklearn import preprocessing

import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from numpy import genfromtxt
import numpy as np
import torch.optim as optim



# training_names = ['aalborg.csv', 'alpine-1.csv', 'f-speedway.csv']
training_names = ['aalborg.csv']
sequence_size = 10
batch_size = 1
row_size = 25
hidden_size = 25
n_hidden_layers = 3

D_in = 25
D_out = 3

track_data = np.empty((1,row_size))



for filename in training_names:
    # Read data from file, add extra batch dimension, convert to Variable containing FloatTensors, and remove first line containting titles
    train_data_with_titles = Variable(torch.from_numpy(np.expand_dims(genfromtxt(filename, delimiter=','), axis=1)).type(torch.FloatTensor))
    train_data = train_data_with_titles[1:len(train_data_with_titles)]


def get_input_target_pair(data, time, seq_size):
    input = data[time:time+seq_size]
    target = data[time+seq_size][0][0:3]
    return input, target




class JNetV1(torch.nn.Module):
    def __init__(self, D_in, h_layer_size, n_hidden_layers, D_out):
        super(JNetV1, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, h_layer_size)
        self.lstm = torch.nn.LSTM(h_layer_size, h_layer_size, n_hidden_layers)
        self.linear2 = torch.nn.Linear(h_layer_size, 10)
        self.linear3 = torch.nn.Linear(10, D_out)


    def init_hidden(self):
        hidden = (autograd.Variable(torch.randn(n_hidden_layers, batch_size, hidden_size)), autograd.Variable(torch.randn((n_hidden_layers, batch_size, hidden_size))))

    def forward(self, inp, h_0):
        f1 = self.linear1(inp)
        lstm_out, h_o = self.lstm(f1, h_0)
        f2 = self.linear2(lstm_out[-1])
        f3 = self.linear3(f2)
        return f3



def train(model):

    for epoch in range(10):
        for t in range((len(train_data) -10)):

            model.zero_grad()
            model.hidden = model.init_hidden()
            inp, target = get_input_target_pair(train_data, t, sequence_size)
            
            out = model(inp, model.hidden)

            loss = loss_function(out[0], target)
            loss.backward()
            optimizer.step()

            if t % 100 == 0:
                print("Timestep: " + str(t) + " Loss: " + str(loss))
                print(out[0],target)

        print("Epoch: " + str(epoch) + " Loss: " + str(loss))

        filename = 'JNetV1' + str(epoch) + '.pkl'
        torch.save(model.state_dict(), filename)

model = JNetV1(D_in, hidden_size, n_hidden_layers, D_out)
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

train(model)
