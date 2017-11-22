import torch

from collections import defaultdict
import time
import random
import torch
from torch.autograd import Variable
import torch.autograd
from random import shuffle
import torch.nn as nn
import numpy as np
import time, math
import sys
import pandas as pd

class simpleGRU(nn.Module):
    def __init__(self, D_in, h_layer_size, n_hidden_layers, D_out):
        super(simpleGRU, self).__init__()
        self.D_in = D_in
        self.h_layer_size = h_layer_size
        self.D_out = D_out
        self.n_hidden_layers = n_hidden_layers

        self.encoder = nn.Linear(D_in, h_layer_size)
        # self.encoder = nn.Embedding(D_in, h_layer_size)
        self.gru = nn.GRU(h_layer_size, h_layer_size, n_hidden_layers)
        self.decoder = nn.Linear(h_layer_size, D_out)
    
    def forward(self, input, hidden):

        encoded = self.encoder(input.view(1, -1))
        out, hidden = self.gru(encoded.view(1, 1, -1), hidden)
        output = self.decoder(out.view(1, -1))

        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.n_hidden_layers, 1, self.h_layer_size))


def random_train(seq_length):
    start_ix = random.randint(0, data_size - seq_length)
    end_ix = start_ix + seq_length + 1
    sequence = data[start_ix:end_ix]
    
    inp = char_tensor(sequence[:-1])
    target = char_tensor(sequence[1:])
    return inp, target

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def train(inp, target):
    hidden = decoder.init_hidden()
    decoder.zero_grad()
    loss = 0

    for c in range(seq_length):
        output, hidden = decoder(inp[c], hidden)
        loss += criterion(output, target[c][0:3].view(1, -1))

    loss.backward()
    decoder_optimizer.step()

    return loss.data[0] / seq_length

def read_data(filename):
    pd_data = pd.read_csv('all_tracks.csv')
    data = torch.from_numpy(pd_data.values).type(torch.FloatTensor)
    data_size = data.shape[0]
    D_in = data.shape[1]
    return data, data_size, D_in

def get_train_pair(data, start_ix, seq_length):
    end_ix = start_ix + seq_length + 1
    sequence = data[start_ix:end_ix]

    inp = Variable(sequence[:-1])
    target = Variable(sequence[1:])

    return inp, target



if __name__ == '__main__':
    np.random.seed(1)

    seq_length = 25 # number of steps to unroll the RNN for
    hidden_size = 25
    n_hidden_layers = 3
    D_out = 3
    lr = 0.005

    n_epochs = 100
    print_every = 1
    plot_every = 1
    save_every = 1

    # Read in file
    filename = 'all_tracks.csv'
    data, data_size, D_in = read_data(filename)
    print('Data length: %d' % data_size)

    # Setup network
    decoder = simpleGRU(D_in, hidden_size, n_hidden_layers, D_out)
    # decoder.load_state_dict(torch.load('simpleGRU_epoch_2000file_sherlock.txt.pkl'))

    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.L1Loss()

    start = time.time()
    all_losses = []
    loss_avg = 0

    print('Starting training: Total Epochs: %d' % (n_epochs))
    # Start training
    for epoch in range(n_epochs):        
        counter = 0
        # Create random indexes for training sequences
        total_samples = data_size // (seq_length + 1)
        sample_list = list(range(total_samples))
        shuffle(sample_list)

        for ix in sample_list:
            inp, target = get_train_pair(data, ix, seq_length)
            # print(inp, target)
            # sys.exit()

            loss = train(inp, target)
            loss_avg += loss
            counter += 1
            if counter % 500 == 0:
                print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / n_epochs * 100, loss))


        if epoch % print_every == 0:
            print('[Finished Epoch %s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / n_epochs * 100, loss))
            # print(evaluate('I ', predict_len=100, temperature=0.4), '\n')
            

        if epoch % plot_every == 0:
            all_losses.append(loss_avg / plot_every)
            loss_avg = 0
        
        if epoch % save_every == 0:
            torch.save(decoder.state_dict(), 'simpleGRU_epoch_' + str(epoch) + '_' + filename + '.pkl')
            print("Model saved, epoch: %d" % (epoch))

    f= open("all_losses.txt","w+")
    for i in range(len(all_losses)):
        f.write(str(all_losses[i]) + ',')


#------------------------------------ RIP --------------------------
# def create_dataset(filenames):
#     df = pd.DataFrame()
#     for filename in filenames:
#         data = pd.read_csv(filename)
#         df = df.append(data, ignore_index=False)
        
#     return df