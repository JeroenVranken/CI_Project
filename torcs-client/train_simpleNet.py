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
from networks import simpleNetV2


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
    # print(hidden)
    # sys.exit()

    for c in range(seq_length):
        output, hidden = decoder(inp[c], hidden)
        print(output)
        # print(output, target[c][0:3].view(1, -1))
        sys.exit()
        loss += criterion(output, target[c][0:3].view(1, -1))
        # print(loss)


    loss.backward()
    decoder_optimizer.step()

    return loss.data[0] / seq_length

def read_data(filename):
    pd_data = pd.read_csv('all_tracks.csv')
    data = torch.from_numpy(pd_data.values).type(torch.FloatTensor)
    data_size = data.shape[0]
    D_in = data.shape[1]
    return data, data_size, D_in

def get_train_pair_sequential(data, start_ix, seq_length):
    end_ix = start_ix + seq_length + 1
    sequence = data[start_ix:end_ix]

    inp = Variable(sequence[:-1])
    target = Variable(sequence[1:])

    return inp, target

def get_train_pair(data, start_ix):
    inp = Variable(data[start_ix])
    target = Variable(data[start_ix+1][0:3])

    return inp, target


if __name__ == '__main__':
    np.random.seed(1)

    lr = 0.01

    n_epochs = 100
    print_every = 1
    plot_every = 1
    save_every = 1

    # Read in file
    filename = 'f-speedway.csv'
    data, data_size, D_in = read_data(filename)
    print('Data length: %d' % data_size)

    # Setup network
    model = simpleNetV2()
    # decoder.load_state_dict(torch.load('simpleGRU_epoch_2000file_sherlock.txt.pkl'))
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)


    start = time.time()
    all_losses = []
    loss_avg = 0

    print('Starting training: Total Epochs: %d' % (n_epochs))
    # Start training
    for epoch in range(n_epochs):        
        counter = 0
        
        # Create random indexes for training sequences
        total_samples = data_size -1
        sample_list = list(range(total_samples))
        shuffle(sample_list)

        for ix in sample_list:
            inp, target = get_train_pair(data, ix)
            output = model(inp)
            loss = criterion(output, target)
            # print(loss)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            loss_avg += loss
            counter += 1
            if counter % 1000 == 0:
                print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / n_epochs * 100, loss.data[0]))


        if epoch % print_every == 0:
            print('[FINISHED EPOCH %s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / n_epochs * 100, loss.data[0]))
            # print(evaluate('I ', predict_len=100, temperature=0.4), '\n')
            

        if epoch % plot_every == 0:
            all_losses.append(loss_avg / plot_every)
            loss_avg = 0
        
        if epoch % save_every == 0:
            torch.save(model.state_dict(), 'simpleNetV1_epoch_' + str(epoch) + '_' + filename + '.pkl')
            print("Model saved, epoch: %d" % (epoch))


#------------------------------------ RIP --------------------------
# def create_dataset(filenames):
#     df = pd.DataFrame()
#     for filename in filenames:
#         data = pd.read_csv(filename)
#         df = df.append(data, ignore_index=False)
        
#     return df