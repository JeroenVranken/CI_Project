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
from networks import simpleNetV2, simpleNetV3, simpleNetV4, simpleNetV5


# Returns time since start of training
def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# Loads the training data
def read_data(filename):
    pd_data = pd.read_csv('all_tracks.csv')
    data = torch.from_numpy(pd_data.values).type(torch.FloatTensor)
    data_size = data.shape[0]
    D_in = data.shape[1]
    return data, data_size, D_in

# Gets an input/target pair for training at a specified index
def get_train_pair(data, start_ix):
    inp = Variable(data[start_ix])
    target = Variable(data[start_ix+1][0:3])

    return inp, target


if __name__ == '__main__':
    np.random.seed(1)

    # Learning rate
    lr = 0.01

    # Total number of training epochs
    n_epochs = 10

    print_every = 1
    plot_every = 1
    save_every = 1

    # Read in file
    filename = 'all_tracks.csv'
    data, data_size, D_in = read_data(filename)
    print('Data length: %d' % data_size)

    # Setup network
    model = simpleNetV3()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Init statistics
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

        # Loop over all samples
        for ix in sample_list:
            inp, target = get_train_pair(data, ix)
            
            # Forward pass
            output = model(inp)
            
            # Calculate loss
            loss = criterion(output, target)

            # Zerog gradients
            optimizer.zero_grad()

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            loss_avg += loss
            counter += 1
            if counter % 1000 == 0:
                print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / n_epochs * 100, loss.data[0]))

        if epoch % print_every == 0:
            print('[FINISHED EPOCH %s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / n_epochs * 100, loss.data[0]))
        
        # Save weights
        if epoch % save_every == 0:
            torch.save(model.state_dict(), 'simpleNetV5_epoch_' + str(epoch) + '_' + filename + '.pkl')
            print("Model saved, epoch: %d" % (epoch))

        # Saves losses over time
        f= open("all_losses_" + filename + ".txt","w+")
        for i in range(len(all_losses)):
            f.write(str(all_losses[i]) + ',')
