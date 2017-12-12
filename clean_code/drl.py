import numpy as np
import argparse
from copy import deepcopy
import torch

from pytocl.main import main
from ddpg_driver import *
from ddpg_networks import *
from torch.optim import RMSprop, Adam


# Deep reinforcement learning to train an torcs agents


def train_agent():
    NUMBER_OF_EPISODES = 1000
    if os.path.isfile("trained_model.h5") == True:
        model = torch.load("trained_model.h5")
        target = torch.load("trained_model.h5")
    else:
        model = DQN(26, 15)
        save_model(model)
        target = torch.load("trained_model.h5") # Make sure target is with the same weight
    models = (model, target)
    exploration_rate = 0.9 # Exploratiore states in the enviroment
    

    # Optimizer
    optim = Adam(model.parameters(), lr=0.000025)
    #Loss function
    criterion = torch.nn.MSELoss()

    for episode in range(NUMBER_OF_EPISODES):
        print("THE GAME NUMBER IS: ", episode)
        agent = main(MyDriver(logdata=False, models=models, explore=exploration_rate))
        average_loss = 0
        train_counter = 0   
        for n in range(20):
            loss = train_policy(optim, criterion, model, target)
            average_loss += loss
            train_counter += 1
        print(average_loss/train_counter)
        if episode % 2 == 0 and episode > 0:
            save_model(model)
            exploration_rate = exploration_rate - 0.05
            if episode % 10 == 0:
                target = torch.load("trained_model.h5") # Update target network
                print("TARGET NETWORK IS UPDATED")

                models = (model, target)
            agent = main(MyDriver(logdata=False, models=models, explore=0, optimizer = optim))# Test drive to see how well agent is behaving
            if exploration_rate < 0.1:# Update exploration rate
                exploration_rate = 0.1
            print("EXPLORATION IS: ", exploration_rate)



# Saving the agent
def save_model(model):
    torch.save(model,"trained_model.h5")

# Get a batch of training samples
def get_batch():
    batch_size = 128
    # Loading the replay memory from the games played
    with open("memory.txt", "rb") as fp:
        replay_memory = pickle.load(fp)
    state_batch = np.ones((batch_size, 26))
    action_batch = np.ones((batch_size, ))
    reward_batch = np.ones((batch_size, ))
    next_state_batch = np.ones((batch_size, 26))
    terminal_batch = np.ones((batch_size, ))
    for n in range(batch_size):
        state, action, reward, next_state, terminal = replay_memory[np.random.randint(0, len(replay_memory))]
        state_batch[n] = state
        action_batch[n] = action
        reward_batch[n] = reward
        next_state_batch[n] = next_state
        if terminal == True:
            terminal_batch[n] = 0
    return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch


# Training the agent by minimising the error function
def train_policy(optim, criterion, model, target):

    state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = get_batch()


    state_batch = Variable(torch.from_numpy(state_batch)).type(torch.FloatTensor)
    action_batch = Variable(torch.from_numpy(action_batch)).type(torch.LongTensor)
    reward_batch = Variable(torch.from_numpy(reward_batch)).type(torch.FloatTensor)
    terminal_batch = Variable(torch.from_numpy(terminal_batch)).type(torch.FloatTensor)
    next_state_batch = Variable(torch.from_numpy(next_state_batch)).type(torch.FloatTensor)

    # Prepare for the target q batch
    next_max_q_values = target(next_state_batch)
    next_max_q_values = Variable(next_max_q_values.data)
    next_max_q_values, _ = next_max_q_values.max(dim = 1, keepdim=True)
    next_max_q_values = next_max_q_values * terminal_batch.unsqueeze(1)

    current_q_values = model(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze()
    #Discount is 0.99 in order to keep future rewards into account
    expected_q_values = reward_batch + 0.99 * next_max_q_values.squeeze()    # 0.2.0


    loss = criterion(current_q_values, expected_q_values)
    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss


train_agent()
