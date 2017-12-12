from pytocl.driver import Driver
from pytocl.car import State, Command
from SimpleRNN import *
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
import math, random
import sys
import os
from checkpointer import *

import logging

import math

from pytocl.analysis import DataLogWriter
from pytocl.car import State, Command, MPS_PER_KMH
from pytocl.controller import CompositeController, ProportionalController, \
    IntegrationController, DerivativeController
from networks import JNetV1, JNetV2, simpleNetV2
from torch.optim import Adam
from networks import JNetV1, JNetV2, simpleNetV2 



import neat
from pytocl.main import main
import time
import visualize
from math import sqrt, exp
_logger = logging.getLogger(__name__)

D_in = 25
D_out = 3
# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

class MyDriver:
    """
    Driving logic.

    Implement the driving intelligence in this class by processing the current
    car state as inputs creating car control commands as a response. The
    ``drive`` function is called periodically every 20ms and must return a
    command within 10ms wall time.
    """

    def __init__(self, logdata=True, models=None, explore = 0.9, optimizer = None):
        self.steering_ctrl = CompositeController(
            ProportionalController(0.4),
            IntegrationController(0.2, integral_limit=1.5),
            DerivativeController(2)
        )
        self.acceleration_ctrl = CompositeController(
            ProportionalController(3.7),
        )
        self.data_logger = DataLogWriter() if logdata else None

        # Algorithm variables
        self.previous_state = None
        if os.path.isfile("memory.txt") == True:
            with open("memory.txt", "rb") as fp:
                self.replay_memory = pickle.load(fp)
        else:
            self.replay_memory = []
        # Maximum size of the replay memory
        self.max_RM_size = 100000
        self.model, self.target = models
        self.criterion = torch.nn.MSELoss()
        self.model_optim  = optimizer
        self.restart = False
        # Discretize the action space
        self.action_space = [0.9,0.7, 0.5, 0.4, 0.3, 0.2, 0.1, 0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.7, -0.9]
        self.action = 0
        self.reward = 0

        # Model that is used for accelerating and braking
        self.model_acc_brake = simpleNetV2()

        self.pheromones = []
        # self.receiver = Receiver(6001)

        weights = 'simpleNetV2_epoch_3_all_tracks.csv.pkl'

        self.model_acc_brake.load_state_dict(torch.load(weights))
        self.input = torch.zeros(D_in)

 


        # STATS
        if os.path.isfile("counter.txt") == True:
            with open("counter.txt", "rb") as fp:
                self.counter = pickle.load(fp)
        else:
            self.counter = 0
        self.counter_per_game = 0
        self.train_counter = 0
        self.average_reward = 0
        self.average_loss = 0

        
        # Hyperparameters
        self.exploration_rate = explore
        self.batch_size = 128
        self.gamma = 0.99


    @property
    def range_finder_angles(self):
        """Iterable of 19 fixed range finder directions [deg].

        The values are used once at startup of the client to set the directions
        of range finders. During regular execution, a 19-valued vector of track
        distances in these directions is returned in ``state.State.tracks``.
        """
        return -90, -75, -60, -45, -30, -20, -15, -10, -5, 0, 5, 10, 15, 20, \
            30, 45, 60, 75, 90

    def on_shutdown(self):
        """
        Server requested driver shutdown.

        Optionally implement this event handler to clean up or write data
        before the application is stopped.
        """
        if self.data_logger:
            self.data_logger.close()
            self.data_logger = None


    def get_state(self, carstate):
        state = np.zeros(26)
        state[0] = carstate.speed_x
        state[1] = carstate.speed_y
        state[2] = carstate.speed_z
        state[3] = carstate.damage
        state[4] = carstate.distance_from_center
        state[5] = carstate.angle*(np.pi/180)
        state[6] = carstate.z
        for index, edge in enumerate(carstate.distances_from_edge):
            state[7 + index] = edge
        return state

    def get_action(self, carstate, state):
        probability = np.random.uniform()
        if probability < self.exploration_rate:
            temp_action = self.steer(carstate, 0)
            action = (np.absolute(np.asarray(self.action_space) - temp_action)).argmin()# Pick the action that is closest to mediocre model

            if np.random.uniform() > 0.75:
                action = np.random.randint(0, len(self.action_space))
        else:
            state = torch.from_numpy(np.array(state))
            action = self.model(Variable(state, volatile=True).type(torch.FloatTensor)).data
            action = np.argmax(action.numpy())
        return action

    def get_reward(self, carstate):
    	#Reward function that is based on optimizing the speed in the right direction and minimizing the speed is wrong direction
        return carstate.speed_x * (np.cos(carstate.angle * (np.pi/180)) - np.absolute(np.sin(carstate.angle * (np.pi/180)))) - np.absolute(carstate.distance_from_center)
 



    def print_stats(self):
        print("The average reward is: ", self.average_reward/self.counter)
        print("The average  loss is: ", self.average_loss/self.train_counter)


    def update_state(self, pred, carstate):
        
        # # Roll the sequence so the last history is now at front
        # throwaway = self.input_sequence[0]
        # print(self.input_sequence.shape)
        
        state_t_plus = torch.FloatTensor(D_in)
        # print(state_t_plus)
        # sys.exit()
        #Overwrite old values with new prediction and carstate
        # print('[BEFORE]')
        # print(state_t_plus[0][2])
        # sys.exit()

        state_t_plus[0] = pred.data[0]
        state_t_plus[1] = pred.data[1]
        state_t_plus[2] = pred.data[2]
        state_t_plus[3] = carstate.speed_x *3.6
        state_t_plus[4] = carstate.distance_from_center
        state_t_plus[5] = carstate.angle

        for index, edge in enumerate(carstate.distances_from_edge):
            state_t_plus[6 + index] = edge
        return state_t_plus

    def drive(self, carstate: State) -> Command:
    	#Get current state
        state = self.get_state(carstate)

        # Get current action
        action = self.get_action(carstate, state)

        # Get current reward
        reward = self.get_reward(carstate)

        # Keep track of the average reward
        self.average_reward += reward

        if self.counter > 0:
            self.replay_memory.append((self.previous_state, self.action, reward, state, False))
            # Check if replay memory is full
            while len(self.replay_memory) > self.max_RM_size:
                del self.replay_memory[np.random.randint(0, len(self.replay_memory))]

        command = Command()
        # Predict brake and accelerate
        out = self.model_acc_brake(Variable(self.input))
        
        self.input = self.update_state(out, carstate)

        #Create command
        command = Command()
        command.brake = out.data[1] * 0.2

        # Maximum speed for training, 
        v_x = 300
        self.counter += 1
        self.counter_per_game += 1
        self.accelerate(carstate, v_x, command)
        command.steering = self.action_space[action]

        self.action = action
        self.previous_state = state
        if carstate.damage > 0 or carstate.last_lap_time > 0:
            command.meta = 1
            # Add terminal states to the replay memory
            if carstate.damage > 0:
                self.replay_memory.append((self.previous_state, self.action, -100, np.ones((26, )), True))
            else:
                self.replay_memory.append((self.previous_state, self.action, 1000, np.ones((26, )), True))
            self.save_variables()
            print("distance:, ", carstate.distance_from_start)
            print("------------------------------------------------------------------------------")
            
            if self.exploration_rate == 0:

                print("TEST DRIVE AVERAGE REWARD IS: ", self.average_reward/self.counter_per_game)
                print("THE DISTANCE DRIVEN BY THE AGENT IS: ", carstate.distance_from_start)
                print("EPISODE ENDED")
                print("------------------------------------------------------------------------------")
        return command

    def save_variables(self):
        with open("memory.txt", "wb") as fp: # save the Replay Memory
            pickle.dump(self.replay_memory, fp)

    def accelerate(self, carstate, target_speed, command):
        # compensate engine deceleration, but invisible to controller to
        # prevent braking:
        speed_error = 1.0025 * target_speed * MPS_PER_KMH - carstate.speed_x
        acceleration = self.acceleration_ctrl.control(
            speed_error,
            carstate.current_lap_time
        )

        # stabilize use of gas and brake:
        acceleration = math.pow(acceleration, 3)

        if acceleration > 0:
            if abs(carstate.distance_from_center) >= 1:
                # off track, reduced grip:
                acceleration = min(0.4, acceleration)

            command.accelerator = min(acceleration, 1)

            if carstate.rpm > 8000:
                command.gear = carstate.gear + 1

        # else:
        #     command.brake = min(-acceleration, 1)

        if carstate.rpm < 2500:
            command.gear = carstate.gear - 1

        if not command.gear:
            command.gear = carstate.gear or 1
        if command.gear < 1:
            command.gear = 1

    def steer(self, carstate, target_track_pos):
        steering_error = target_track_pos - carstate.distance_from_center
        steering = self.steering_ctrl.control(
            steering_error,
            carstate.current_lap_time
        )
        return steering

