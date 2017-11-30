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

import logging

import math

from pytocl.analysis import DataLogWriter
from pytocl.car import State, Command, MPS_PER_KMH
from pytocl.controller import CompositeController, ProportionalController, \
    IntegrationController, DerivativeController
from networks import JNetV1, JNetV2

_logger = logging.getLogger(__name__)

# seq_length = 10
batch_size = 1
row_size = 25
h_layer_size = 25
n_hidden_layers = 3

D_in = 25
D_out = 3

seq_length = 5 # number of steps to unroll the RNN for


class simpleGRU(nn.Module):
    def __init__(self, D_in, h_layer_size, n_hidden_layers, D_out):
        super(simpleGRU, self).__init__()
        self.D_in = D_in
        self.h_layer_size = h_layer_size
        self.D_out = D_out
        self.n_hidden_layers = n_hidden_layers

        self.encoder = nn.Linear(D_in, h_layer_size)
        self.gru = nn.GRU(h_layer_size, h_layer_size, n_hidden_layers)
        self.decoder = nn.Linear(h_layer_size, D_out)
    
    def forward(self, input, hidden):
        # print(input, hidden)
        # sys.exit()
        # print(input.view(1,-1))
        encoded = self.encoder(input.view(1, -1))
        # print(encoded)
        # sys.exit()
        out, hidden = self.gru(encoded.view(1, 1, -1), hidden)
        output = self.decoder(out.view(1, -1))

        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.n_hidden_layers, 1, self.h_layer_size))


class MyDriver(Driver):


    # Override the `drive` method to create your own driver
    ...
    # def drive(self, carstate: State) -> Command:
    #     # Interesting stuff
    #     command = Command(...)
    #     return command
    def __init__(self, logdata=True):
        self.steering_ctrl = CompositeController(
            ProportionalController(0.4),
            IntegrationController(0.2, integral_limit=1.5),
            DerivativeController(2)
        )
        self.acceleration_ctrl = CompositeController(
            ProportionalController(3.7),
        )
        self.data_logger = DataLogWriter() if logdata else None

        self.model = simpleGRU(D_in, h_layer_size, n_hidden_layers, D_out)
        weights = 'saves/simpleGRU_epoch_99_all_tracks.csv.pkl'

        self.model.load_state_dict(torch.load(weights))
        self.hidden = self.model.init_hidden()
        self.nstate = torch.FloatTensor(1, D_in)
        
        self.pred = [1.0, 17, 20]
        self.counter = 0
        state_history = torch.zeros(seq_length, D_in)
        
        # Set inital first 10 states to full acceleration
        for state in state_history:
            state[0] = 1.0

        # state_history[0][0] = 20.0
        # state_history[0][1] = 17.0

        # print(state_history)
        # sys.exit()
        self.input_sequence = state_history

        print('Using: ' + weights)
        


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
    
    
    def update_history_seq_len(self, output, carstate):
        
        # Roll the sequence so the last history is now at front
        throwaway = self.input_sequence[0]
        keep = self.input_sequence[1:]
        self.input_sequence = torch.cat((keep, throwaway.view(1, -1)), 0)

        #Overwrite old values with new prediction and carstate
        self.input_sequence[-1][0] = self.pred[0]
        self.input_sequence[-1][1] = self.pred[1]
        self.input_sequence[-1][2] = self.pred[2]
        self.input_sequence[-1][3] = carstate.speed_x *3.6
        self.input_sequence[-1][4] = carstate.distance_from_center
        self.input_sequence[-1][5] = carstate.angle

        for index, edge in enumerate(carstate.distances_from_edge):
            self.input_sequence[-1][6 + index] = edge

    def update_state(self, pred, carstate):
        
        # # Roll the sequence so the last history is now at front
        # throwaway = self.input_sequence[0]
        # print(self.input_sequence.shape)
        
        state_t_plus = torch.FloatTensor(1, D_in)
        # print(state_t_plus)
        # sys.exit()
        #Overwrite old values with new prediction and carstate
        # print('[BEFORE]')
        # print(state_t_plus[0][2])
        # sys.exit()

        state_t_plus[0][0] = pred.data[0][0]

        state_t_plus[0][1] = pred.data[0][1]
        state_t_plus[0][2] = pred.data[0][2]
        state_t_plus[0][3] = carstate.speed_x *3.6
        state_t_plus[0][4] = carstate.distance_from_center
        state_t_plus[0][5] = carstate.angle

        for index, edge in enumerate(carstate.distances_from_edge):
            state_t_plus[0][6 + index] = edge
        # print('[AFTER]')
        # print(state_t_plus[0][2])
        return state_t_plus

    def drive(self, carstate: State) -> Command:
        """
        Produces driving command in response to newly received car state.

        """

        # Create new network input based on previous states

        #---------- One by one
        # inp = self.nstate
        # true_inp = Variable(inp)
        # newOut, self.hidden = self.model(true_inp, self.hidden)

        #------------------
        # Full unroll
        # print(self.input_sequence)
        # sys.exit()

        # sys.exit()
        for c in range(seq_length):
            out_unrolled, self.hidden = self.model(Variable(self.input_sequence[c]), self.hidden)

        print(out_unrolled)

        # print(out_unrolled)
        # sys.exit()
        
        # Update input sequence
        # self.nstate = self.update_state(newOut, carstate) # One by one
        self.update_history_seq_len(out_unrolled, carstate) # Full unroll


        #Create command
        command = Command()
        command.accelerator = out_unrolled.data[-1][0]
        command.brake = out_unrolled.data[-1][1]
        command.steering = out_unrolled.data[-1][2]
        command.gear = 1

        # if out.data[0][0] > 0.5:
        #     command.accelerator = 0.4
        # print(out.data[0][0])
        # sys.exit()

        # v_x = 60
         # self.accelerate(carstate, v_x, command)

        self.counter += 1
        # if self.counter == 20000:
            # sys.exit()

        if self.counter % 20 == 0:
            print(command)


        return command

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

    def steer(self, carstate, target_track_pos, command):
        steering_error = target_track_pos - carstate.distance_from_center
        command.steering = self.steering_ctrl.control(
            steering_error,
            carstate.current_lap_time
        )

