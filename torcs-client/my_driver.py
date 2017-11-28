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
from networks import simpleNetV2, simpleNetV3, simpleNetV5
import re

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


class simpleNetV6(nn.Module):
    def __init__(self):
        super(simpleNetV6, self).__init__()
        self.D_in = 25
        self.h1_size = 300
        self.h2_size = 300
        self.h3_size = 300
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

        self.model = simpleNetV2()
        weights = 'simpleNetV2_epoch_3_all_tracks.csv.pkl'

        self.model.load_state_dict(torch.load(weights))
        self.input = torch.FloatTensor(D_in)
        self.counter = 0
        self.name = '3001'
        self.log = []
        self.logname = '../logs/example3.log'

        logging.basicConfig(filename=self.logname, level=logging.INFO, format='%(message)s')
        
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
        """
        Produces driving command in response to newly received car state.

        """

        # Create new network input based on previous states

        out = self.model(Variable(self.input))
        
        self.input = self.update_state(out, carstate)


        #Create command
        command = Command()
        command.accelerator = out.data[0]
        command.brake = out.data[1]
        command.steering = out.data[2]
        # command.gear = 1

        # if out.data[0] > 0.5:
            # command.accelerator = 0.4
        # command.brake = 0.1
        # print(out.data[0][0])
        # sys.exit()

        v_x = 500
        self.accelerate(carstate, v_x, command)

        self.counter += 1
        # if self.counter == 20000:
            # sys.exit()

        if self.counter % 20 == 0:
            # print(command)
            self.printToLog(carstate, command)
            self.readFromLog(carstate)

        if self.counter % 1000 == 0:
            print(self.log)

        self.input[2] = command.accelerator
        

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

    def printToLog(self, carstate, command):
        # print(carstate)
        # sys.exit()
        # print(self.name, carstate.distance_from_start, carstate.distance_from_center, command.brake, command.accelerator)
        logging.info('%s,%d,%d,%d,%d', self.name, carstate.distance_from_start, carstate.distance_from_center, command.brake, command.accelerator)


    def readFromLog(self, carstate):
        f = open(self.logname, "r")
        lines = f.readlines()
        for line in lines:
            if re.match('3001', line):
                if not line in self.log:
                    self.log.append(line)
                    

        # print(self.log)
        # print(lines)
        # sys.exit()


