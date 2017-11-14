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

_logger = logging.getLogger(__name__)

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, H2, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.rnn = torch.nn.LSTM(H, H2, 5)
        self.linear2 = torch.nn.Linear(H2, H)
        self.tanh = torch.nn.Linear(H, D_out)

    def forward(self, x, hidden=None):
        input = self.linear1(x.view(1, -1)).unsqueeze(1)
        output, hidden = self.rnn(input, hidden)
        output = self.linear2(output.squeeze(1))
        output2 = self.tanh(output)
        return output2, hidden


sequence_size = 3
row_size = 25
D_in = sequence_size * 25
D_out = 3
hidden_size = 1024
hidden_size_2 = 512

fullData = np.zeros((row_size, sequence_size))
#initData = np.empty(25)
#for i in range(sequence_size):
#	fullData = np.append(fullData, initData)


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
        self.model = TwoLayerNet(D_in, hidden_size, hidden_size_2, D_out)
        self.model.load_state_dict(torch.load('LaurensNet30.pkl'))

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

    def drive(self, carstate: State) -> Command:
        """
        Produces driving command in response to newly received car state.

        This is a dummy driving routine, very dumb and not really considering a
        lot of inputs. But it will get the car (if not disturbed by other
        drivers) successfully driven along the race track.

        """
        global fullData
        data = np.empty(25)
        data[3] = carstate.speed_x
        data[4] = carstate.z
        data[5] = carstate.angle
        for index, edge in enumerate(carstate.distances_from_edge):
        	data[3 + index] = edge
        # data = torch.from_numpy(data.T)
        # data = data.type(torch.FloatTensor)
        # data=torch.unsqueeze(data,1)
        # data =  Variable(data)

        net_input = torch.from_numpy(fullData.flatten())
        net_input = Variable(net_input.type(torch.FloatTensor))

        # force = np.random.random() < 0.5
        # data = torch.transpose(data, 0, 1)
        predict, hidden_layer = self.model.forward(net_input)
        command = Command()
        predict = predict[0].data.numpy()
        data[0] = predict[0]
        data[1] = predict[1]
        data[2] = predict[2]
        if predict[0] > 0.5:
        	command.accelerator = 1
        else:
        	command.accelerator = 0
        if predict[1] > 0.5:
            command.brake = 1
            print("BRAKING")
        else:
        	command.brake = 0
        command.steer = predict[2]
        self.steer(carstate, 0.0, command)


        # ACC_LATERAL_MAX = 6400 * 5
        # v_x = min(80, math.sqrt(ACC_LATERAL_MAX / abs(command.steering)))
        v_x = 300

        self.accelerate(carstate, v_x, command)

        if self.data_logger:
            self.data_logger.log(carstate, command)

        fullData = np.delete(fullData, 0, axis=1)
        fullData = np.append(fullData, data.reshape((25,1)), axis=1)

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

