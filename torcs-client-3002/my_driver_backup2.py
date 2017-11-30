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

sequence_size = 10
batch_size = 1
row_size = 25
hidden_size = 25
n_hidden_layers = 3

D_in = 25
D_out = 3

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
        self.model = JNetV2()
        self.model.init_states()
        
        self.model = JNetV1(D_in, hidden_size, n_hidden_layers, D_out)
        
        filename = 'JNetV19.pkl'
        self.model.load_state_dict(torch.load(filename))
        self.model.hidden = self.model.init_hidden()
        self.pred = [1.0, 0, 0]
        self.counter = 0
        init_seq = torch.zeros(sequence_size, batch_size, D_in)
        
        # Set inital first 10 states to full acceleration
        for state in init_seq:
            state[0][0] = 1.0

        self.input_sequence = Variable(init_seq)

        print('Using: ' + filename)
        


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

        c_state = torch.zeros(1, batch_size, D_in)
        # print(self.input_sequence)

        # print(carstate)
        # sys.exit()

        speed_kph = carstate.speed_x *3.6

        # TODO: actual speed
        c_state[0][0][0] = self.pred[0]
        c_state[0][0][1] = self.pred[1]
        c_state[0][0][2] = self.pred[2]
        c_state[0][0][3] = speed_kph
        c_state[0][0][4] = carstate.distance_from_center
        c_state[0][0][5] = carstate.angle
        for index, edge in enumerate(carstate.distances_from_edge):
            c_state[0][0][6 + index] = edge
        # print(c_state)
        # sys.exit()        
        # print(c_state)
        # print(self.input_sequence)
        self.input_sequence = torch.cat((self.input_sequence, c_state), 0)[1:1+sequence_size]
        # print(torch.cat(self.input_sequence))
        # sys.exit()

        # self.input_sequence = torch.cat((c_state, self.input_sequence), 1)[0:sequence_size]
        # print(self.input_sequence)
        # print(self.input_sequence)
        out, self.model.hidden = self.model(self.input_sequence, self.model.hidden)
        # out2, self.model.hidden = self.model(self.input_sequence, self.model.hidden)
        # print(out)
        # print(out2)

        # out, _ = self.model(self.input_sequence, self.model.states)

        command = Command()
        if out.data[0][0] > 0.5:
            command.accelerator = 0.4
        # print(out.data[0][0])
        # sys.exit()

        # command.accelerator = out.data[0][0] 


        
        # if speed_kph > 30:
        #     command.accelerator = 0
        # else:
        #     command.accelerator = out.data[0][0]    

        command.brake = out.data[0][1]
        # if abs(out.data[0][2]) < 0.003:
            # command.steering = 0
        # else:       
            # command.steering = -out.data[0][2]
        command.steering = out.data[0][2]

        # command.steering = 0.5
        # print(self.pred)
        self.pred[0] = out.data[0][0]
        self.pred[1] = command.brake
        self.pred[2] = command.steering
        # print(self.pred)
        # print(out.data[0][0])

        # sys.exit()


        v_x = 60
        # command.gear = 



        self.accelerate(carstate, v_x, command)

        self.counter += 1

        if self.counter % 20 == 0:
            # print(self.input_sequence)
            print(out)
            # print(command)
            # print(carstate.distance_from_center)
            # print(c_state[0][0][3])
            # sys.exit()

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

