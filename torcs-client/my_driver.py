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
import neat
import pickle

from pytocl.analysis import DataLogWriter
from pytocl.car import State, Command, MPS_PER_KMH
from pytocl.controller import CompositeController, ProportionalController, \
    IntegrationController, DerivativeController
from networks import JNetV1, JNetV2, simpleNetV2
import os.path


_logger = logging.getLogger(__name__)

# seq_length = 10
batch_size = 1
row_size = 25
h_layer_size = 25
n_hidden_layers = 3

D_in = 25
D_out = 3

seq_length = 5 # number of steps to unroll the RNN for

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
        self.input = torch.zeros(D_in)
        self.crashCounter = 0
        self.reverseCounter = 0
        self. forwardCounter = 0
        self.resetGear = False
        self.crashed  = False
        self.counter = 0
        self.name = '3001'

        # NEAT
        self.history = np.zeros((5, 2), dtype=float)
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config-neat')
        with open('winner-feedforward', 'rb') as f:
            winner = pickle.load(f)
        self.net = neat.nn.FeedForwardNetwork.create(winner, config)

        #SWARM
        self.pheromones = []
        pickle.dump(self.pheromones, open("../sent_3001.txt", "wb" ) )
        self.straight_begin = 0
        self.straight_end = 0
        self.list_straight = []
        self.list_corner = []
        self.received = []
        self.list_average_corcer = []


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


    def make_input(self, carstate, out):
        inputs = [carstate.rpm/10000, carstate.speed_x/100, out[0], out[1]]
        return inputs

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

    def update_history(self, carstate):
        # Roll the sequence so the last history is now at front
        self.history = np.roll(self.history, 1)
        self.history[0][0] = carstate.distance_from_start

    def detect_crash(self):
        if (self.history[0][0] - self.history[-1][0] < 0.2):
            self.crashCounter += 1
        else:
            self.crashCounter = 0

        if self.crashCounter > 100:
            self.crashed = True
            self.crashCounter = 0

    def check_back_on_track(self, carstate):
        if self.reverseCounter < 200:
            # print(abs(carstate.distance_from_center))
            
            if abs(carstate.distance_from_center) < 0.5 and abs(carstate.angle) < 45:
                # print(abs(carstate.angle))
                # print("-------------Found center!")
                self.crashed = False
                self.forwardCounter = 250
                self.reverseCounter = 0
            else:
                self.reverseCounter += 1
        else:
            # print("---------------ReverseCounter Finished!")
            self.crashed = False
            self.forwardCounter = 250
            self.reverseCounter = 0

    def check_for_corner(self, steering, distance_from_start):
        begin_corner = 0
        if steering > 0.4 or steering < -0.4:
            self.list_corner.append(distance_from_start)
            self.list_average_corcer.append(steering)
        else:
            if len(self.list_corner) > 25:
                average = np.average(np.asarray(self.list_average_corcer))
                begin_corner = self.list_corner[0]
                print(average)
            self.list_corner = []
        return begin_corner
        

    def check_for_acceleration(self, steering, distance_from_start):
        begin_accel = 0
        end_accel = 0
        if steering > -0.05 and steering < 0.05:
            self.list_straight.append((steering, distance_from_start))
        else:
            if len(self.list_straight) > 500:
                begin_accel = self.list_straight[0][1]
                end_accel = self.list_straight[-1][1] - 100
                if begin_accel > end_accel:

                    self.pheromones.append(("Accel", begin_accel, 100000))
                    self.begin_accel = 0
                self.list_straight = []

        return begin_accel, end_accel

    def check_for_info(self, received, distance_from_start, carspeed):
        for info in received:
            #if info[0] == "Accel" and distance_from_start > info[1] + 75  and distance_from_start < info[2] - 30:
                #return 360, 1, 0, None
            if info[0] == "Corner" and distance_from_start > info[1] - 75 and distance_from_start < info[1] - 20:
                if carspeed * 3.6 < 135:
                    return 135, 0.6 , 0, None
                return 135, 0, 0.6, 0
        return None, None, None, None 

    def drive(self, carstate: State) -> Command:
        command = Command()
        out = self.model(Variable(self.input))
        self.input = self.update_state(out, carstate)

        # If crashed -> reversing
        if self.crashed:
            self.check_back_on_track(carstate)

            command.accelerator = 0.5
            command.gear = -1

            # Only steer if incorrect angle and off track
            if abs(carstate.angle) > 20 and abs(carstate.distance_from_center) < 1:

                # If car orientated towards the right
                if carstate.angle > 0:

                    # Steer to right while reversing
                    command.steering = -1
                        # self.forward_steer_direction = 1
                    self.forward_steer_direction = 0.5

                else:
                    # Steer to left while reversing
                    command.steering = 1
                        
                    # else:
                         # Steer to right while reversing
                        # command.steering = -1
                        # self.forward_steer_direction = 1
                    self.forward_steer_direction = -0.5


            # command.steering = 1
            # self.forwardCounter -= 1
            self.resetGear = True
        
        
        elif self.forwardCounter > 0:
            # print("Turning to correct direction")
            if self.forwardCounter > 200:
                command.brake = 1
            
            # if abs(carstate.angle) > :
            if carstate.angle > 0:
                command.steering = 0.5
            else:
                command.steering = -0.5
            # command.steering = self.forward_steer_direction
            command.accelerator = 0.5
            command.gear = 1
            self.forwardCounter -= 1

        # Normal behaviour
        else:

            out = self.model(Variable(self.input))
            
            self.input = self.update_state(out, carstate)
            self.update_history(carstate)
            self.detect_crash()


            command.accelerator = out.data[0]
            command.brake = out.data[1]
            command.steering = out.data[2]

            begin_corner = self.check_for_corner(command.steering, carstate.distance_from_start)
            begin_acceleration, end_acceleration = self.check_for_acceleration(command.steering, carstate.distance_from_start)
            if begin_corner > 0:
                self.pheromones.append(("Corner", begin_corner))
            if begin_acceleration > 0:
                self.pheromones.append(("Accel", begin_acceleration, end_acceleration))
            if self.counter % 50 == 0:
                pickle.dump(self.pheromones, open("../sent_3001.txt", "wb" ) )
                if os.path.isfile("../sent_3001.txt"):
                    self.received = pickle.load( open( "../sent_3001.txt", "rb" ) )
            v_x = 500

            max_speed, accel, brake, steer = self.check_for_info(self.received, carstate.distance_from_start, carstate.speed_x)
            if max_speed is not None:
                v_x = max_speed
                command.accelerator = accel
                command.brake = brake
                if steer == 0:
                   command.steering = steer

            #command.gear = self.get_gear(carstate)  USE NEAT TO SHIFT GEARS

            self.accelerate(carstate, v_x, command)
            if self.resetGear:
                #command.gear = self.get_gear(carstate)
                self.resetGear = False 

            if carstate.speed_x * 3.6 < 20:
                command.brake = 0

        self.counter += 1


        self.input[2] = command.accelerator
        return command

    def get_gear(self, carstate):
        output = self.net.activate([carstate.rpm/10000])

        # convert the outcome into a command
        command = Command()
        if(output[0] < -0.5):
            gear = carstate.gear - 1
        elif(output[0] > 0.5):
            gear = carstate.gear + 1
        else:
            gear = carstate.gear
        if gear < 1:
            gear = 1
        if gear > 6:
            gear = 6
        return gear

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

        if carstate.rpm < 4000:
            command.gear = carstate.gear - 1

        if not command.gear:
            command.gear = carstate.gear or 1
     

    def steer(self, carstate, target_track_pos, command):
        steering_error = target_track_pos - carstate.distance_from_center
        command.steering = self.steering_ctrl.control(
            steering_error,
            carstate.current_lap_time
        )


#-------------------GRAVEYARD

    # def readFromLog(self, carstate):
    #     f = open(self.logname, "r")
    #     lines = f.readlines()
    #     for line in lines:
    #         if re.match('3001', line):
    #             if not line in self.log:
    #                 self.log.append(line)
                    

        # print(self.log)
        # print(lines)
        # sys.exit()


# if self.counter % 100 == 0:
        #     print("Sending mesage")
        #     msg = self.sender.sendReceive(carstate.distance_from_start, )
        #     self.pheromones.append(msg)
        #     print(self.pheromones)
            # print("Message received:")
            # print(msg)

    # def printToLog(self, carstate, command):
    #     # print(carstate)
    #     # sys.exit()
    #     # print(self.name, carstate.distance_from_start, carstate.distance_from_center, command.brake, command.accelerator)
    #     logging.info('%s,%d,%d,%d,%d', self.name, carstate.distance_from_start, carstate.distance_from_center, command.brake, command.accelerator)

