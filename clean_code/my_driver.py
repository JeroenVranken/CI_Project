from pytocl.driver import Driver
from pytocl.car import State, Command
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
import math, random
import sys
from networks import simpleNetV3
import re

import logging

import math
import pickle

from pytocl.analysis import DataLogWriter
from pytocl.car import State, Command, MPS_PER_KMH
from pytocl.controller import CompositeController, ProportionalController, \
    IntegrationController, DerivativeController
from networks import simpleNetV3
import os.path


_logger = logging.getLogger(__name__)


# Number of input for the network
D_in = 25

# Number of outputs for the network
D_out = 3

class MyDriver(Driver):

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

        # Setup the PyTorch model and load the weights
        self.model = simpleNetV3()
        weights = 'simpleNetV3_lr=0.01_epoch_1_all_tracks.csv.pkl'
        self.model.load_state_dict(torch.load(weights))
        
        # Initialize inputs, counters and history
        self.input = torch.zeros(D_in)
        self.crashCounter = 0
        self.reverseCounter = 0
        self.forwardCounter = 0
        self.resetGear = False
        self.crashed  = False
        self.counter = 0
        self.name = '3001'
        self.history = np.zeros((5, 2), dtype=float)

        # Initialize SWARM
        self.pheromones = []
        pickle.dump(self.pheromones, open("sent_3001.txt", "wb" ) )
        self.straight_begin = 0
        self.straight_end = 0
        self.list_straight = []
        self.list_corner = []
        self.received = []
        self.list_average_corner = []


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
    
    # Create new input for the network with predicted values and carstate
    def update_state(self, pred, carstate):
        
        # Create empty vector
        state_t_plus = torch.FloatTensor(D_in)

        # Fill with predicted values and carstate
        state_t_plus[0] = pred.data[0][0]
        state_t_plus[1] = pred.data[0][1]
        state_t_plus[2] = pred.data[0][2]
        state_t_plus[3] = carstate.speed_x *3.6
        state_t_plus[4] = carstate.distance_from_center
        state_t_plus[5] = carstate.angle

        for index, edge in enumerate(carstate.distances_from_edge):
            state_t_plus[6 + index] = edge

        return state_t_plus

    # Update the distance_from_start history
    def update_history(self, carstate):
        # Roll the sequence so the last history is now at front
        self.history = np.roll(self.history, 1)
        self.history[0][0] = carstate.distance_from_start

    # Checks if distance_from_start in the last 5 ticks increased less than 0.2 meters,
    # if so, adds one to the crashCounter. 
    # Once the crashCounter reaches 100, sets crashed to true which starts auto-reset behaviour 
    def detect_crash(self):
        # Check if not standing still or moving really slowly
        if (self.history[0][0] - self.history[-1][0] < 0.2):
            self.crashCounter += 1
        else:
            self.crashCounter = 0

        # Activate auto-reset behaviour
        if self.crashCounter > 100:
            self.crashed = True
            self.crashCounter = 0

    # Checks whether the car is back on track and has the correct angle.
    # If after 200 ticks still not back on track, start driving forward
    def check_back_on_track(self, carstate):
        if self.reverseCounter < 200:
            
            # Check back on track with correct angle, if correct activate forward counter
            if abs(carstate.distance_from_center) < 0.5 and abs(carstate.angle) < 45:
                self.crashed = False
                self.forwardCounter = 250
                self.reverseCounter = 0
            else:
                self.reverseCounter += 1
        # Reversecounter finished, so start moving in forward direction
        else:
            self.crashed = False
            self.forwardCounter = 250
            self.reverseCounter = 0

    # Detects sharp corners by checking if the steering angle is more than 0.4 for 25 ticks  
    def check_for_corner(self, steering, distance_from_start):
        begin_corner = 0
        
        # Check if steering angle is more than 0.4 in either direction
        if steering > 0.4 or steering < -0.4:
            self.list_corner.append(distance_from_start)
            self.list_average_corner.append(steering)
        else:
            # Check if steering sharply for more than 25 ticks
            if len(self.list_corner) > 25:
                average = np.average(np.asarray(self.list_average_corner))
                begin_corner = self.list_corner[0]
                print(average)
            self.list_corner = []
        return begin_corner

    # Checks for pheromones placed on track, if 20 meters ahead, start braking
    def check_for_info(self, received, distance_from_start, carspeed):
        for info in received:
            if info[0] == "Corner" and distance_from_start > info[1] - 75 and distance_from_start < info[1] - 20:
                if carspeed * 3.6 < 135:
                    return 135, 0.6 , 0, None
                return 135, 0, 0.6, 0
        return None, None, None, None 

    # Returns appropriate commands for the car, depending on the currenst carstate
    def drive(self, carstate: State) -> Command:
        command = Command()

        # If crashed, reverse
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
                    self.forward_steer_direction = 0.5

                # Car orientated towards left
                else:
                    # Steer to left while reversing
                    command.steering = 1
                    self.forward_steer_direction = -0.5

            self.resetGear = True
        
        # If forwardCounter is activated, drive forward, steering in the correct direction
        elif self.forwardCounter > 0:

            # Start by braking
            if self.forwardCounter > 200:
                command.brake = 1
            
            # Then drive forward steering in the correct direction
            if carstate.angle > 0:
                command.steering = 0.5
            else:
                command.steering = -0.5

            command.accelerator = 0.5
            command.gear = 1

            self.forwardCounter -= 1

        # Normal behaviour
        else:
            # Run input through model to predict next commands
            out = self.model(Variable(self.input))
            self.input = self.update_state(out, carstate)
            
            self.update_history(carstate)
            self.detect_crash()

            # Create command with predictions
            command.accelerator = out.data[0][0]
            command.brake = out.data[0][1]
            command.steering = out.data[0][2]


            # Check for corners, if found, place pheromones
            begin_corner = self.check_for_corner(command.steering, carstate.distance_from_start)
          
            if begin_corner > 0:
                self.pheromones.append(("Corner", begin_corner))

            # Save pheromones to file
            if self.counter % 50 == 0:
                pickle.dump(self.pheromones, open("sent_3001.txt", "wb" ) )
                if os.path.isfile("sent_3001.txt"):
                    self.received = pickle.load( open( "sent_3001.txt", "rb" ) )

            # Maximum target speed
            v_x = 500

            # Use information from pheromones
            max_speed, accel, brake, steer = self.check_for_info(self.received, carstate.distance_from_start, carstate.speed_x)
            if max_speed is not None:
                v_x = max_speed
                command.accelerator = accel
                command.brake = brake
                if steer == 0:
                   command.steering = steer


            self.accelerate(carstate, v_x, command)
            if self.resetGear:
                self.resetGear = False 

            # Prevent the car from braking when driving slowly, causing it to get stuck when off track
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
     

    # def steer(self, carstate, target_track_pos, command):
    #     steering_error = target_track_pos - carstate.distance_from_center
    #     command.steering = self.steering_ctrl.control(
    #         steering_error,
    #         carstate.current_lap_time
    #     )
