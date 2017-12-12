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

from pytocl.analysis import DataLogWriter
from pytocl.car import State, Command, MPS_PER_KMH
from pytocl.controller import CompositeController, ProportionalController, \
    IntegrationController, DerivativeController
from networks import JNetV1, JNetV2, simpleNetV2

import neat
from pytocl.main import main
import time
import visualize
from math import sqrt, exp
_logger = logging.getLogger(__name__)

D_in = 25
D_out = 3


class NeatDriver:
    """
    In this current implementation this neatdriver drives on the track by commands from
    the output of a neural network namely the simpleNetV2_epoch_3_all_tracks.csv.pkl.
    The command for the gear is predicted by the neural network that is created with the neat-
    algorithm. The winner genome 'winner-feedforward' is loaded in and controlling the gear.
    Below this class there is code of how the winner-feedforward is obtained.
    """

    def __init__(self, logdata=True, net=None):
        self.steering_ctrl = CompositeController(
            ProportionalController(0.4),
            IntegrationController(0.2, integral_limit=1.5),
            DerivativeController(2)
        )
        self.acceleration_ctrl = CompositeController(
            ProportionalController(3.7),
        )
        self.data_logger = DataLogWriter() if logdata else None
        self.counter = 0

        # import the neural net that drives the car
        self.model = simpleNetV2()
        self.weights = 'simpleNetV2_epoch_3_all_tracks.csv.pkl'
        self.model.load_state_dict(torch.load(self.weights))
        self.input = torch.FloatTensor(D_in)
        self.track_check1 = False
        self.track_check2 = False

        # import the neat neural network to handle the gear
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config-neat')
        with open('winner-feedforward', 'rb') as f:
            winner = pickle.load(f)
        self.net = neat.nn.FeedForwardNetwork.create(winner, config)

        self.clock = time.time()
        self.done = False

        # initialize the fitness with zero
        self.temp_fitness = 0


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

    def update_state(self, pred, carstate):

        state_t_plus = torch.FloatTensor(D_in)
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

        global net
        # define the output
        out = self.model(Variable(self.input))
        self.input = self.update_state(out, carstate)       

        # code to check wheter the driver is stuck, namely if it is for a long time of track
        if self.track_check1 == True and self.check_offtrack(carstate) == True:
            self.track_check2 = True

        if self.counter % 100 == 0:
            self.track_check1 = self.check_offtrack(carstate)

        # this is a previous used fitness function
        #self.temp_fitness += carstate.speed_x * (np.cos(carstate.angle*(np.pi/180)) - np.absolute(np.sin(carstate.angle * (np.pi/180))))

        # calculate the fitness
        if(carstate.rpm > 8000 or carstate.rpm < 2500):
            reward = -1
        else:
            reward = 1
        self.temp_fitness += reward

        # calculate the gear with the neat network
        output = self.net.activate([carstate.rpm/10000])

        # convert the outcome into a command
        command = Command()
        if(output[0] < -0.5):
            command.gear = carstate.gear - 1
        elif(output[0] > 0.5):
            command.gear = carstate.gear + 1
        else:
            command.gear = carstate.gear
        if command.gear < 1:
            command.gear = 1
        if command.gear > 6:
            command.gear = 6

        # Define the command with output of the other neural net
        command.accelerator = out.data[0]
        command.brake = out.data[1]
        command.steering = out.data[2]

        # update the counter
        self.counter += 1

        # log the command
        if self.data_logger:
            self.data_logger.log(carstate, command)
        
        # If car has too much damage or max time has exceeded do a restart
        if carstate.damage >= 9500 or carstate.last_lap_time > 0 or carstate.current_lap_time > 120:
            global fitness
            fitness =  (self.temp_fitness/self.counter)  
            command.meta = 1
        return command


    def check_offtrack(self, carstate):
        """
        Function that checks whether the car is offtrack
        param: carstate
        return: boolean
        """
        counter = 0
        for values in carstate.distances_from_edge:
            if values == -1:
                counter += 1
        if counter == len(carstate.distances_from_edge):
            return True
        else: 
            return False

"""
From here the code starts for evolving a network to control the gear with neat
"""

# initialize values
net = None
fitness = 0
counter = 1 
eta = 1

# Use the NN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
    """
    evaluate the genome by giving it a fitness
    param: genome
    param: config file
    return: fitness
    """
    
    global net
    global counter
    global fitness

    # create network
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # initialize the driver with the network
    driver = main(NeatDriver(logdata=False, net=net))
    print("ID number of genome: ", counter)
    print("The fitness is: ", fitness + eta)
    print("----------------------------------------")
    counter += 1
    
    # fitness is + eta so it will always be a real number
    return fitness + eta

# loop through genomes and evaluate them
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)

def run():
    # Load the config file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-neat')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    #create population or restore a population
    pop = neat.Population(config)
    #pop = Checkpointer.restore_checkpoint('neat-checkpoint-4')

    # statistic outputs
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(Checkpointer(generation_interval=15))

    # run the algorithm for max 300 generations and find the winner
    winner = pop.run(eval_genomes, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Save the winner.
    with open('winner-feedforward', 'wb') as f:
        pickle.dump(winner, f)


    # visualize the evolution
    visualize.plot_stats(stats, ylog=True, view=True, filename="feedforward-fitness.svg")
    visualize.plot_species(stats, view=True, filename="feedforward-speciation.svg")

    node_names = {-1: 'x', -2: 'dx', -3: 'theta', -4: 'dtheta', 0: 'control'}
    visualize.draw_net(config, winner, True, node_names=node_names)

    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-feedforward.gv")
    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-feedforward-enabled.gv", show_disabled=False)
    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-feedforward-enabled-pruned.gv", show_disabled=False, prune_unused=True)


if __name__ == '__main__':
    run()
