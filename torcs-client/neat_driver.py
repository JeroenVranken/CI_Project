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

import logging

import math

from pytocl.analysis import DataLogWriter
from pytocl.car import State, Command, MPS_PER_KMH
from pytocl.controller import CompositeController, ProportionalController, \
    IntegrationController, DerivativeController
from networks import JNetV1, JNetV2, simpleNetV2

import neat
import evolve
from pytocl.main import main
import time
import visualize

_logger = logging.getLogger(__name__)

D_in = 25
D_out = 3


class NeatDriver:
    """
    Driving logic.

    Implement the driving intelligence in this class by processing the current
    car state as inputs creating car control commands as a response. The
    ``drive`` function is called periodically every 20ms and must return a
    command within 10ms wall time.
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
        self.eta = 20
        self.counter = 0
        self.model = simpleNetV2()
        self.weights = 'simpleNetV2_epoch_3_all_tracks.csv.pkl'

        self.model.load_state_dict(torch.load(self.weights))
        self.input = torch.FloatTensor(D_in)
        self.track_check1 = False
        self.track_check2 = False
        self.action_neat = None
        self.net = net
        self.clock = time.time()
        self.done = False
        


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

    def eval_genomes2(self, pos, p, dp, d, s):
        fitness = self.eta - p - d + s + dp
        print('p is', p)
        print('d is, ', d)
        print('s is ', s)
        print('fitness is, ', fitness)
        print('dp is, ', dp)

    def make_input(self, carstate, out):
        inputs = []
        inputs.extend([carstate.angle, carstate.speed_x, carstate.speed_y, carstate.speed_z, carstate.distance_from_center, carstate.z])
        inputs.extend(carstate.opponents)
        inputs.extend(carstate.distances_from_edge)
        inputs.extend(out)
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

    def drive(self, carstate: State) -> Command:
        """
        Produces driving command in response to newly received car state.

        This is a dummy driving routine, very dumb and not really considering a
        lot of inputs. But it will get the car (if not disturbed by other
        drivers) successfully driven along the race track."""

        global net
        out = self.model(Variable(self.input))
        
        self.input = self.update_state(out, carstate)

        inputs = self.make_input(carstate, out.data)           

        if self.track_check1 == True and self.check_offtrack(carstate) == True:
            self.track_check2 = True

        if self.counter % 100 == 0:
            self.track_check1 = self.check_offtrack(carstate)

        self.counter += 1


        outputs = net.activate(inputs)


        command = Command()
        command.accelerator = outputs[0]
        command.brake = outputs[1]
        command.steering = outputs[2]
        v_x = 180

        self.accelerate(carstate, v_x, command)

        if self.data_logger:
            self.data_logger.log(carstate, command)
        
        if carstate.damage >= 9500 or carstate.last_lap_time > 0 or carstate.current_lap_time >  60:
            positions_won = 28 - carstate.race_position 
            damage = (carstate.damage) / 1000
            distance = carstate.distance_raced/100
            global fitness
            fitness = positions_won - damage + distance
            command.meta = 1

        return command


    def check_offtrack(self, carstate):
        counter = 0
        for values in carstate.distances_from_edge:
            if values == -1:
                counter += 1
        if counter == len(carstate.distances_from_edge):
            return True
        else: 
            return False


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

net = None
fitness = 0

# Use the NN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
    
    global net
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    driver = main(NeatDriver(logdata=False, net=net))
    global fitness
    print("THE FITNESS IS: ", fitness)
    return fitness
    
    
    # print('carstate', inputs)
    # self.action_neat = net.activate(inputs)

    # # Apply action to the simulated driver
    # force = neat_driver.command
    # sim.step(force)

    # fitness = sim.t

    # fitnesses.append(fitness)

    # # The genome's fitness is its worst performance across all runs.
    # return min(fitnesses)


def eval_genomes(genomes, config):
    counter = 1 
    for genome_id, genome in genomes:
        print("ID number of genome: ", counter)
        genome.fitness = eval_genome(genome, config)
        counter += 1

def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-neat')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(1, eval_genome)
    winner = pop.run(pe.evaluate)

   # Save the winner.
    with open('winner-feedforward', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)
    
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

# # Display the winning genome.
# print('\nBest genome:\n{!s}'.format(winner))

# # Display the winning genome.
# print('\nBest genome:\n{!s}'.format(winner))

# # Show output of the most fit genome against training data.
# print('\nOutput:')
# winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
# for xi, xo in zip(xor_inputs, xor_outputs):
#     output = winner_net.activate(xi)
#     print("  input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))