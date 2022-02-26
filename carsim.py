#!/usr/bin/python3
# AI Car Simulator
# Code is heavily inspired by CheesyAI and NeuralNine
# https://github.com/NeuralNine/ai-car-simulation/
import argparse
import sys, time
import neat, pygame
from car import AICar

DEFAULT_CONFIGPATH = './config.ini'
DEFAULT_TRACK = './images/track.png'


class AICarSimulator:
    def __init__(self, opts):
        self.opts = opts            # Command line options
        self.width = 1920           # Track width
        self.height = 1080          # Track height
        self.fps = 60               # Frames per second to run
        self.screen = None          # Pygame screen for rendering
        self.track = None           # Current race track
        self.font = None            # Font to display metrics
        self.clock = pygame.time.Clock()
        # AI metrics
        self.cars = []              # List of current cars
        self.networks = []          # List of current neural networks
        self.generation = 0         # Current AI generation

    def run(self):
        pygame.init()
        pygame.event.set_allowed([pygame.QUIT])
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.DOUBLEBUF, 8)
        self.font = pygame.font.Font('fonts/Roboto-Regular.ttf', 20)
        self.track = pygame.image.load(opts.track).convert()
        # Create population, add reporters, and run simulation
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
            neat.DefaultSpeciesSet, neat.DefaultStagnation, self.opts.configpath)
        population = neat.Population(config)
        population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)
        population.run(self.simulation, opts.maxgenerations)
    
    def simulation(self, genomes, config):
        """ Callback from NEAT.run() for each generation. """
        # For all genomes passed, create A new neural network
        self.networks = []
        self.cars = []
        for i, genome in genomes:
            network = neat.nn.FeedForwardNetwork.create(genome, config)
            self.networks.append(network)
            self.cars.append(AICar(self.opts))
            genome.fitness = 0
        self.generation += 1
        starttime = time.time()
        while True:  # Run for roughly 20s
            # Exit on quit event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit(0)
            # Calculate action each car should take
            for i, car in enumerate(self.cars):
                output = self.networks[i].activate(car.get_data())
                choice = output.index(max(output))
                if choice == 0:  # Turn left
                    car.angle += 5
                elif choice == 1:  # Turn right
                    car.angle -= 5
                elif choice == 2:  # Slow down
                    car.speed = max(car.speed-5, 10)
                else:  # Speed Up
                    car.speed += 5
            # Update car position and get new rewards
            for i, car in enumerate(self.cars):
                if car.alive:
                    car.update(self.track)
                    genomes[i][1].fitness += car.get_reward()
            # Check we want to start a new generation
            numalive = sum(1 for car in self.cars if car.alive)
            runtime = time.time() - starttime
            if numalive == 0 or runtime > 20:
                break
            self.draw(numalive)
            
    def draw(self, numalive=0):
        """ Redraw the Pygame screen. """
        self.screen.blit(self.track, (0,0))
        for car in self.cars:
            car.draw(self.screen)
        lines = ['PKs AI Car Simulation']
        lines.append(f'Generation: {self.generation}')
        lines.append(f'Cars Alive: {numalive}')
        lines.append(f'FPS: {int(self.clock.get_fps())}')
        textpos = [20, 20]
        for line in lines:
            text = self.font.render(line, True, (20,20,20))
            textrect = text.get_rect()
            textrect.topleft = textpos
            self.screen.blit(text, textrect)
            textpos[1] += 22
        pygame.display.flip()
        self.clock.tick(self.fps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AI Car Simulation')
    parser.add_argument('-c', '--configpath', default=DEFAULT_CONFIGPATH, help='Neat configuration file to use.')
    parser.add_argument('-t', '--track', default=DEFAULT_TRACK, help='Path to track image to use.')
    parser.add_argument('-n', '--maxgenerations', default=20, help='Maximum generations to iterate.')
    parser.add_argument('--drawradar', default=False, action='store_true', help='Draw the car radars.')
    parser.add_argument('--drawreward', default=False, action='store_true', help='Draw car rewards.')
    opts = parser.parse_args()
    carsim = AICarSimulator(opts)
    carsim.run()
