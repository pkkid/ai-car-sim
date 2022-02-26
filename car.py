#!/usr/bin/python3
# AI Car Simulator - Single Car
# Code is heavily inspired by CheesyAI and NeuralNine
# https://github.com/NeuralNine/ai-car-simulation/
import math, pygame


class AICar:
    RADAR_COLOR = (180,30,30)
    IMAGEPATH = './images/car.png'

    def __init__(self, opts):
        self.opts = opts                    # Command line options
        self.carlength = 60                 # Car width (ignoring image size)
        self.carwidth = 30                  # Car height (ignoring image size)
        self.alive = True                   # Set False when car has crashed
        self.position = [860, 910]          # Default starting position
        self.corners = []                   # Position of 4 corners of the car
        self.angle = 0                      # Current car angle. 0=Facing Right
        self.speed = 0                      # Current car speed (0-x)
        self.radars = []                    # List For Sensors / Radars
        self.distance = 0                   # Distance driven
        self.font = pygame.font.Font('fonts/Roboto-Regular.ttf', 14)
        # PyGame Sprite
        self.car = pygame.image.load(self.IMAGEPATH).convert_alpha()
        self.car = pygame.transform.scale(self.car, (self.carlength, self.carwidth))
        self.center = None                  # Center position of the car
        self.rotated = None                 # Rotated car sprite
        self.rect = None                    # Rotated car sprite rectangle
        self.rotate_car()

    def check_collision(self, track):
        self.alive = True
        bordercolor = track.get_at((0,0))
        for x,y in self.corners:
            # If any corner touches a border, we crashed
            if track.get_at((int(x), int(y))) == bordercolor:
                self.alive = False
                break

    def check_radar(self, angle, track):
        """ Check a single car radar. """
        length = 0
        bordercolor = track.get_at((0,0))
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + angle))) * length)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + angle))) * length)
        # While we don't hit BORDER_COLOR and length < 300 go further and further
        while not track.get_at((x,y)) == bordercolor and length < 300:
            length = length + 5
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + angle))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + angle))) * length)
        # Calculate distance to border and append to radars list
        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([(x,y), dist])
    
    def get_data(self):
        """ Return data to send the neural network. """
        return [int(radar[1])/30 for radar in self.radars] or [0]*5

    def get_reward(self):
        """ Calculate the reward. """
        return self.distance / (self.carlength / 2)

    def draw(self, screen):
        """ Draw this car on the specified screen. """
        if self.alive:
            self.draw_radar(screen)
            screen.blit(self.rotated, self.rect)
            self.draw_reward(screen)
            
    def draw_radar(self, screen):
        """ Draw car radars. """
        if not self.opts.drawradar:
            return
        for radar in self.radars:
            position = radar[0]
            pygame.draw.line(screen, self.RADAR_COLOR, self.center, position, 1)
            pygame.draw.circle(screen, self.RADAR_COLOR, position, 2)
            for corner in self.corners:
                pygame.draw.circle(screen, (150,150,230), corner, 1)
    
    def draw_reward(self, screen):
        """ Draw car reward. """
        if not self.opts.drawreward:
            return
        text = self.font.render(str(int(self.get_reward())), True, (230,230,230))
        textrect = text.get_rect()
        textrect.topleft = self.corners[1]
        screen.blit(text, textrect)
    
    def rotate_car(self):
        """ Render the rotated car and update self.center. """
        self.rotated = pygame.transform.rotozoom(self.car, self.angle, 1)
        self.center = self.car.get_rect(topleft=self.position).center
        self.rect = self.rotated.get_rect(center=self.center)

    def update(self, track):
        """ Update car position. """
        self.speed = 5
        self.distance += self.speed
        self.rotate_car()
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        # Calculate the four corners of of the car and check we crashed
        frontleft = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * self.carwidth, self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * self.carwidth]  # noqa
        frontright = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * self.carwidth, self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * self.carwidth]  # noqa
        backleft = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * self.carwidth, self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * self.carwidth]  # noqa
        backright = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * self.carwidth, self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * self.carwidth]  # noqa
        self.corners = [frontleft, backleft, backright, frontright]
        self.check_collision(track)
        # Check five car radar sensors
        self.radars.clear()
        for angle in range(-90, 120, 45):
            self.check_radar(angle, track)
