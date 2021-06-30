###################################
#           Imports               #
###################################
import cv2
import numpy as np
import math
import random
from motion_model import Motion_Model

###################################
#         Particle Class          #
###################################


class Particle:
    ''' x, y are arrays of 3 values, containing the nth, n-1th and n-2th position'''
    __world_size_x = 800            # World width
    __world_size_y = 800            # World height
    def __init__(self, bound=None, world_size=None):
        if(world_size!= None):
            if(isinstance(world_size, int)):
                Particle.__world_size_x = world_size;
                Particle.__world_size_y = world_size;
            if(isinstance(world_size, tuple)):
                Particle.__world_size_x = world_size[0];
                Particle.__world_size_y = world_size[1];
        if bound:
            self.x = np.array([[random.randint(bound[0][0], bound[1][0]), 0.0, 0.0]])
            self.y = np.array([[random.randint(bound[0][1], bound[1][1]), 0.0, 0.0]])
        else:
            self.x = np.array([[int(random.random()*Particle.__world_size_x), 0.0, 0.0]])
            self.y = np.array([[int(random.random()*Particle.__world_size_y), 0.0, 0.0]])
        self.forward_noise =   2          # variance of the noise
        self.measurement_noise = 4.0    # variance of the mesurement (needed to calcualte the gaussian)

    def set_xy(self, x, y):
        ''' Set the coordinate of x and y for the particle '''
        self.x = np.append(np.array([[x]]), self.x[:, :2], axis=1)
        self.y = np.append(np.array([[y]]), self.y[:, :2], axis=1)

    def set_noise(self, forward = 2, measurement = 4.0):
        ''' Set the forward and measurement noise'''
        self.forward_noise = forward
        self.measurement_noise = measurement
    
    def check_bound(self , x,y):
        ''' Check postion to find of the particle is outside the world boundaries '''
        if x>=Particle.__world_size_x:
            x = Particle.__world_size_x - 1
        elif x<=0:
            x = 1
        if y>=Particle.__world_size_y:
            y = Particle.__world_size_y - 1
        elif y<=0:
            y = 1
        return x,y

    def apply_noise(self, var):
        '''Apply motion noise for randomness '''
        if(var<self.forward_noise):
            var = self.forward_noise
        noise = (int(random.gauss(0.0,var)))
        return noise

    def motion_model(self, parameters):
        ''' Function defining the motion model for predicting the next possible position '''
        x = round(float(parameters['ax']@self.x.T + self.apply_noise(parameters['varx'])))
        y = round(float(parameters['ay']@self.y.T + self.apply_noise(parameters['vary'])))
        x,y = self.check_bound(x,y)
        r = Particle()
        r.set_xy(self.x[0][1],self.y[0][1])
        r.set_xy(self.x[0][0],self.y[0][0])
        r.set_xy(x,y)
        return r

    def Gaussian(self, mu, var, x):
        ''' Gaussian function to get a normalised weight for accuracy '''
        return math.exp(- ((mu - x) ** 2) / var*2.0) / math.sqrt(2.0 * math.pi * var)

    def measurement_prob(self, center):
        ''' Probabilty of whether the particle is near the actual object (coordinate specified by the center) '''
        dist = math.sqrt((center[0]-self.x[0][0])**2 + (center[1]-self.y[0][0])**2)
        prob = self.Gaussian(0.0, self.measurement_noise, dist)
        return prob

    def __str__(self):
        return "[x: %4d y: %4d ]"%(self.x[0][0], self.y[0][0])

    def insert_initial_coord(self, state):
        self.x[0][2] = self.x[0][1]
        self.x[0][1] = state[0]

        self.y[0][2] = self.y[0][1]
        self.y[0][1] = state[1]



###################################
#          Filter Class           #
###################################

class Particle_Filter:
    def __init__(self, bound=None, no_particles=1024, world_size = None, past_record = 10):
        self.motion_model = Motion_Model(past_record)
        self.frame_skips = 0                                                   # Count the number of frames skiped, required to tell the motion model to wipe previous data
        self.num_particle = no_particles
        self.p = []
        for i in range(self.num_particle):
            self.p.append(Particle(bound, world_size))

    def set_noise(self, forward=2, measurement=4.0):
        for i in range(self.num_particle):
            self.p[i].set_noise(forward, measurement)

    def add_state(self, state):
        return_coordinates = self.motion_model.update(state)
        if(return_coordinates == 0):
            for i in range(self.num_particle):
                self.p[i].insert_initial_coord(state)

    def weight(self, center):
        ''' Calculate the weight for all the Particles in filter 
            Also, add, current coordinate to update motion parameters '''
        if(self.frame_skips > 2):
            self.motion_model.clear_data()                                          # if 2 frames are skiped clear previous data
            self.frame_skips = 0
            #print("Clear model history!!!")
        self.frame_skips = 0                                                        # since weight call always gives detector data, frame skip is set to 0
        self.add_state(center)
        w = []
        for i in range(self.num_particle):
            w.append(self.p[i].measurement_prob(center))
        for i in range(self.num_particle):
             w[i] = w[i]/(sum(w)+1e-34)
        return w

    def forward(self):
        ''' Shift the Particle by the specified model or external input '''

        self.frame_skips += 1                                                        # increment to count frame skips
        new = []
        for i in range(self.num_particle):
            new.append(self.p[i].motion_model(self.motion_model.parameters))
        self.p = new

    def resampling(self, w):
        ''' Resample the particles according to weight (w) '''
        new_sample = []
        index = [i for i in range(self.num_particle)]
        new_index = random.choices(index, weights=w, k=self.num_particle)
        for i in new_index:
            new_sample.append(self.p[i])
        self.p = new_sample

    def print_particle(self, num=10):
        ''' Print the first n particles '''
        print("First %4d particles"%(num))
        for i in range(num):
            print(self.p[i])

    def __getitem__(self, index):
        ''' Get the contents of ith particle '''
        return self.p[index]

    def particle_plot(self, frame, size=1, color=(255,0,0)):
        ''' Plot the particles in a given frame '''
        for i in range(self.num_particle):
            cv2.circle(frame, (int(self.p[i].x[0][0]), int(self.p[i].y[0][0])), size, color, 2)
        return frame