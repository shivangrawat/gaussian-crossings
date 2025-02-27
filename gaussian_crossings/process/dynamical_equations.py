import torch 
import math
import numpy as np

class OU_noise_dynamical:
    def __init__(self, dt, tau, sigma, x0):
        self.dt = dt
        self.tau = tau
        self.sigma = sigma
        self.x0 = x0
        self.x = self.x0

    def step(self):
        self.x = self.x + (-self.x/self.tau)*self.dt + self.sigma*math.sqrt(2*self.dt)*np.random.normal()
        return self.x

    def reset(self):
        self.x = self.x0