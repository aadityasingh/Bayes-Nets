import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Gaussian(object):
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0,1)
    
    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))
    
    def sample(self):
        epsilon = self.normal.sample(self.rho.size())
        if torch.cuda.is_available():
            epsilon = epsilon.cuda()
        return self.mu + self.sigma * epsilon

    def pruned_sample(self, noise_threshold):
        epsilon = self.normal.sample(self.rho.size())
        if torch.cuda.is_available():
            epsilon = epsilon.cuda()
        return torch.where(torch.abs(self.mu)/self.sigma <= noise_threshold, torch.zeros(self.mu.shape), self.mu + self.sigma * epsilon)
    
    def naive_pruned_sample(self, weight_threshold):
        epsilon = self.normal.sample(self.rho.size())
        if torch.cuda.is_available():
            epsilon = epsilon.cuda()
        return torch.where(torch.abs(self.mu) <= weight_threshold, torch.zeros(self.mu.shape), self.mu + self.sigma * epsilon)

    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2))


class ScaleMixtureGaussian(object):
    def __init__(self, pi, sigma1, sigma2):
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(0,sigma1)
        self.gaussian2 = torch.distributions.Normal(0,sigma2)
    
    def log_prob(self, input):
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.pi * prob1 + (1-self.pi) * prob2))