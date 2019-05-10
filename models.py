import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from distributions import Gaussian, ScaleMixtureGaussian

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5,-4))
        self.weight = Gaussian(self.weight_mu, self.weight_rho)

        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5,-4))
        self.bias = Gaussian(self.bias_mu, self.bias_rho)

        self.weight_prior = prior
        self.bias_prior = prior
        self.log_prior = 0
        self.log_vp = 0

    def forward(self, input, sample=False, calculate_log_probs=False):
        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training or calculate_log_probs:
            self.log_prior = self.weight_prior.log_prob(weight).sum() + self.bias_prior.log_prob(bias).sum()
            self.log_vp = self.weight.log_prob(weight).sum() + self.bias.log_prob(bias).sum()
        else:
            self.log_prior, self.log_vp = 0, 0

        return F.linear(input, weight, bias)

    # This method should only ever be used at test, as the log probs don't make sense after pruning
    # If naive is true, we prune naively based on posterior weight means, so the threshold should be a weight threshold
    # If naive is false, we prune based on noise, so the threshold should be a noise threshold
    def forward_pruned(self, input, threshold, sample=False, threshold_bias=False, naive=False):
        if sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        if naive:
            weight = torch.where(torch.abs(self.weight.mu) <= threshold, torch.zeros(weight.shape), weight)
        else:
            weight = torch.where(torch.abs(self.weight.mu)/self.weight.sigma <= threshold, torch.zeros(weight.shape), weight)

        if threshold_bias:
            if naive:
                bias = torch.where(torch.abs(self.bias.mu) <= threshold, torch.zeros(bias.shape), bias)
            else:
                bias = torch.where(torch.abs(self.bias.mu)/self.bias.sigma <= threshold, torch.zeros(bias.shape), bias)

        return F.linear(input, weight, bias)



class BayesianNetwork(nn.Module):
    def __init__(self, latent_dim=400, prior=ScaleMixtureGaussian(0.5, torch.FloatTensor([math.exp(-0)]), torch.FloatTensor([math.exp(-6)]))):
        super().__init__()
        self.l1 = BayesianLinear(28*28, latent_dim, prior)
        self.l2 = BayesianLinear(latent_dim, latent_dim, prior)
        self.l3 = BayesianLinear(latent_dim, 10, prior)
    
    def forward(self, x, sample=False):
        x = x.view(-1, 28*28)
        x = F.relu(self.l1(x, sample))
        x = F.relu(self.l2(x, sample))
        x = F.log_softmax(self.l3(x, sample), dim=1)
        return x

    def forward_pruned(self, x, noise_threshold, sample=False, threshold_bias=False, naive=False):
        x = x.view(-1, 28*28)
        x = F.relu(self.l1.forward_pruned(x, noise_threshold, sample, threshold_bias, naive))
        x = F.relu(self.l2.forward_pruned(x, noise_threshold, sample, threshold_bias, naive))
        x = F.log_softmax(self.l3.forward_pruned(x, noise_threshold, sample, threshold_bias, naive), dim=1)
        return x
    
    def log_prior(self):
        return self.l1.log_prior + self.l2.log_prior + self.l3.log_prior
    
    def log_vp(self):
        return self.l1.log_vp + self.l2.log_vp + self.l2.log_vp
    
    def sample_elbo(self, input, target, kl_weight, samples=2):
        outputs = torch.zeros(samples, input.shape[0], 10)
        log_priors = torch.zeros(samples)
        log_vps = torch.zeros(samples)
        if torch.cuda.is_available():
            outputs, log_priors, log_vps = outputs.cuda(), log_priors.cuda(), log_vps.cuda()
        for i in range(samples):
            outputs[i] = self(input, sample=True)
            log_priors[i] = self.log_prior()
            log_vps[i] = self.log_vp()
        log_prior = log_priors.mean()
        log_vp = log_vps.mean()
        negative_log_likelihood = F.nll_loss(outputs.mean(0), target, size_average=False)
        loss = (log_vp - log_prior)*kl_weight + negative_log_likelihood
        return loss, log_prior, log_vp, negative_log_likelihood

