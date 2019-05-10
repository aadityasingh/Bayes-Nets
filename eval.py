# This file is used for reading in models and testing them
# Testing includes: error on test set, weight histogram saving, weight signal saving, and weight pruning results

import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import argparse
import os

from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm, trange

import distributions
import models
from train import BayesianTrainer
from load_data import load_data

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', dest='cuda', type=int, default=0, help="Whether or not cuda is available")

    parser.add_argument('--run_folder', default='600epochruns')
    # Run folder should have following structure
    # subfolders corresponding to different latent dimensions
    # each subfolder has subfolders for different normalizatoin schemes
    # each of those normalization scheme subfolders has runs using models of that latent dimension+norm scheme
    parser.add_argument('--base_path', default='.')
    parser.add_argument('--load_from_chkpt', default='checkpoint_best.pth.tar', help="Which checkpoint to load from")

    parser.add_argument('--dataset', type=str, default='mnist', help="Choose between 'fmnist' and 'mnist'")
    parser.add_argument('--test_on', type=str, default='test', help="Choose between 'test' (default) and 'val'") #not used
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=100, help="Training batch size") #not used
    parser.add_argument('--test_batch_size', dest='test_batch_size', type=int, default=5, help="Test batch size")
    parser.add_argument('--val_size', type=int, default=10000, help="Size of val set") #not used

    parser.add_argument('--prune', dest='prune', type=float, nargs='*', default=[0.25, 0.5, 0.75, 0.95])
    parser.add_argument('--colors', dest='colors', type=str, nargs='*', default=['m', 'r', 'g', 'b'])
    parser.add_argument('--naive', type=int, default=0, help="Wehter or not to apply naive pruning")

    parser.add_argument('--test_samples', dest='test_samples', type=int, default=10, help="How many samples to use for each test forward pass")

    parser.add_argument('--normalization', default='none', help="Gets overwritten, ignore")


    return parser

def test_ensemble(net, test_loader, opts, pruned=False, threshold=0, threshold_bias=False, naive=False, verbose=False):
    correct = 0
    corrects = np.zeros(opts.test_samples+1, dtype=int)
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            outputs = torch.zeros(opts.test_samples+1, opts.test_batch_size, 10)
            if opts.cuda:
                data, target, outputs = data.cuda(), target.cuda(), outputs.cuda()
            for i in range(opts.test_samples):
                if pruned:
                    outputs[i] = net.forward_pruned(data, threshold, sample=True, threshold_bias=threshold_bias, naive=naive)
                else:
                    outputs[i] = net(data, sample=True)
            if pruned:
                outputs[opts.test_samples] = net.forward_pruned(data, threshold, sample=False, threshold_bias=threshold_bias, naive=naive)
            else:
                outputs[opts.test_samples] = net(data, sample=False)
            output = outputs.mean(0)
            preds = outputs.max(2, keepdim=True)[1]
            pred = output.max(1, keepdim=True)[1] # index of max log-probability
            corrects += preds.eq(target.view_as(pred)).sum(dim=1).squeeze().cpu().numpy()
            correct += pred.eq(target.view_as(pred)).sum().item()
    for i, num in enumerate(corrects):
        if i < opts.test_samples:
            if verbose:
                #print('Component {} Accuracy: {}/{}'.format(i, num, len(test_loader.dataset)))
                print('Component ', i, ' error: ', 1-num/len(test_loader.dataset))
        else:
            #print('Posterior Mean Accuracy: {}/{}'.format(num, len(test_loader.dataset)))
            print('Posterior mean error: ', 1-num/len(test_loader.dataset))
    #print('Ensemble Accuracy: {}/{}'.format(correct, len(test_loader.dataset)))
    print('Ensemble error: ', 1-correct/len(test_loader.dataset))
    return 1-correct/len(test_loader.dataset)

def save_weight_hist(net, values, thresholds, colors, root, naive=False):
    fig1, axs = plt.subplots(len(thresholds), 1, sharex=True)
    for i, thresh in enumerate(thresholds):
        weight_sample = np.zeros(shape=(0, ))
        for layer in list(net.modules())[1:]:
            if naive:
                weight_sample = np.hstack((weight_sample, layer.weight.naive_pruned_sample(values[int(len(values)*thresh)]).detach().numpy().flatten()))
            else:
                weight_sample = np.hstack((weight_sample, layer.weight.pruned_sample(values[int(len(values)*thresh)]).detach().numpy().flatten()))
        axs[i].hist(weight_sample[weight_sample != 0], bins=np.linspace(-0.3, 0.3, 100), color=colors[i])
    if naive:
        fig1.savefig(root + "/pruned_weights_naive.png")
    else:
        fig1.savefig(root + "/pruned_weights.png")

def save_noise_hist(net, root, opts):
    noises = np.zeros(shape=(0, ))
    for layer in list(net.modules())[1:]:
        noises = np.hstack((noises, (torch.abs(layer.weight.mu)/layer.weight.sigma).detach().numpy().flatten()))
    noises.sort()
    fig1, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.hist(np.log(noises), bins=np.arange(math.log(noises[0]), math.log(noises[-1]), 0.1), color='k')
    for i, pct in enumerate(opts.prune):
        ax1.axvline(np.log(noises)[int(len(noises)*pct)], color=opts.colors[i])
    ax2.plot(np.log(noises), np.linspace(0, 1, len(noises)))
    fig1.savefig(root+'/noises.png')
    plt.close(fig1)
    return noises

def ordered_posterior_means(net):
    weight_means = np.zeros(shape=(0, ))
    for layer in list(net.modules())[1:]:
        weight_means = np.hstack((weight_means, (torch.abs(layer.weight.mu)).detach().numpy().flatten()))
    weight_means.sort()
    return weight_means

def save_weight_mu_hist(net, root):
    fig1, ax1 = plt.subplots()
    weight_means = np.zeros(shape=(0, ))
    for layer in list(net.modules())[1:]:
        weight_means = np.hstack((weight_means, layer.weight.mu.detach().numpy().flatten()))
    ax1.hist(weight_means, bins=np.linspace(-0.3, 0.3, 100), color='k')
    fig1.savefig(root+'/weight_means.png')
    plt.close(fig1)

if __name__ == "__main__":
    parser = create_parser()
    opts = parser.parse_args()

    opts.cuda = 1 if torch.cuda.is_available() else 0
    print("Is there a gpu???")
    print(opts.cuda)

    for ldstr in os.listdir('/'.join([opts.base_path, opts.run_folder])):
        try:
            latent_dim = int(ldstr)
        except:
            continue

        net = models.BayesianNetwork(latent_dim=latent_dim, prior=torch.distributions.Normal(0,1))
        # Note during test time we don't care about the prior, so we just put in a N(0,1) because we need to pass in something

        if opts.cuda:
            net.cuda()

        for norm_type in os.listdir('/'.join([opts.base_path, opts.run_folder, ldstr])):
            if norm_type == 'none' or norm_type == 'weird' or norm_type == 'normal':
                opts.normalization = norm_type
                _, _, test_loader = load_data(opts)
                assert (len(test_loader.dataset) % opts.test_batch_size) == 0

                for run in os.listdir('/'.join([opts.base_path, opts.run_folder, ldstr, norm_type])):
                    if run == '.DS_Store':
                        continue # annoying mac stuff
                    print("Latent dim", ldstr, "Norm type", norm_type, "run", run)
                    if opts.cuda:
                        checkpoint = torch.load('/'.join([opts.base_path, opts.run_folder, ldstr, norm_type, run,'checkpoints',opts.load_from_chkpt]))
                    else:
                        checkpoint = torch.load('/'.join([opts.base_path, opts.run_folder, ldstr, norm_type, run,'checkpoints',opts.load_from_chkpt]), map_location="cpu")
                    net.load_state_dict(checkpoint['state_dict'])
                    net.eval()
                    root = '/'.join([opts.base_path, opts.run_folder, ldstr, norm_type, run])
                    
                    save_weight_mu_hist(net, root)
                    continue
                    test_ensemble(net, test_loader, opts)
                    if opts.naive:
                        weights = ordered_posterior_means(net)
                        save_weight_hist(net, weights, [0]+opts.prune, ['k']+opts.colors, root, naive=True)
                        for thresh in opts.prune:
                            print("Pruning at ", thresh)
                            test_ensemble(net, test_loader, opts, pruned=True, threshold=weights[int(len(weights)*thresh)], threshold_bias=False, naive=True)
                    else:
                        noises = save_noise_hist(net, root, opts)
                        save_weight_hist(net, noises, [0]+opts.prune, ['k']+opts.colors, root)
                        for thresh in opts.prune:
                            print("Pruning at ", thresh)
                            test_ensemble(net, test_loader, opts, pruned=True, threshold=noises[int(len(noises)*thresh)], threshold_bias=False)
            else:
                continue

    # weight_sample = np.zeros(shape=(0, ))
    # weight_sample2 = np.zeros(shape=(0, ))

    # print(weight_sample.shape)
    # weight_sample2 = torch.cat((net.l1.weight.sample().flatten(), net.l2.weight.sample().flatten(), net.l3.weight.sample().flatten()), 0).detach().numpy()
    # for layer in list(net.modules())[1:]:
    #     weight_sample = np.hstack((weight_sample, layer.weight.sample().detach().numpy().flatten()))
    #     print(weight_sample.shape)

    # plt.figure(1)
    # plt.hist(weight_sample, bins=np.linspace(-0.3, 0.3, 100), color='k')
    # plt.figure(2)
    # plt.hist(weight_sample, bins=np.linspace(-0.604, 0.722, 30), color='k')
    # plt.figure(3)
    # plt.hist(weight_sample2, bins=np.linspace(-0.3, 0.3, 100), color='k')
    # plt.figure(4)
    # plt.hist(weight_sample2, bins=np.linspace(-0.604, 0.722, 30), color='k')
    # plt.show()
    






