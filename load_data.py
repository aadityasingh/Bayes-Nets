import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import argparse

from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import SubsetRandomSampler
from tqdm import tqdm, trange

def load_data(opts):
    """Creates training, val, and test data loaders.
    """

    if opts.normalization == 'none':
        normalization = transforms.ToTensor()
    elif opts.normalization == 'weird':
        normalization = transforms.compose([transforms.ToTensor(), transforms.Normalize((0,), (126/255,))])
    elif opts.normalization == 'normal':
        normalization = transforms.compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    else:
        raise NotImplementedError

    # load the dataset
    train_dataset = datasets.MNIST('/'.join([opts.base_path, 'mnist']), train=True, download=True, transform=normalization) if opts.dataset == 'mnist' else datasets.FashionMNIST('/'.join([opts.base_path, 'fmnist']), train=True, download=True, transform=transforms.ToTensor())

    valid_dataset = datasets.MNIST('/'.join([opts.base_path, 'mnist']), train=True, download=True, transform=normalization) if opts.dataset == 'mnist' else datasets.FashionMNIST(root='/'.join([opts.base_path, 'fmnist']), train=True, download=True, transform=transforms.ToTensor())

    test_dataset = datasets.MNIST('/'.join([opts.base_path, 'mnist']), train=False, download=True, transform=normalization) if opts.dataset == 'mnist' else datasets.FashionMNIST('/'.join([opts.base_path, 'mnist']), train=False, download=True, transform=transforms.ToTensor())

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = opts.val_size

    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opts.batch_size, sampler=train_sampler, **loader_kwargs)

    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=opts.batch_size, sampler=valid_sampler, **loader_kwargs)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opts.test_batch_size, shuffle=False, **loader_kwargs)

    return train_loader, valid_loader, test_loader