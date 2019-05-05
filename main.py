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
from tqdm import tqdm, trange

import distributions
import models
from train import BayesianTrainer
from load_data import load_data

def create_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', default='train', help="Mode to use. Currently supports 'train' and 'test'")

	parser.add_argument('--cuda', dest='cuda', type=int, default=0, help="Whether or not cuda is available")

	parser.add_argument('--run', default='run')
	parser.add_argument('--base_path', default='/Bayes-Nets')
	parser.add_argument('--load_from_chkpt', default=None, help="Which checkpoint to load from")

	parser.add_argument('--dataset', type=str, default='mnist', help="Choose between 'fmnist' and 'mnist'")
	parser.add_argument('--test_on', type=str, default='test', help="Choose between 'test' (default) and 'val'")
	parser.add_argument('--batch_size', dest='batch_size', type=int, default=100, help="Training batch size")
	parser.add_argument('--test_batch_size', dest='test_batch_size', type=int, default=5, help="Test batch size")
	parser.add_argument('--val_size', type=int, default=10000, help="Size of val set")

	parser.add_argument('--epochs', type=int, default=600, help="Number of training epochs")
	parser.add_argument('--optimizer', type=str, default='sgd', help="Optimizer to use. Currently supports 'adam' and 'sgd' (default)")
	parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate, only used with SGD")
	parser.add_argument('--test_every', type=int, default=10, help="How often to test the model during training")
	parser.add_argument('--chkpt_every', type=int, default=50, help="How often to save a checkpoint")

	parser.add_argument('--latent_dim', type=int, default=400, help="Number of hidden neurons in layers")
	parser.add_argument('--samples', dest='samples', type=int, default=2, help="How many samples to use for each training iteration")
	parser.add_argument('--test_samples', dest='test_samples', type=int, default=10, help="How many samples to use for each test forward pass")
	
	parser.add_argument('--normalization', default='none', help="Type of normalization to use (for MNIST only), choose from 'none', 'weird' (dividing by 126, used in paper), 'normal' (canonical mnist normalization)")
	parser.add_argument('--kl_reweight', type=int, default=1, help="Whether or not to use the KL reweighting (section 3.4)")
	parser.add_argument('--use_scale_prior', type=int, default=1, help="Whether or not to use the Scale Mixture prior")
	parser.add_argument('--prior_pi', type=float, default=0.5, help="Weight parameter for Scale Mixture prior, must be between 0 and 1")
	parser.add_argument('--prior_sigma1', type=float, default=1, help="Std. Dev. of gaussian prior or first component of Scale Mixture prior")
	parser.add_argument('--prior_sigma2', type=float, default=math.exp(-6), help="Std. Dev. of second component of Scale Mixture prior")

	return parser

if __name__ == "__main__":
	parser = create_parser()
	opts = parser.parse_args()

	opts.cuda = 1 if torch.cuda.is_available() else 0
	print("Is there a gpu???")
	print(opts.cuda)


	train_loader, val_loader, test_loader = load_data(opts)

	if opts.test_on == 'val':
		test_loader = val_loader

	assert (len(train_loader.dataset) % opts.batch_size) == 0
	assert (len(test_loader.dataset) % opts.test_batch_size) == 0

	if opts.use_scale_prior:
		net = models.BayesianNetwork(latent_dim=opts.latent_dim, prior=distributions.ScaleMixtureGaussian(opts.prior_pi, opts.prior_sigma1, opts.prior_sigma2))
	else:
		print("using gaussian")
		net = models.BayesianNetwork(latent_dim=opts.latent_dim, prior=distributions.Gaussian(0, opts.prior_sigma1))

	if opts.cuda:
		net.cuda()

	optimizer = optim.SGD(net.parameters(), lr=opts.lr) if opts.optimizer == 'sgd' else optim.Adam(net.parameters())

	if opts.load_from_chkpt != None:
		checkpoint = torch.load('/'.join([opts.base_path,'runs',opts.run,'checkpoints',opts.load_from_chkpt]))
		net.load_state_dict(checkpoint['state_dict'])
		opts.start_epoch = checkpoint['epoch']
		print("Start epoch", opts.start_epoch)
		optimizer.load_state_dict(checkpoint['optimizer'])

	if opts.mode == 'train':
		BayesianTrainer(net, optimizer, train_loader, test_loader, opts).train()

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
	






