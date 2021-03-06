import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm, trange

import utils

class BayesianTrainer:
    def __init__(self, net, optimizer, train_loader, test_loader, opts):
        self.net = net
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.summary_dir = '/'.join([opts.base_path, 'runs', opts.run, 'logs'])
        self.checkpoint_dir = '/'.join([opts.base_path, 'runs', opts.run, 'checkpoints'])
        utils.create_dir(self.summary_dir)
        utils.create_dir(self.checkpoint_dir)

        self.start_epoch = opts.start_epoch
        self.num_epochs = opts.epochs
        self.lr = opts.lr

        self.test_every = opts.test_every
        self.chkpt_every = opts.chkpt_every
        self.samples = opts.samples
        self.test_samples = opts.test_samples
        self.batch_size = opts.batch_size
        self.test_batch_size = opts.test_batch_size

        self.test_size = opts.val_size if opts.test_on == 'val' else len(self.test_loader.dataset)

        self.reweight = opts.kl_reweight

        self.cuda = opts.cuda

        self.writer = SummaryWriter(log_dir=self.summary_dir)

    def write_weight_histograms(self, epoch):
        self.writer.add_histogram('histogram/w1_mu', self.net.l1.weight_mu,epoch)
        self.writer.add_histogram('histogram/w1_rho', self.net.l1.weight_rho,epoch)
        self.writer.add_histogram('histogram/w2_mu', self.net.l2.weight_mu,epoch)
        self.writer.add_histogram('histogram/w2_rho', self.net.l2.weight_rho,epoch)
        self.writer.add_histogram('histogram/w3_mu', self.net.l3.weight_mu,epoch)
        self.writer.add_histogram('histogram/w3_rho', self.net.l3.weight_rho,epoch)
        self.writer.add_histogram('histogram/b1_mu', self.net.l1.bias_mu,epoch)
        self.writer.add_histogram('histogram/b1_rho', self.net.l1.bias_rho,epoch)
        self.writer.add_histogram('histogram/b2_mu', self.net.l2.bias_mu,epoch)
        self.writer.add_histogram('histogram/b2_rho', self.net.l2.bias_rho,epoch)
        self.writer.add_histogram('histogram/b3_mu', self.net.l3.bias_mu,epoch)
        self.writer.add_histogram('histogram/b3_rho', self.net.l3.bias_rho,epoch)

        self.writer.add_histogram('histogram/weights', torch.cat((self.net.l1.weight.sample().flatten(), self.net.l2.weight.sample().flatten(), self.net.l3.weight.sample().flatten()), 0), epoch)

    def write_loss_scalars(self, epoch, loss, log_prior, log_vp, negative_log_likelihood):
        self.writer.add_scalar('training/loss', loss, epoch)
        self.writer.add_scalar('training/complexity_cost', log_vp-log_prior, epoch)
        self.writer.add_scalar('training/log_prior', log_prior, epoch)
        self.writer.add_scalar('training/log_vp', log_vp, epoch)
        self.writer.add_scalar('training/negative_log_likelihood', negative_log_likelihood, epoch)

    def train(self):
        self.net.train()
        best_error = 1
        for epoch in range(self.start_epoch, self.start_epoch+self.num_epochs):
            if epoch == 0: # write initial distributions
                self.write_weight_histograms(epoch)
                self.test_ensemble(epoch)
            avg_loss = 0
            avg_lp = 0
            avg_lvp = 0
            avg_nll = 0
            for batch_idx, (data, target) in enumerate(tqdm(self.train_loader)):
                if self.cuda:
                    data, target = data.cuda(), target.cuda()
                self.net.zero_grad()
                weight = 1/len(self.train_loader)
                if self.reweight:
                    weight = (2**(len(self.train_loader)-batch_idx-1))/(2**len(self.train_loader)-1)
                loss, log_prior, log_vp, negative_log_likelihood = self.net.sample_elbo(data, target, weight, self.samples)
                avg_loss += loss
                avg_lp += log_prior
                avg_lvp += log_vp
                avg_nll += negative_log_likelihood
                loss.backward()
                self.optimizer.step()
            avg_loss /= self.batch_size
            avg_lp /= self.batch_size
            avg_lvp /= self.batch_size
            avg_nll /= self.batch_size
            self.write_loss_scalars(epoch, avg_loss, avg_lp, avg_lvp, avg_nll)
            self.write_weight_histograms(epoch+1)
            if epoch % self.test_every == 0 and epoch != 0:
                new_error = self.test_ensemble(epoch)
                if new_error < best_error:
            #if epoch % self.chkpt_every == 0:
                    best_error = new_error
                    print("saving checkpoint...")
                    torch.save({
                            'epoch': epoch + 1,
                            'state_dict': self.net.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                        }, '/'.join([self.checkpoint_dir, 'checkpoint_best.pth.tar']))
        print("saving checkpoint...")
        torch.save({
                'epoch': epoch + 1,
                'state_dict': self.net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, '/'.join([self.checkpoint_dir, 'checkpoint'+str(self.start_epoch+self.num_epochs)+'.pth.tar']))

    def test_ensemble(self, epoch):
        self.net.eval()
        print("Testing...", epoch)
        correct = 0
        corrects = np.zeros(self.test_samples+1, dtype=int)
        with torch.no_grad():
            for data, target in tqdm(self.test_loader):
                outputs = torch.zeros(self.test_samples+1, self.test_batch_size, 10)
                if self.cuda:
                    data, target, outputs = data.cuda(), target.cuda(), outputs.cuda()
                for i in range(self.test_samples):
                    outputs[i] = self.net(data, sample=True)
                outputs[self.test_samples] = self.net(data, sample=False)
                output = outputs.mean(0)
                preds = outputs.max(2, keepdim=True)[1]
                pred = output.max(1, keepdim=True)[1] # index of max log-probability
                corrects += preds.eq(target.view_as(pred)).sum(dim=1).squeeze().cpu().numpy()
                correct += pred.eq(target.view_as(pred)).sum().item()
        for i, num in enumerate(corrects):
            if i < self.test_samples:
                print('Component {} Accuracy: {}/{}'.format(i, num, self.test_size))
            else:
                print('Posterior Mean Accuracy: {}/{}'.format(num, self.test_size))
        print('Ensemble Accuracy: {}/{}'.format(correct, self.test_size))
        self.writer.add_scalar('test/error', 1-correct/self.test_size, epoch)
        self.net.train()
        return 1-correct/self.test_size
