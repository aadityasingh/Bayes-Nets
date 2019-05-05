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

        self.num_epochs = opts.epochs
        self.lr = opts.lr

        self.test_every = opts.test_every
        self.chkpt_every = opts.chkpt_every
        self.samples = opts.samples
        self.test_samples = opts.test_samples
        self.batch_size = opts.batch_size
        self.test_batch_size = opts.test_batch_size

        self.device = opts.device

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

    def write_loss_scalars(self, epoch, loss, log_prior, log_variational_posterior, negative_log_likelihood):
        self.writer.add_scalar('training/loss', loss, epoch)
        self.writer.add_scalar('training/complexity_cost', log_variational_posterior-log_prior, epoch)
        self.writer.add_scalar('training/log_prior', log_prior, epoch)
        self.writer.add_scalar('training/log_variational_posterior', log_variational_posterior, epoch)
        self.writer.add_scalar('training/negative_log_likelihood', negative_log_likelihood, epoch)

    def train(self):
        self.net.train()
        for epoch in range(self.num_epochs):
            if epoch == 0: # write initial distributions
                self.write_weight_histograms(epoch)
                self.test_ensemble(epoch)
            avg_loss = 0
            avg_lp = 0
            avg_lvp = 0
            avg_nll = 0
            for batch_idx, (data, target) in enumerate(tqdm(self.train_loader)):
                data, target = data.to(self.device), target.to(self.device)
                self.net.zero_grad()
                loss, log_prior, log_variational_posterior, negative_log_likelihood = self.net.sample_elbo(data, target, 1/len(self.train_loader), self.samples)
                avg_loss += loss
                avg_lp += log_prior
                avg_lvp += log_variational_posterior
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
                self.test_ensemble(epoch)
            #if epoch % self.chkpt_every == 0:

                print("saving checkpoint...")
                torch.save({
                        'epoch': epoch + 1,
                        'state_dict': self.net.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                    }, '/'.join([self.checkpoint_dir, 'checkpoint.pth.tar']))
        print("saving checkpoint...")
        torch.save({
                'epoch': epoch + 1,
                'state_dict': self.net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, '/'.join([self.checkpoint_dir, 'checkpoint'+str(self.num_epochs)+'.pth.tar']))


    def save_checkpoint(self, state, is_best=False, filename='checkpoint.pth.tar'):
        '''
        a function to save checkpoint of the training
        :param state: {'epoch': cur_epoch + 1, 'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict()}
        :param is_best: boolean to save the checkpoint aside if it has the best score so far
        :param filename: the name of the saved file
        '''
        torch.save(state, '/'.join([self.checkpoint_dir,filename]))
        # if is_best:
        #     shutil.copyfile('/'.join([self.checkpoint_dir,filename]),
        #                     '/'.join([self.checkpoint_dir,'model_best.pth.tar']))

    def test_ensemble(self, epoch):
        self.net.eval()
        print("Testing...", epoch)
        correct = 0
        corrects = np.zeros(self.test_samples+1, dtype=int)
        with torch.no_grad():
            for data, target in tqdm(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                outputs = torch.zeros(self.test_samples+1, self.test_batch_size, 10).to(self.device)
                for i in range(self.test_samples):
                    outputs[i] = self.net(data, sample=True)
                outputs[self.test_samples] = self.net(data, sample=False)
                output = outputs.mean(0)
                preds = outputs.max(2, keepdim=True)[1]
                pred = output.max(1, keepdim=True)[1] # index of max log-probability
                corrects += preds.eq(target.view_as(pred)).sum(dim=1).squeeze().cpu().numpy()
                correct += pred.eq(target.view_as(pred)).sum().item()
        for index, num in enumerate(corrects):
            if index < self.test_samples:
                print('Component {} Accuracy: {}/{}'.format(index, num, len(self.test_loader.dataset)))
            else:
                print('Posterior Mean Accuracy: {}/{}'.format(num, len(self.test_loader.dataset)))
        print('Ensemble Accuracy: {}/{}'.format(correct, len(self.test_loader.dataset)))
        self.writer.add_scalar('test/error', 1-correct/len(self.test_loader.dataset), epoch)
        self.net.train()
        return 1-correct/len(self.test_loader.dataset)
