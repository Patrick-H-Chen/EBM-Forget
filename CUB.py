from __future__ import print_function
import os
os.environ["OMP_NUM_THREADS"] = "1"

import torchvision
import numpy as np
import gzip
import pickle
import sys
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import torch
import random

import os, sys
import numpy as np
import torch
from torchvision import datasets, transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch import autograd
import copy
import torch
import configparser

from copy import deepcopy
from torch import Tensor
from torch.autograd import Variable
from torch.autograd import grad
from torch import nn


## GPU config
args = configparser.ConfigParser()
args.cuda_num = 3
args.widen_factor = 2

variance = 0.01


num_tasks = 2
total_class = 100
groups = 10
total_tasks = total_class // num_tasks
num_train = 5000


# Data Loader

class CIFAR100Generator():
    def __init__(self):
        alllabels = np.load("/data/BIRDtrainlabel.npy",allow_pickle=True)
        alldata = np.load("/data/BIRDtrain.npy",allow_pickle=True).swapaxes(2,3).swapaxes(1,2)
        alldata = alldata/255.#((alldata/255. - 0.5)/0.5)
        self.train_label = alllabels[:num_train]
        self.X_train = alldata[:num_train]
        self.val_label = alllabels[num_train:]
        self.X_val = alldata[num_train:] 
        self.X_test = np.load("/data/BIRDtest.npy",allow_pickle=True).swapaxes(2,3).swapaxes(1,2)
        self.X_test = self.X_test/255.#(self.X_test/255. - 0.5)/0.5
        self.test_label = np.load("/data/BIRDtestlabel.npy",allow_pickle=True).reshape(-1)
        p = np.unique(self.train_label)
        np.random.seed(4431)
        p = np.random.permutation(p)
        self.sets = p.reshape(total_tasks,num_tasks)
        self.transMap = []
        for dd in range(total_tasks):
            pick = self.sets[dd]
            subMap  ={}
            for k,p in enumerate(pick):
                subMap[p] = k
            self.transMap.append(subMap)
    
        self.max_iter = len(self.sets)
        self.cur_iter = 0

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], 2

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            # Retrieve train data
            train_0_id = np.where(np.isin(self.train_label,self.sets[self.cur_iter]))[0]
            
            next_x_train = self.X_train[train_0_id]
            next_y_train = self.train_label[train_0_id]
            next_y_train = np.asarray([self.transMap[self.cur_iter][yy] for yy in next_y_train])

            val_0_id = np.where(np.isin(self.val_label,self.sets[self.cur_iter]))[0]

            next_x_val = self.X_val[val_0_id]
            next_y_val = self.val_label[val_0_id]
            next_y_val = np.asarray([self.transMap[self.cur_iter][yy] for yy in next_y_val])
            
            
            test_0_id = np.where(np.isin(self.test_label,self.sets[self.cur_iter]))[0]

            # Retrieve test data
            next_x_test = self.X_test[test_0_id]
            next_y_test = self.test_label[test_0_id]
            next_y_test = np.asarray([self.transMap[self.cur_iter][yy] for yy in next_y_test])
            

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_val, next_y_val, next_x_test, next_y_test
data_gen = CIFAR100Generator()


group = {}

for g in range(groups):
    za_data = {}
    
    for i in range(total_tasks//groups):
        za_data[i] = {}

    for i in range(total_tasks//groups):
        x, y, x_val, y_val, x_test, y_test = data_gen.next_task()
        pika = {}
        idx = np.arange(x.shape[0])
        idx = np.random.permutation(idx)
        TTTT = len(idx)
        ALAL = int(TTTT*1.)
        x = x[idx[:ALAL]]
        y = y[idx[:ALAL]]
        pika['x'] = deepcopy(torch.from_numpy(x).float()) 
        pika['y'] = deepcopy(torch.from_numpy(y).long())
        za_data[i]['train'] = pika
        pika = {}
        pika['x'] = deepcopy(torch.from_numpy(x_test).float())
        pika['y'] = deepcopy(torch.from_numpy(y_test).long())
        za_data[i]['test'] = pika

        pika = {}
        pika['x'] = deepcopy(torch.from_numpy(x_val).float())
        pika['y'] = deepcopy(torch.from_numpy(y_val).long())
        za_data[i]['val'] = pika


    group[g] = za_data

## VAE Model ##


class VAEBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(VAEBasicBlock, self).__init__()
        self.bn1 = nn.Identity()
        self.relu1 = nn.LeakyReLU(.2)
        self.stride = stride
        
        self.bn2 = nn.Identity()
        self.relu2 = nn.LeakyReLU(.2)
        self.droprate = dropRate
        self.dropout = nn.Identity() if dropRate == 0.0 else nn.Dropout(p=dropRate)
        self.shortCut =  stride != 1 or in_planes != out_planes

        self.c1 = nn.Parameter(regular(in_planes,out_planes,3).cuda(args.cuda_num),requires_grad = True)
        self.c2 = nn.Parameter(regular(out_planes,out_planes,3).cuda(args.cuda_num),requires_grad = True)
        if self.shortCut:
            self.sc1 = nn.Parameter(regular(in_planes,out_planes,1).cuda(args.cuda_num),requires_grad = True)

        self.id = nn.Identity()
    def reparameterize(self, mu, var=0):
        eps = torch.randn_like(mu) 
        return mu + variance*eps 

    def forward(self, x):
        c1_w  = self.reparameterize(self.c1)
        c2_w  = self.reparameterize(self.c2)
                
        out = self.dropout(F.conv2d(self.relu1(self.bn1(x)),c1_w, padding = 1))
        out = F.conv2d(self.relu2(self.bn2(out)),c2_w,stride=self.stride,padding = 1)
        if self.shortCut:
            sc1_w  = self.reparameterize(self.sc1)
            out += F.conv2d(x,sc1_w, stride=self.stride)  
        else:
            out += self.id(x)
        return out


class VAEWideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(VAEWideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = VAEBasicBlock
        self.c1 = nn.Parameter(regular(3,nChannels[0],3).cuda(args.cuda_num),requires_grad = True)
        self.in_planes = 16
        
        self.block1 = self._wide_layer(block, nChannels[1], n, dropRate, stride=1)
        self.block2 = self._wide_layer(block, nChannels[2], n, dropRate, stride=2)
        self.block3 = self._wide_layer(block, nChannels[3], n, dropRate, stride=2)

        self.bn1 = nn.Identity()
        self.relu = nn.LeakyReLU(.2)
        self.fcs = [ nn.Linear(nChannels[3], num_tasks).cuda(args.cuda_num) for p in range(total_tasks)]
        self.nChannels = nChannels[3]

    def reparameterize(self, mu, var=0):
        eps = torch.randn_like(mu) 
        return mu + variance*eps  
    
    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,dropout_rate))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def predict(self, x,ii):
        c1_w  = self.reparameterize(self.c1)
        out = F.conv2d(x,c1_w, stride=1, padding=1)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fcs[ii](out)

    def forward(self, x,ii,num_samples):
        return torch.stack([self.predict(x,ii) for p in range(num_samples)])




### Helper Function for Energy-Based Generative Model ###


def regular(in_planes,planes,kernel_size):
    conv = nn.Conv2d(in_planes,planes, kernel_size=kernel_size, padding=1, bias=True)
    return conv.weight.data

class SampleBuffer:
    def __init__(self, max_samples=10000):
        self.max_samples = max_samples
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def push(self, samples, class_ids=None):
        samples = samples.detach().to('cpu')
        class_ids = class_ids.detach().to('cpu')

        for sample, class_id in zip(samples, class_ids):
            self.buffer.append((sample.detach(), class_id))

            if len(self.buffer) > self.max_samples:
                self.buffer.pop(0)

    def get(self, n_samples, device='cuda:'+str(args.cuda_num)):
        items = random.sample(self.buffer, k=n_samples)
        samples, class_ids = zip(*items)
        samples = torch.stack(samples, 0)
        class_ids = torch.tensor(class_ids)
        samples = samples.to(device)
        class_ids = class_ids.to(device)

        return samples, class_ids


def single_fix_var_KLD(v1,v2=None,first=False):
    loss = 0
    s1  = [p[1] for p in v1.named_parameters() if (p[0][-1] == "1" or p[0][-1] == "2")]
    if first == True:
        for p in range(len(s1)):
            loss += 0.5*torch.sum(s1[p]**2/variance)
        return loss
    s2  = [p[1] for p in v2.named_parameters() if (p[0][-1] == "1" or p[0][-1] == "2")]
    s1v = [p[1] for p in v1.named_parameters() if p[0][-1] == 'v']
    s2v = [p[1] for p in v2.named_parameters() if p[0][-1] == 'v']
    for p in range(len(s1)):
        m0,m = s2[p], s1[p]
        log_std_diff = 0
        mu_diff_term = 0.5 * torch.sum((variance + (m0 - m)**2) / variance)
        loss += (log_std_diff + mu_diff_term)
    return loss

def sample_buffer(buffer, batch_size=128, p=0.95, device='cuda:'+str(args.cuda_num),y=None):
    buffer_size = len(buffer) if y is None else len(buffer) // num_tasks
    inds = torch.randint(0, buffer_size, (batch_size,))
    if y is not None:
        inds = y.cpu() * buffer_size + inds
    buffer_samples = buffer[inds]
    random_samples = torch.FloatTensor(batch_size, 3, 32, 32).uniform_(0, 1)
    choose_random = (torch.rand(batch_size) >= p).float()[:, None, None, None]
    samples = choose_random * random_samples + (1 - choose_random) * buffer_samples
    return samples.to(device), inds


def testAcc(model,ii):
    acc = []
    losses = []
    for pp in range(ii+1):
        correct = 0
        tt_loss = 0
        cnt = 0
        x = za_data[pp]['test']['x']
        y = za_data[pp]['test']['y']
        wrong_idx = []
        for i in range(0,x.size(0),sbatch):
            b=torch.LongTensor(np.arange(i,np.min([i+sbatch,x.size(0)]))).cuda(args.cuda_num)
            data=torch.autograd.Variable(x[b],volatile=False).cuda(args.cuda_num).float()
            target=torch.autograd.Variable(y[b],volatile=False).cuda(args.cuda_num)
            output = model(data,pp,5)
            output = torch.stack([F.softmax(o,1) for o in output]).mean(0)
            pred = output.argmax(dim=1, keepdim=True) # get the index of he max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            cnt += pred.shape[0]
        acc.append(correct/cnt)
    return acc

def valAcc(model,ii):
    acc = []
    losses = []
    for pp in range(ii+1):
        correct = 0
        tt_loss = 0
        cnt = 0
        x = za_data[pp]['val']['x']
        y = za_data[pp]['val']['y']
        wrong_idx = []
        for i in range(0,x.size(0),64):
            b=torch.LongTensor(np.arange(i,np.min([i+sbatch,x.size(0)]))).cuda(args.cuda_num)
            data=torch.autograd.Variable(x[b],volatile=False).cuda(args.cuda_num).float()
            target=torch.autograd.Variable(y[b],volatile=False).cuda(args.cuda_num)
            output = model(data,pp,1)
            output = torch.stack([F.softmax(o,1) for o in output]).mean(0)
            pred = output.argmax(dim=1, keepdim=True) # get the index of he max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            cnt += pred.shape[0]
        acc.append(correct/cnt)
    return acc


def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag

def clip_grad(parameters, optimizer):
    with torch.no_grad():
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]

                if 'step' not in state or state['step'] < 1:
                    continue

                step = state['step']
                exp_avg_sq = state['exp_avg_sq']
                _, beta2 = group['betas']

                bound = 3 * torch.sqrt(exp_avg_sq / (1 - beta2 ** step)) + 0.1
                p.grad.data.copy_(torch.max(torch.min(p.grad.data, bound), -bound))

def samples(model,batch_size=10,time_stpes=200):
    neg_img = torch.randn(batch_size, 1, 28, 28).cuda(args.cuda_num)
    neg_img.requires_grad = True    
    neg_id = torch.randint(0, total_class, (batch_size,)).cuda(args.cuda_num)
    parameters = model.parameters()
    requires_grad(parameters, False)
    model.eval()
    noise = torch.randn(neg_img.shape[0], 1, 28, 28).cuda(args.cuda_num)
    outer_steps = time_stpes // 60

    for oo in range(outer_steps):
        for k in range(60):
            noise.normal_(0, 0.005)
            neg_img.data.add_(noise.data)
            neg_out = model.forward(neg_img,10).mean(0)

            neg_out[torch.arange(neg_id.shape[0]),neg_id].sum().backward()

            neg_img.grad.data.clamp_(-0.01, 0.01)

            neg_img.data.add_(10, neg_img.grad.data)

            neg_img.grad.detach_()
            neg_img.grad.zero_()

            neg_img.data.clamp_(0, 1)
        model.train()
        requires_grad(model.parameters(), True)

    neg_img = neg_img.detach()
    return neg_img,neg_id







### Main Training ###

tasks = []
sampling_times = 3
for za_group in range(10):
    za_data = group[za_group]

    Results = []
    for repeated_iters_for_a_group in range(10):
        model = VAEWideResNet(16, 10,
                            args.widen_factor, dropRate=0.)
        model = model.cuda(args.cuda_num)

        parameters = list(model.parameters())
        for p in model.fcs:
            for j in p.parameters():
                parameters.append(j)

        optimizer = torch.optim.Adam(parameters, lr=0.0001, betas=[.9, .999], weight_decay=0)

        sbatch = 64

        prevs = []
        all_models = []
        important_point_idx = []
        worst_case_record = []
        crit = nn.CrossEntropyLoss()
        tmp_result = []
        total_prev = []
        prev_data = []
        for terms in range(5):#All tasks in a group is 5

            if terms > 0:
                prev_data.append(deepcopy(buffers[terms-1]))

            ii = terms
            x = za_data[terms]['train']['x']
            y = za_data[terms]['train']['y']
            model.train()

            if terms > 0:
                buffer = torch.FloatTensor(200, 3, 32, 32).uniform_(0, 1)
                buffers.append(buffer)
            else:
                buffers = []
                buffer = torch.FloatTensor(200, 3, 32, 32).uniform_(0, 1)
                buffers.append(buffer)


            for j in range(200): #inner iterations for a binary classification task
                losses = 0

                for i in range(0,x.size(0),sbatch):

                    b=torch.LongTensor(np.arange(i,np.min([i+sbatch,x.size(0)])))
                    data=torch.autograd.Variable(x[b],volatile=False).view(-1,3,32,32).cuda(args.cuda_num).float()
                    target=torch.autograd.Variable(y[b],volatile=False).cuda(args.cuda_num)
                    optimizer.zero_grad()

                    output = model(data,terms,sampling_times)

                    loss1 = torch.mean(torch.stack([crit(l,target) for l in output]))

                    if terms == 0:
                        KL_loss = single_fix_var_KLD(model,None,True)/x.shape[0]*0.0001
                    else:
                        KL_loss = single_fix_var_KLD(model,prior_model,False)/x.shape[0]*0.0001

                    pos_img = data
                    pos_id = target


                    neg_img, buffer_inds = sample_buffer(buffers[terms], target.shape[0],p=.95,y=target)
                    neg_id = target
                    neg_img.requires_grad = True
                    model.eval()
                    x_k = torch.autograd.Variable(neg_img, requires_grad=True)

                    for k in range(20):
                        neg_out = model.forward(x_k,terms,sampling_times).mean(0)
                        f_prime = torch.autograd.grad(torch.gather(neg_out,1,neg_id[:,None]).sum(), [x_k], retain_graph=True)[0]
                        x_k.data += 1 * f_prime + 0.01 * torch.randn_like(x_k)
                        x_k.data.clamp_(0, 1)
                    neg_img = x_k.detach()
                    buffers[terms][buffer_inds] = neg_img.cpu()
                    requires_grad(model.parameters(), True)

                    model.train()

                    neg_out = torch.gather(model.forward(neg_img,terms,sampling_times).mean(0),1,neg_id[:,None]).mean()
                    pos_out = torch.gather(model.forward(pos_img,terms,sampling_times).mean(0),1,pos_id[:,None]).mean()

                    L2_loss =  ((pos_out ** 2).mean() + (neg_out ** 2).mean())


                    sample_loss = (-pos_out.mean() + neg_out.mean())
                    loss = loss1 + .2*sample_loss+ 0.2*L2_loss + KL_loss
                    loss.backward()


                    ### Generate Sample Via Langevin Dynamics ###
                    for pp in range(terms):
                        pos_id = target
                        pos_img, buffer_inds = sample_buffer(prev_data[pp], target.shape[0],p=100,y=pos_id)

                        neg_img, buffer_inds = sample_buffer(buffers[pp], pos_id.shape[0],p=.95,y=pos_id)
                        neg_id = pos_id
                        neg_img.requires_grad = True
                        pos_output = model(pos_img,pp,sampling_times)

                        requires_grad(parameters, False)
                        model.eval()
                        x_k = torch.autograd.Variable(neg_img, requires_grad=True)

                        for k in range(20):
                            neg_out = model.forward(x_k,pp,sampling_times).mean(0)
                            f_prime = torch.autograd.grad(torch.gather(neg_out,1,pos_id[:,None]).sum(), [x_k], retain_graph=True)[0]
                            x_k.data += 1 * f_prime + variance * torch.randn_like(x_k)
                            x_k.data.clamp_(-1, 1)

                        neg_img = x_k.detach()
                        buffers[pp][buffer_inds] = neg_img.cpu()
                        requires_grad(model.parameters(), True)

                        model.train()

                        pos_out = torch.gather(pos_output.mean(0),1,neg_id[:,None]).mean()
                        neg_out = torch.gather(model.forward(neg_img,pp,sampling_times).mean(0),1,neg_id[:,None]).mean()

                        L2_loss =  ((pos_out ** 2).mean() + (neg_out ** 2).mean())


                        sample_loss = (-pos_out.mean() + neg_out.mean())

                        loss = .2*(sample_loss) + 0.2*L2_loss
                        loss.backward()
                    optimizer.step()
                print(j,loss1.item(),sample_loss.item())
            acc = testAcc(model,terms)
            prior_model = deepcopy(model)
            tmp_result.append(acc)
            print(acc,np.mean(acc))
        Results.append(tmp_result)



    tasks.append(Results)

    


