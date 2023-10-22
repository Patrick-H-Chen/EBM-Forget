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
args.coreset = True # Coreset or Pure Generative


def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

class SplitMnistGenerator():
    def __init__(self):
        self.X_train, self.train_label = load_mnist('/data/', kind='train')
        self.X_test , self.test_label = load_mnist('/data/', kind='t10k')


        self.sets_0 = [0, 2, 4, 6, 8]
        self.sets_1 = [1, 3, 5, 7, 9]
        
        self.max_iter = len(self.sets_0)
        self.cur_iter = 0

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], 2

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            # Retrieve train data
            train_0_id = np.where(self.train_label == self.sets_0[self.cur_iter])[0]
            train_1_id = np.where(self.train_label == self.sets_1[self.cur_iter])[0]
            next_x_train = np.vstack((self.X_train[train_0_id], self.X_train[train_1_id]))

            next_y_train = np.vstack((np.ones((train_0_id.shape[0], 1)), np.zeros((train_1_id.shape[0], 1))))
            next_y_train = np.hstack((next_y_train, 1-next_y_train))

            # Retrieve test data
            test_0_id = np.where(self.test_label == self.sets_0[self.cur_iter])[0]
            test_1_id = np.where(self.test_label == self.sets_1[self.cur_iter])[0]
            next_x_test = np.vstack((self.X_test[test_0_id], self.X_test[test_1_id]))

            next_y_test = np.vstack((np.ones((test_0_id.shape[0], 1)), np.zeros((test_1_id.shape[0], 1))))
            next_y_test = np.hstack((next_y_test, 1-next_y_test))

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test

data_gen = SplitMnistGenerator()
za_data = {}
for i in range(5):
    za_data[i] = {}

for i in range(5):
    x, y, x_test, y_test = data_gen.next_task()
    tmp = {}
    idx = np.arange(x.shape[0])
    idx = np.random.permutation(idx)
    x = x[idx]
    y = y[idx]
    tmp['x'] = deepcopy(torch.from_numpy(x).float()/255.)
    tmp['y'] = deepcopy(torch.from_numpy(np.where(y == 1)[1]).long())
    za_data[i]['train'] = tmp
    tmp = {}
    tmp['x'] = deepcopy(torch.from_numpy(x_test).float()/255.)
    tmp['y'] = deepcopy(torch.from_numpy(np.where(y_test == 1)[1]).long())
    za_data[i]['test'] = tmp

    


## VAE Model ##


dim1 = 256
dim2 = 256
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.w1 = nn.Parameter(.1*torch.randn((784,dim1)).cuda(args.cuda_num),requires_grad = True)
        self.b1 = nn.Parameter(.1*torch.randn((dim1)).cuda(args.cuda_num),requires_grad = True)
        self.w2 = nn.Parameter(.1*torch.randn((dim1,dim2)).cuda(args.cuda_num),requires_grad = True)
        self.b2 = nn.Parameter(.1*torch.randn((dim2)).cuda(args.cuda_num),requires_grad = True)
        self.w3 = nn.Parameter(.1*torch.randn((dim2,2)).cuda(args.cuda_num),requires_grad = True)
        self.b3 = nn.Parameter(.1*torch.randn((2)).cuda(args.cuda_num),requires_grad = True)
        
        self.w1v = nn.Parameter(-6.*torch.ones((784,dim1)).cuda(args.cuda_num),requires_grad = True)
        self.b1v = nn.Parameter(-6.*torch.ones((dim1)).cuda(args.cuda_num),requires_grad = True)
        self.w2v = nn.Parameter(-6.*torch.ones((dim1,dim2)).cuda(args.cuda_num),requires_grad = True)
        self.b2v = nn.Parameter(-6.*torch.ones((dim2)).cuda(args.cuda_num),requires_grad = True)
        self.w3v = nn.Parameter(-6.*torch.ones((dim2,2)).cuda(args.cuda_num),requires_grad = True)
        self.b3v = nn.Parameter(-6.*torch.ones((2)).cuda(args.cuda_num),requires_grad = True)

    def encode(self):
        return self.z

    def reparameterize(self, K, mu,var):
        eps = torch.stack([torch.randn_like(mu) for k in range(K)])
        return mu + torch.exp(0.5*var) * eps

    def predict(self, x,n_samples,extract):
        x = x.view(-1,784)
        x = x.view(-1,784)
        fc1_w = self.reparameterize(n_samples,self.w1,self.w1v)
        fc1_b = self.reparameterize(n_samples,self.b1,self.b1v).unsqueeze(1)
        fc2_w = self.reparameterize(n_samples,self.w2,self.w2v)
        fc2_b = self.reparameterize(n_samples,self.b2,self.b2v).unsqueeze(1)
        fc3_w = self.reparameterize(n_samples,self.w3,self.w3v)
        fc3_b = self.reparameterize(n_samples,self.b3,self.b3v).unsqueeze(1)
        l = torch.einsum('mni,bn->mbi',fc1_w,x)
        l = F.leaky_relu(l + fc1_b)
        l = torch.einsum('mni,mbn->mbi',fc2_w,l)
        l = F.leaky_relu(l + fc2_b)
        if extract:
            return l
        l = torch.einsum('mni,mbn->mbi',fc3_w,l) + fc3_b

        return l
    
    def forward(self, x,num_samples,extract = False):
        return self.predict(x,num_samples,extract)

### Helper Function for Energy-Based Generative Model ###

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


def single_complete_KLD(v1,v2):
    loss = 0
    s1  = [p[1] for p in v1.named_parameters() if p[0][-1] != 'v']
    s2  = [p[1] for p in v2.named_parameters() if p[0][-1] != 'v']
    s1v = [p[1] for p in v1.named_parameters() if p[0][-1] == 'v']
    s2v = [p[1] for p in v2.named_parameters() if p[0][-1] == 'v']
    for p in range(len(s1)):
        m0,v0,m,v = s2[p], s2v[p], s1[p], s1v[p]
        log_std_diff = 0.5 * torch.sum(v0 - v)
        mu_diff_term = 0.5 * torch.sum((torch.exp(v) + (m0 - m)**2) / torch.exp(v0))
        loss += (log_std_diff + mu_diff_term)
    return loss


total_class = 2
def sample_buffer(buffer, batch_size=128, p=0.95, device='cuda:'+str(args.cuda_num)):
    if len(buffer) < 1:
        return (
            torch.rand(batch_size, 1, 28, 28, device=device),
            torch.randint(0, total_class, (batch_size,), device=device),
        )

    n_replay = (np.random.rand(batch_size) < p).sum()

    replay_sample, replay_id = buffer.get(n_replay)
    random_sample = torch.rand(batch_size - n_replay, 1, 28, 28, device=device)
    random_id = torch.randint(0, total_class, (batch_size - n_replay,), device=device)

    return (
        torch.cat([replay_sample, random_sample], 0),
        torch.cat([replay_id, random_id], 0),
    )



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

Results = []
for NUMOFEXPERIMENTS in range(1): #num of repeated experiments

    prev_data = []
    if args.coreset:
        for p in range(5):
            x = za_data[p]['train']['x']
            y = za_data[p]['train']['y']
            prev_data.append(random.sample(list(zip(x,y)),50))

    sbatch = 512
    tmp_result = []
    za_vae = VAE()
    prior_vae = VAE()
    prior_vae.cuda(args.cuda_num)
    for p in prior_vae.parameters():
        p.data = torch.zeros_like(p.data)
    za_vae.cuda(args.cuda_num)
    crit = nn.CrossEntropyLoss()

    all_models = []



    for terms in range(5):    
        x = za_data[terms]['train']['x']
        y = za_data[terms]['train']['y']
        parameters = za_vae.parameters()

        pp = list(za_vae.parameters())
        optimizer = optim.Adam(pp,lr=0.001,betas=(0,0.999))


        if args.coreset == False: 
            if terms > 0:
                del prev_data[-1]
                tmp,la = samples(all_models[-1],x.shape[0],1000)
                prev_data.append(list(zip(tmp,la)))
            prev_data.append(random.sample(list(zip(x,y)),5000)) 

        ## Prepare buffers
        if terms > 0:
            buffer = SampleBuffer(10000)
            buffers.append(buffer)
        else:
            buffers = []
            buffer = SampleBuffer(10000)
            buffers.append(buffer)
        inner_iters = 15

        for j in range(inner_iters):
            losses = 0
            for i in range(0,x.size(0),sbatch):
                b=torch.LongTensor(np.arange(i,np.min([i+sbatch,x.size(0)]))).cuda(args.cuda_num)
                data=torch.autograd.Variable(x[b],volatile=False).cuda(args.cuda_num).float()
                target=torch.autograd.Variable(y[b],volatile=False).cuda(args.cuda_num)

                optimizer.zero_grad()
                output = za_vae(data,10)

                # Discriminative Loss of Bayesian Models
                disc_loss = torch.mean(torch.stack([crit(l,target) for l in output]))
                # KL Loss
                KL_loss = single_complete_KLD(za_vae,prior_vae)/x.shape[0]


                sample_loss = 0

                za_vae.eval()

                ### Generate Sample Via Langevin Dynamics ###
                sample_loss = 0
                L2_loss = 0
                for pp in range(terms+1):
                    if pp == terms:
                        za_model = za_vae
                    else:
                        za_model = all_models[pp]

                    # Sometimes the sampling will get error when p is too small
                    try:
                        neg_img, neg_id = sample_buffer(buffers[pp], data.shape[0],p=0.95)
                    except:
                        neg_img, neg_id = sample_buffer(buffers[pp], data.shape[0],p=0.95)

                    neg_img.requires_grad = True
                    noise = torch.randn(neg_img.shape[0], 1, 28, 28).cuda(args.cuda_num)
                    requires_grad(parameters, False)
                    
                    # Langevin Dynamic Updates
                    for k in range(5):
                        noise.normal_(0, 0.005)
                        neg_img.data.add_(noise.data)
                        neg_out = za_model.forward(neg_img,10).mean(0)

                        neg_out[torch.arange(neg_id.shape[0]),neg_id].sum().backward()

                        neg_img.grad.data.clamp_(-0.01, 0.01)
                        step_size = 10
                        neg_img.data.add_(step_size, neg_img.grad.data)

                        neg_img.grad.detach_()
                        neg_img.grad.zero_()
                        neg_img.data.clamp_(0, 1)

                    pos_img, class_ids = zip(*random.sample(prev_data[pp], k=32))
                    pos_img = torch.stack(pos_img, 0).cuda(args.cuda_num)
                    pos_id = torch.tensor(class_ids).cuda(args.cuda_num)

                    neg_img = neg_img.detach()
                    buffers[pp].push(neg_img, neg_id)
                    requires_grad(za_model.parameters(), True)

                    za_model.train()
                    za_model.zero_grad()

                    pos_out = za_model.forward(pos_img, 10).mean(0)[torch.arange(pos_id.shape[0]),pos_id]
                    neg_out = za_model.forward(neg_img, 10).mean(0)[torch.arange(neg_id.shape[0]),neg_id]
                    L2_loss +=  ((pos_out ** 2).mean() + (neg_out ** 2).mean())
                    sample_loss += (-pos_out.mean() + neg_out.mean())

                loss =   L2_loss  + sample_loss + disc_loss + KL_loss
                loss.backward() 
                clip_grad(parameters, optimizer)

                optimizer.step()


        ## Copy the model for next round
        tmp = deepcopy(za_vae)
        tmp.w1 = za_vae.w1
        tmp.b1 = za_vae.b1
        tmp.w1v = za_vae.w1v
        tmp.b1v = za_vae.b1v
        tmp.w2 = za_vae.w2
        tmp.b2 = za_vae.b2
        tmp.w2v = za_vae.w2v
        tmp.b2v = za_vae.b2v

        all_models.append(tmp)
        prior_vae = deepcopy(za_vae)
        
        # Creat a new head for the next iteration 
        za_vae.w3 = nn.Parameter(.1*torch.randn((dim2,2)).cuda(args.cuda_num),requires_grad = True)
        za_vae.b3 = nn.Parameter(.1*torch.randn((2)).cuda(args.cuda_num),requires_grad = True)
        za_vae.w3v = nn.Parameter(-6.*torch.ones((dim2,2)).cuda(args.cuda_num),requires_grad = True)
        za_vae.b3v = nn.Parameter(-6.*torch.ones((2)).cuda(args.cuda_num),requires_grad = True)


        # Get testing Accuracy upto current task

        acc = []
        for pp in range(terms+1):
            correct = 0
            tt_loss = 0
            cnt = 0
            x = za_data[pp]['test']['x']
            y = za_data[pp]['test']['y']
            wrong_idx = []
            for i in range(0,x.size(0),64):
                b=torch.LongTensor(np.arange(i,np.min([i+sbatch,x.size(0)]))).cuda(args.cuda_num)
                data=torch.autograd.Variable(x[b],volatile=False).cuda(args.cuda_num).float()
                target=torch.autograd.Variable(y[b],volatile=False).cuda(args.cuda_num)
                output = all_models[pp](data,100)
                output = torch.stack([F.softmax(o,1) for o in output]).mean(0)
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                cnt += pred.shape[0]
            acc.append(correct/cnt)
        tmp_result.append(acc)

        print(acc,np.mean(acc))
    Results.append(tmp_result)


