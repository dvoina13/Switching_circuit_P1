from __future__ import print_function
import torch
import numpy as np
import scipy.io
import math
from os import listdir
import matplotlib.pyplot as plt
import os

#load all Ws
W_stat = np.load('/home/dvoina/simple_vids/scripts/results/W_43x43_34filters_static_simple_3px.npy')
W_mov = np.load('/home/dvoina/simple_vids/scripts/results/W_43x43_34filters_moving_simple_3px_tau2.npy')

W_s2p = np.copy(W_stat)
W_s2p[W_s2p>0] = 0
W_moving_minus_static = W_mov - W_stat

W_stat_matrix_torch = torch.from_numpy(W_stat)
W_mov_matrix_torch = torch.from_numpy(W_mov)
W_s2p_torch = torch.from_numpy(W_s2p)
W_moving_minus_static_torch = torch.from_numpy(W_moving_minus_static)

# load all X's
XTrain = np.load('/home/dvoina/simple_vids/repository_project1/XTrain_forPython_34filters_noNoise_reviewSimple.npy')
XTrain = np.squeeze(XTrain, axis=1)

c1 = 3; c2 = 3; c3 = 3;
N, D, H = 14700, 167, 167 #19400, 167, 167
Nf = 34
Nf2 = 5 #CHANGE!!!

print(Nf2)
n1 = int(math.floor(c1/2)); n2 = int(math.floor(c2/2)); n3 = int(math.floor(c3/2));

#class for Model
class Main_NN(torch.nn.Module):
    def __init__(self, D, H):

        super(Main_NN, self).__init__()
        self.conv_v2s = torch.nn.Linear(1, Nf*D*H, bias=True)
        self.conv_s2p = torch.nn.Conv2d(Nf, Nf, 43, bias=None, padding=0)
        self.conv_v2p = torch.nn.Linear(1, Nf*D*H, bias=True)
        self.conv_movstat = torch.nn.Conv2d(Nf, Nf, 43, bias=None, padding=0)

    def forward(self, x, cuda):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        [n, NF, Nx, Ny] = x.size()

        x = self.conv_movstat(x)
        if cuda:
            y1 = -self.conv_v2p(torch.from_numpy(np.array([0])).cuda().float()).view(1, NF, Nx, Ny).expand(n,NF,Nx,Ny)
            y2 = -self.conv_v2s(torch.from_numpy(np.array([0])).cuda().float()).view(1, NF, Nx, Ny).expand(n,NF,Nx,Ny)
        else:
            y1 = -self.conv_v2p(torch.from_numpy(np.array([0])).float()).view(1, NF, Nx, Ny).expand(n, NF, Nx,Ny)
            y2 = -self.conv_v2s(torch.from_numpy(np.array([0])).float()).view(1, NF, Nx, Ny).expand(n, NF, Nx,Ny)

        y3 = self.conv_s2p(y2)

        y1 = y1[:,:,21:-21, 21:-21]

        res= x - y1 - y3
        return res

cuda = torch.cuda.is_available()

# Construct our model by instantiating the class defined above
model = Main_NN(D, H)

model.conv_s2p.weight.data.copy_(W_s2p_torch)
model.conv_movstat.weight.data.copy_(W_moving_minus_static_torch)

if cuda:
    model.cuda()

for name, param in model.named_parameters():
        param.requires_grad = True

for name, param in model.named_parameters():
    if 's2p' in name:
        param.requires_grad = False
    if 'movstat' in name:
        param.requires_grad = False
    if 'weight' in name:
        param.requires_grad = False

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

#separate Train and Validation data-sets
M = 50 #int(N/10)
print(N,M,D,H)

XTrain_pre = np.zeros((N-M,Nf,D,D))
XValid_pre = np.zeros((M,Nf,D,D))

YTrain_pre = np.zeros((N,Nf,D-2*21,D-2*21))  #CHANGE

#randD = np.random.permutation((1600))
randD = np.random.permutation(N)

XTrain = XTrain[randD, :,:,:]
XTrain_pre[0:int((N-M)/2),:,:,:] = np.squeeze(XTrain[0:int((N-M)/2),:,:,:])
XTrain_pre[int((N-M)/2):N-M,:,:,:] = np.squeeze(XTrain[int((N-M)/2) + int(M/2):N-int(M/2),:,:,:])
XValid_pre[0:int(M/2),:,:,:] = np.squeeze(XTrain[int((N-M)/2):int(N/2),:,:,:])
XValid_pre[int(M/2):M,:,:,:] = np.squeeze(XTrain[N-int(M/2):N,:,:,:])

XTrain = torch.from_numpy(XTrain_pre).float()
XValid = torch.from_numpy(XValid_pre).float()

YTrain = torch.from_numpy(YTrain_pre[0:N-M,:,:,:]).float()
YValid = torch.from_numpy(YTrain_pre[N-M:N,:,:,:]).float()

#train the network

criterion = torch.nn.MSELoss()  # reduction='sum')
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

Nit = 1
batch_size = 1
num_batches = XTrain.shape[0] // batch_size
num_batches = N-M #17400
step = 1000

save_results = np.zeros((Nit, num_batches))
save_valid_results = np.zeros((Nit, int(N/step)+1))

bias_v2p = np.zeros((Nit*int(N/step)+1, Nf*D*H))
bias_v2s = np.zeros((Nit*int(N/step)+1, Nf*D*H))


for t in range(Nit):
    for i in range(num_batches):
        # Forward pass: Compute predicted y by passing x to the model

        x = XTrain[i, :,:,:].unsqueeze(0)
        y = YTrain[i, :, :, :].unsqueeze(0)

        if cuda:
            x = x.cuda()
            y = y.cuda()

        y_pred = model(x, cuda)

        # Compute and print loss
        loss = criterion(y_pred, y)
        print(i, loss.item())
        save_results[t, i] = loss.item()

        if (i % step == 0):
            model.cpu()
            print("valid")
            yvalid_pred = model(XValid, 0)
            loss_valid = criterion(yvalid_pred, YValid)
            save_valid_results[t, int(i / step)] = loss_valid.item()
            print(i, loss_valid.item())

            bias_v2p[t * (int(num_batches/step)) + int(i/step), :] = model.conv_v2p.bias.data.numpy()
            bias_v2s[t * (int(num_batches/step)) + int(i/step), :] = model.conv_v2s.bias.data.numpy()

            if cuda:
                model.cuda()

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # clamp weights
        model.conv_v2p.bias.data.clamp_(min=0)
        model.conv_v2s.bias.data.clamp_(min=0)

model.cpu()
i = 14000
bias_v2p[t * (int(num_batches / step)) + int(i / step), :] = model.conv_v2p.bias.data.numpy()
bias_v2s[t * (int(num_batches / step)) + int(i / step), :] = model.conv_v2s.bias.data.numpy()

yvalid_pred = model(XValid, 0)
loss_valid = criterion(yvalid_pred, YValid)
save_valid_results[t,int(i/step)] = loss_valid.item()

np.save('results34_forNatImagsVids_valid_params_3px_tau2_inc_Nf' + str(Nf2) + '.npy', save_valid_results)
np.save('results34_forNatImagsVids_training_params_3px_tau2_inc_Nf' + str(Nf2) +'.npy', save_results)
np.save('Wv2p_34_param_3px_tau2_incomplete_Nf' + str(Nf2), bias_v2p)
np.save('Wv2s_34_param_3px_tau2_incomplete_Nf' + str(Nf2), bias_v2s)

print(np.min(save_valid_results))
ind = np.argmin(save_valid_results)
print(ind)
print("v2p")
print(np.mean(bias_v2p[ind,:]))
print("v2s")
print(np.mean(bias_v2s[ind,:]))

model.conv_v2p.bias.data.copy_(torch.zeros((Nf*D*H)))
model.conv_v2s.bias.data.copy_(torch.zeros((Nf*D*H)))

y_pred2 = model(XValid, 0)
y_pred2 = y_pred2
loss2 = criterion(y_pred2, YValid)
Loss = loss2.item()
print("normed base loss")
print(Loss)
