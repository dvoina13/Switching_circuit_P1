from __future__ import print_function
import torch
import numpy as np
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--my_path", type=str, default='/home/dvoina/ramsmatlabprogram/BSR_2/BSDS500/data/images/train/im_dir/', required=False, help="Set path for image dataset")
parser.add_argument("--Nf2", type=int, default=5, required=False, help="number of VIP neuros/u.s.")
parser.add_argument("--c", type=int, default=3, required=False, help="dimension of spatial kernel")
parser.add_argument("--tau", type=int, default=2, required=False, help="set tau/synaptic delay")

args = parser.parse_args()

mypath = args.my_path
Nf2 = args.Nf2
c = args.c
tau = args.tau

#load all Ws
W_stat = np.load('/home/dvoina/simple_vids/results/W_43x43_34filters_static_simple_3px_ReviewSparse2.npy')
W_mov = np.load('/home/dvoina/simple_vids/results/W_43x43_34filters_moving_simple_3px_tau' + str(tau) + '_ReviewSparse.npy')

W_s2p = np.copy(W_stat)
W_s2p[W_s2p>0] = 0
W_moving_minus_static = W_mov - W_stat

W_stat_matrix_torch = torch.from_numpy(W_stat)
W_mov_matrix_torch = torch.from_numpy(W_mov)
W_s2p_torch = torch.from_numpy(W_s2p)
W_moving_minus_static_torch = torch.from_numpy(W_moving_minus_static)

# load all X's
#XTrain = np.load('/home/dvoina/Prep_forPython/XTrain_forPython_34filters.npy')
X = np.load('XTrain_forPython_34filters_noNoise_reviewSimple.npy')
X = X.squeeze()


c1 = c; c2 = c; c3 = c;
N, D, H = 4900, 167, 167
Nf = 34

print(Nf2)
n1 = int(math.floor(c1/2)); n2 = int(math.floor(c2/2)); n3 = int(math.floor(c3/2));
n = int(N/2/100)

#class for Model

class Main_NN(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Main_NN, self).__init__()
        self.conv_p2v = torch.nn.Conv2d(Nf, Nf2, c1, bias=None, padding=0)
        self.conv_v2s = torch.nn.Conv2d(Nf2, Nf, c2, bias=None, padding=0)
        self.conv_s2p = torch.nn.Conv2d(Nf, Nf, 43, bias=None, padding=0)  # conv for sure
        self.conv_v2p = torch.nn.Conv2d(Nf2, Nf, c3, bias=None, padding=0)
        self.conv_movstat = torch.nn.Conv2d(Nf, Nf, 43, bias=None, padding=0)  # conv for sure

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        #x1 = self.relu(self.conv_p2v(x))
        x1 = self.conv_p2v(x)
        x2 = self.conv_movstat(x)
        y1 = -self.conv_v2p(x1)
        y2 = -self.conv_v2s(x1)
        y3 = self.conv_s2p(y2)

        if (n1+n3!=0):
            x2 = x2[:,:,n3+n1:-n3-n1, n3+n1:-n3-n1]

        y1 = y1[:,:,21:-21, 21:-21]
        res = x2 - y1 - y3

        return res

cuda = torch.cuda.is_available()

# Construct our model by instantiating the class defined above
model = Main_NN()

model.conv_s2p.weight.data.copy_(W_s2p_torch)
model.conv_movstat.weight.data.copy_(W_moving_minus_static_torch)
model.conv_v2s.weight.data.fill_(0.5)
model.conv_v2p.weight.data.fill_(0.1)

if cuda:
    model.cuda()

for name, param in model.named_parameters():
        param.requires_grad = True

for name, param in model.named_parameters():
    if 'bias' in name:
        param.requires_grad = False
    if 's2p' in name:
        param.requires_grad = False
    if 'movstat' in name:
        param.requires_grad = False
    
for name, param in model.named_parameters():
        #if param.requires_grad:
        print(name, param.requires_grad)

#separate Train and Validation data-sets
M = int(N/10)

XTrain = np.zeros((N-M,Nf,D,D))
XValid = np.zeros((M,Nf,D,D))
#randD = np.random.permutation((1600))
randD = np.random.permutation(N)

X = X[randD, :,:,:]
XTrain[0:int((N-M)/2),:,:,:] = np.squeeze(X[0:int((N-M)/2),:,:,:])
XTrain[int((N-M)/2):N-M,:,:,:] = np.squeeze(X[int((N-M)/2) + int(M/2):N-int(M/2),:,:,:])
XValid[0:int(M/2),:,:,:] = np.squeeze(X[int((N-M)/2):int(N/2),:,:,:])
XValid[int(M/2):M,:,:,:] = np.squeeze(X[N-int(M/2):N,:,:,:])

XTrain = torch.from_numpy(XTrain).float()
XValid = torch.from_numpy(XValid).float()

YTrain = torch.zeros(N-M, Nf, D-2*21-2*n1-2*n2, D-2*21-2*n1-2*n2).float()
YValid = torch.zeros(M, Nf, D-2*21-2*n1-2*n2, D-2*21-2*n1-2*n2).float()

#train the network
criterion = torch.nn.MSELoss()  # reduction='sum')
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

Nit = 5
batch_size = 5
num_batches = XTrain.shape[0] // batch_size
#num_batches = N-M
step = 1000

save_results = np.zeros((Nit, num_batches))
save_valid_results = np.zeros((Nit, int(num_batches/step)+2))

Wp2v = np.zeros((Nit, int(num_batches/step)+2, Nf2, Nf, c1, c1))
Wv2s = np.zeros((Nit, int(num_batches/step)+2, Nf, Nf2, c2, c2))
Wv2p = np.zeros((Nit, int(num_batches/step)+2, Nf, Nf2, c3, c3))

# num_batches = 1
for t in range(Nit):
    #for i in range(num_batches):
    for i in range(num_batches-1):
        # Forward pass: Compute predicted y by passing x to the model

        x = XTrain[i*batch_size:(i+1)*batch_size, :,:,:]#.unsqueeze(1)
        y = YTrain[i*batch_size:(i+1)*batch_size, :, :, :]#.unsqueeze(1)

        if cuda:
            x = x.cuda()
            y = y.cuda()

        y_pred = model(x)

        # Compute and print loss
        loss = criterion(y_pred, y)
        print(i, loss.item())
        save_results[t, i] = loss.item()

        if (i % step == 0) | (i == num_batches-1):
            model.cpu()
            print("valid")
            yvalid_pred = model(XValid)

            loss_valid = criterion(yvalid_pred, YValid)
            save_valid_results[t, int(i / step)] = loss_valid.item()
            print(i, loss_valid.item())

            Wp2v[t, int(i/step), :, :, :, :] = model.conv_p2v.weight.data.numpy()
            Wv2p[t, int(i/step), :, :, :, :] = model.conv_v2p.weight.data.numpy()
            Wv2s[t, int(i/step), :, :, :, :] = model.conv_v2s.weight.data.numpy()

            if cuda:
                model.cuda()

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # clamp weights
        #model.conv_p2v.weight.data.clamp_(min=0)  #comment out if you want SST to VIP connection to be included!
        model.conv_v2p.weight.data.clamp_(min=0)
        model.conv_v2s.weight.data.clamp_(min=0)

model.cpu()

print("final validation results")
yvalid_pred = model(XValid)
loss_valid = criterion(yvalid_pred, YValid)
print(loss_valid.item())
#save_valid_results[t,int(i/step)] = loss_valid.item()

np.save('/home/dvoina/simple_vids/results/results34_forNatImagsVids_valid_params_simple3px_Nf' + str(Nf2) + '_review2.npy', save_valid_results)
np.save('/home/dvoina/simple_vids/results/results34_forNatImagsVids_training_params_simple3px_Nf' + str(Nf2) +'review2.npy', save_results)

np.save('/home/dvoina/simple_vids/results/Wp2v_34_param_simple3px_sst2vip_Nf' + str(Nf2) + '_review2', Wp2v)
np.save('/home/dvoina/simple_vids/results/Wv2p_34_param_simple3px_sst2vip_Nf' + str(Nf2) + '_review2', Wv2p)
np.save('/home/dvoina/simple_vids/results/Wv2s_34_param_simple3px_sst2vip_Nf' + str(Nf2)+ '_review2', Wv2s)

print("min validation results")
print(np.min(save_valid_results))
ind = np.argmin(save_valid_results, axis=1)
print(ind)
print("v2p")
print(np.mean(Wv2p[2,ind,:,:,:,:]))
print("v2s")
print(np.mean(Wv2s[2,ind,:,:,:,:]))
print("p2v")
print(np.mean(Wp2v[2,ind,:,:,:,:]))

model.conv_p2v.weight.data.copy_(torch.zeros((Nf2, Nf, c1, c1)))
y_pred2 = model(XValid)
loss2 = criterion(y_pred2, YValid)
Loss = loss2.item()
print("normed base loss")
print(Loss)
