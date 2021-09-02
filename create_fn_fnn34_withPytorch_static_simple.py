import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
from os import listdir
from os.path import isfile, join
from scipy.io import loadmat # for loading mat files
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.image as mpimg
import math

def ToGrayscale(sample):
        # Convert to numpy array and then make the image grayscale
        sample = np.asarray(sample)
        sample = sample.sum(axis=-1)  # sum over last axis
        sample = sample - np.mean(sample)
        sample = sample / np.max(sample)  # divide by max over the image
        sample = np.pad(sample, (7, 7), 'symmetric')  # pad with symmetric borders (assumes 15x15 filter)
        #sample = np.expand_dims(sample, axis=-1)  # add extra channels dimension

        return sample

class Net(nn.Module):
    def __init__(self, filters_temp, filters_notemp, num_rfs):
        super(Net, self).__init__()
        # Convert set of numpy ndarray filters into Pytorch
        self.filts_temp = nn.Parameter(torch.from_numpy(filters_temp).permute(1, 0, 2, 3, 4).float(),
                                       requires_grad=False)
        self.filts_notemp = nn.Parameter(torch.from_numpy(filters_notemp).permute(1, 0, 2, 3, 4).float(),
                                          requires_grad=False)
        self.num_rfs = num_rfs  # Number of RFs to look out for correlations

    def forward(self, x):
        # Define spatial extent of correlations
        corr_x = self.num_rfs * self.filts_temp.shape[3]
        corr_y = self.num_rfs * self.filts_temp.shape[4]
        num_filts = self.filts_temp.shape[0] + self.filts_notemp.shape[0]

        # Convolve filters with input image
        x = torch.squeeze(x)

        x_new = x.expand(2, x.size()[0], x.size()[1])
        x = x_new.view(1,1,x_new.size()[0], x_new.size()[1], x_new.size()[2])
        x = x.float()

        x_temp = F.relu(F.conv3d(x, self.filts_temp)/2)
        x_notemp = F.relu(F.conv3d(x[:, :, x.size()[2]-1, :, :].unsqueeze(2), self.filts_notemp))
        x = torch.cat((x_temp, x_notemp), dim=1).float()

        # Normalization with added eps in denominator
        x1 = torch.div(x, torch.sum(x, dim=1).unsqueeze(1) + np.finfo(float).eps)

        x_max = x1.size()[4]
        y_max = x1.size()[3]

        x1_filts = x1[:, :, :, corr_y:y_max - corr_y, corr_x:x_max - corr_x].contiguous().view(num_filts, 1,1, y_max - 2 * corr_y, x_max - 2 * corr_x)  # select subset

        x1 = x1.view(1,1,num_filts,y_max,x_max)

        x2 = F.conv3d(x1, x1_filts, groups=1)
        x2 = x2.squeeze().view(1, num_filts, num_filts, 2 * corr_y + 1, 2 * corr_x + 1)

        # We are using a 231x391 size filter
        x2 = torch.div(x2, (y_max - 2 * corr_y) * (x_max - 2 * corr_y))  # normalize by size of filter

        return x1, x2

def train():
    model.eval()

    mypath = '/home/dvoina/ramsmatlabprogram/BSR_2/BSDS500/data/images/train/im_dir'
    # mypath = '/home/dvoina/vip_project/dir_for_videos'
    onlyfiles = [f for f in listdir(mypath)]

    for batch_idx in range(np.shape(onlyfiles)[0]):

            data = mpimg.imread(
            '/home/dvoina/ramsmatlabprogram/BSR_2/BSDS500/data/images/train/im_dir/' + onlyfiles[batch_idx])

            data = ToGrayscale(data)
            data = torch.from_numpy(data).float()

            if cuda:
                data = data.cuda()

            with torch.no_grad():
                data = Variable(data)  # convert into pytorch variables

                x1, x2 = model(data)  # forward inference

                x1 = x1.view(1,34,x1.size()[3],x1.size()[4])
                filt_avgs = torch.mean(x1.data.view(1, x1.shape[1], -1), dim=2).squeeze()
                fn_array.append(filt_avgs.cpu().numpy())  # load back to cpu and convert to numpy

                # f_nn's
                grid_space = 1  # can also choose 7 (original)
                x2_subset = x2[:, :, :, (45 - 21):(45 + 21 + 1):grid_space, (45 - 21):(
                            45 + 21 + 1):grid_space].data.squeeze()  # Python doesn't include end, so add 1
                fnn_array.append(x2_subset.cpu().numpy())

    return np.asarray(fn_array), np.asarray(fnn_array)

#first, load the filters (34 spatio-temporal filters)
filters34_2 = loadmat('data_filts_px3_v2.mat')

filters34_temp = np.array(filters34_2['data_filts2'][0,0:16].tolist())
filters34_temp = np.expand_dims(filters34_temp, axis=0)
filters34_temp = np.transpose(filters34_temp, (0,1,4,2,3))

filters34_notemp = np.array(filters34_2['data_filts2'][0,16:34].tolist())
filters34_notemp = np.expand_dims(filters34_notemp, axis=1)
filters34_notemp = np.expand_dims(filters34_notemp, axis=0)

# Let's zero mean the filters (make use of numpy broadcasting)
filters34_temp = np.transpose(np.transpose(filters34_temp, (0,2,3,4,1))-filters34_temp.reshape(1,filters34_temp.shape[1],-1).mean(axis=2), (0,4,1,2,3))
filters34_notemp = np.transpose(np.transpose(filters34_notemp, (0,2,3,4,1))-filters34_notemp.reshape(1,filters34_notemp.shape[1],-1).mean(axis=2), (0,4,1,2,3))

# Training settings
cuda = torch.cuda.is_available() # disables using the GPU and cuda if False
batch_size = 1 # input batch size for training (TODO: figure out how to group images with similar orientation)

# Create a new instance of the network
model = Net(filters34_temp, filters34_notemp, num_rfs=3)

if cuda:
    model.cuda()

# Use a list for saving the activations for each image
fn_array = []
fnn_array = []

filt_avgs, fnn_avgs = train()
filt_avgs_images = np.mean(filt_avgs, axis=0)
fnn_avgs_images = np.mean(fnn_avgs, axis=0)

W = np.empty(fnn_avgs_images.shape)
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        W[i,j,:] = fnn_avgs_images[i,j,:].squeeze()/(filt_avgs_images[i]*filt_avgs_images[j]) - 1

def construct_row4(w, dim, flag):

    Nx = dim[0]
    Ny = dim[1]

    center2 = int(math.floor(Ny/2))

    #grid1 = np.concatenate((np.array(range(center2-3*7, center2, 7)), np.array(range(center2, center2+4*7, 7))))
    #grid2 = np.concatenate((np.array(range(center2-3*7, center2, 7)), np.array(range(center2, center2+4*7, 7))))

    grid1 = [4, 11, 18, 21, 24, 31, 38]
    grid2 = grid1

    W_fine = np.zeros((Nx,Ny))

    for nx in range(7):
        for ny in range(7):

            W_fine[grid1[nx], grid2[ny]] = w[nx,ny];

            if (nx==3) & (ny==3) & (flag==1):
                W_fine[grid1[nx], grid2[ny]] = 0;

    return W_fine

#W_stat2 = W[:,:,range(0,43,7),:]
#W_stat3 = W_stat2[:,:,:,range(0,43,7)]
W_stat2 = W[:, :, [4, 11, 18, 21, 24, 31, 38], :]
W_stat3 = W_stat2[:, :, :, [4, 11, 18, 21, 24, 31, 38]]

flag = 1
dim = [43,43]
NF = 34

W1_stat = np.zeros((NF, NF, dim[0], dim[1]))

for f1 in range(NF):
    for f2 in range(NF):

        W1_stat[f1,f2,:,:] = construct_row4(W_stat3[f1, f2, :, :], dim, flag)

W_stat = W1_stat
np.save('/home/dvoina/simple_vids/results/W_43x43_34filters_static_simple_3px_ReviewComplete2.npy', W)
np.save('/home/dvoina/simple_vids/results/W_43x43_34filters_static_simple_3px_ReviewSparse2.npy', W_stat)

