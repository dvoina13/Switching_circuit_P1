import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
from os import listdir
from scipy.io import loadmat # for loading mat files
import numpy as np
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--my_path", type=str, default='/home/dvoina/simple_vids/moving_videos_bsr_jumps_simple_3px_train/', required=False, help="Set path for video frames")
parser.add_argument("--tau", type=int, default=1, required=False, help="set tau/synaptic delay")

args = parser.parse_args()
mypath = args.my_path

tau = args.tau
print("tau", tau)

filters34_2 = loadmat('data_filts_px3_v2.mat')

filters34_temp = np.array(filters34_2['data_filts2'][0,0:16].tolist())
filters34_temp = np.expand_dims(filters34_temp, axis=0)
filters34_temp = np.transpose(filters34_temp, (0,1,4,2,3))

filters34_notemp = np.array(filters34_2['data_filts2'][0,16:34].tolist())
filters34_notemp = np.expand_dims(filters34_notemp, axis=1)
filters34_notemp = np.expand_dims(filters34_notemp, axis=0)

filter_len = np.shape(filters34_temp)[2]

# Let's zero mean the filters (make use of numpy broadcasting)
filters34_temp = np.transpose(np.transpose(filters34_temp, (0,2,3,4,1))-filters34_temp.reshape(1,filters34_temp.shape[1],-1).mean(axis=2), (0,4,1,2,3))
filters34_notemp = np.transpose(np.transpose(filters34_notemp, (0,2,3,4,1))-filters34_notemp.reshape(1,filters34_notemp.shape[1],-1).mean(axis=2), (0,4,1,2,3))

def ToGrayscale(sample):
        sample = np.asarray(sample)
        #sample = sample.sum(axis=-1)  # sum over last axis
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

    def forward(self, x, x_prev):
        # Define spatial extent of correlations
        corr_x = self.num_rfs * self.filts_temp.shape[3]
        corr_y = self.num_rfs * self.filts_temp.shape[4]
        num_filts = self.filts_temp.shape[0] + self.filts_notemp.shape[0]

        # Convolve filters with input image
        # x = F.relu(F.conv2d(x, self.filts))
        x_temp = F.relu(F.conv3d(x, self.filts_temp))/2
        x_notemp = F.relu(F.conv3d(x[:, :, x.size()[2]-1, :, :].unsqueeze(2), self.filts_notemp))
        x = torch.cat((x_temp, x_notemp), dim=1)

        x_prev_temp = F.relu(F.conv3d(x_prev, self.filts_temp))/2
        x_prev_notemp = F.relu(F.conv3d(x_prev[:, :, x_prev.size()[2]-1, :, :].unsqueeze(2), self.filts_notemp))
        x_prev = torch.cat((x_prev_temp, x_prev_notemp), dim=1)

        # Normalization with added eps in denominator
        x1 = torch.div(x, torch.sum(x, dim=1).unsqueeze(1) + np.finfo(float).eps)
        x1_prev = torch.div(x_prev, torch.sum(x_prev, dim=1).unsqueeze(1) + np.finfo(float).eps)

        # Get dimensions of the image
        x_max = x1.size()[4]
        y_max = x1.size()[3]

        x1_filts = x1[:, :, :, corr_y:y_max - corr_y, corr_x:x_max - corr_x].contiguous().view(num_filts, 1,1, y_max - 2 * corr_y, x_max - 2 * corr_x)  #select subset

        x1_prev = x1_prev.view(1,1,num_filts,y_max,x_max)
        x2 = F.conv3d(x1_prev, x1_filts, groups=1)
        x2 = x2.squeeze().view(1, num_filts, num_filts, 2 * corr_y + 1, 2 * corr_x + 1)

        # We are using a 231x391 size filter
        x2 = torch.div(x2, (y_max - 2 * corr_y) * (x_max - 2 * corr_y))  # normalize by size of filter

        return x1, x2

def train(tau, filter_len):
    model.eval()

    mypath = '/home/dvoina/simple_vids/moving_videos_bsr_jumps_simple_3px_train/'
    onlyfiles = [f for f in listdir(mypath)]

    fn_array = torch.zeros(34)
    fnn_array = torch.zeros(34,34,43,43)
    total = 0

    for i in range(len(onlyfiles)):
    #for i in range(1):
        print(i, onlyfiles[i])

        video = loadmat('/home/dvoina/simple_vids/moving_videos_bsr_jumps_simple_3px_train/' + onlyfiles[i])
        video = video["s_modified"]

        data_dict = {}
        # Put all BSDS images in the 'train' folder in the current working directory
        for batch_idx in range(np.shape(video)[0]):

            data = video[batch_idx, :, :]
            data = ToGrayscale(data)
            data = torch.from_numpy(data).float()

            #if (batch_idx >= 2):
            if (batch_idx >= filter_len-1+tau):

                #Data = np.zeros((1, 1, 2, np.shape(data)[0], np.shape(data)[1]))
                Data = np.zeros((1, 1, filter_len, np.shape(data)[0], np.shape(data)[1]))
                #for j in range(1, 2, 1):
                for j in range(tau, tau+filter_len-1, 1):
                    #Data[0, 0, j - 1, :, :] = data_dict[str(j)]
                    Data[0,0,j-tau,:,:] = data_dict[str(j)]
                Data[0, 0, filter_len-1, :, :] = data

                Data_prev = np.zeros((1, 1, filter_len, np.shape(data)[0], np.shape(data)[1]))
                for j in range(filter_len):
                    Data_prev[0, 0, j, :, :] = data_dict[str(j)]

                Data = torch.from_numpy(Data).float()
                Data_prev = torch.from_numpy(Data_prev).float()

                if cuda:
                    Data = Data.cuda()
                    Data_prev = Data_prev.cuda()

                with torch.no_grad():
                    Data = Variable(Data)  # convert into pytorch variables
                    Data_prev = Variable(Data_prev)

                    x1, x2 = model(Data, Data_prev)  # forward inference

                    # f_n's
                    x1 = x1.view(1, 34, x1.size()[3], x1.size()[4])
                    filt_avgs = torch.mean(x1.data.view(1, x1.shape[1], -1), dim=2).squeeze()
                    fn_array += filt_avgs.cpu()  # load back to cpu and convert to numpy

                    # f_nn's
                    grid_space = 1  # can also choose 7 (original)
                    x2_subset = x2[:, :, :, (45 - 21):(45 + 21 + 1):grid_space, (45 - 21):(
                                45 + 21 + 1):grid_space].data.squeeze()  # Python doesn't include end, so add 1
                    fnn_array += x2_subset.cpu()

                    total += 1
                for j in range(filter_len-1+tau-1):
                    data_dict[str(j)] = data_dict[str(j+1)]

                j = filter_len-1+tau-1
                data_dict[str(j)] = data
                #data_dict['0'] = data_dict['1']
                #data_dict['1'] = data
            else:
                data_dict[str(batch_idx)] = data

    fn_array = fn_array/total
    fnn_array = fnn_array/total
    print("total", total)

    return np.asarray(fn_array), np.asarray(fnn_array)

# Training settings
cuda = torch.cuda.is_available() # disables using the GPU and cuda if False
batch_size = 1 # input batch size for training (TODO: figure out how to group images with similar orientation)

# Create a new instance of the network
model = Net(filters34_temp, filters34_notemp, num_rfs=3)
if cuda:
    model.cuda()

# Use a list for saving the activations for each image

filt_avgs_images, fnn_avgs_images = train(tau, filter_len)

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
    grid1 = [4, 11, 18, 24, 31, 38]
    grid2 = grid1

    W_fine = np.zeros((Nx,Ny))

    for nx in range(7):
        for ny in range(7):

            W_fine[grid1[nx], grid2[ny]] = w[nx,ny]

            if (nx==3) & (ny==3) & (flag==1):
                W_fine[grid1[nx], grid2[ny]] = 0

    return W_fine

Ny = 43
center2 = int(math.floor(Ny/2))

#grid1 = np.concatenate((np.array(range(center2-3*7, center2, 7)), np.array(range(center2, center2+4*7, 7))))
#grid2 = np.concatenate((np.array(range(center2-3*7, center2, 7)), np.array(range(center2, center2+4*7, 7))))

W_mov2 = W[:, :, [4, 11, 18, 24, 31, 38], :]
W_mov3 = W_mov2[:, :, :, [4, 11, 18, 24, 31, 38]]
#W_mov2 = W[:, :, grid1, :]
#W_mov3 = W_mov2[:, :, :, grid2]

flag = 1
dim = [43,43]
NF = 34

W1_mov = np.zeros((NF, NF, dim[0], dim[1]))

for f1 in range(NF):
    for f2 in range(NF):

        W1_mov[f1,f2,:,:] = construct_row4(W_mov3[f1, f2, :, :], dim, flag);

W_mov = W1_mov

np.save('/home/dvoina/simple_vids/results/W_43x43_34filters_moving_simple_3px_tau'+str(tau)+'_ReviewComplete.npy', W)
np.save('/home/dvoina/simple_vids/results/W_43x43_34filters_moving_simple_3px_tau' + str(tau)+ '_ReviewSparse.npy', W_mov)
