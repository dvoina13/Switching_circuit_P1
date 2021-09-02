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

def ToGrayscale(sample):
        # Convert to numpy array and then make the image grayscale
        sample = np.asarray(sample)
        #sample = sample.sum(axis=-1)  # sum over last axis
        sample = sample - np.mean(sample)
        sample = sample / np.max(sample)  # divide by max over the image
        sample = np.pad(sample, (7, 7), 'symmetric')  # pad with symmetric borders
        return sample

class Net(nn.Module):
    def __init__(self, filters_temp, filters_notemp, noise_val, num_rfs):
        super(Net, self).__init__()
        # Convert set of numpy ndarray filters into Pytorch
        self.filts_temp = nn.Parameter(torch.from_numpy(filters_temp).permute(1, 0, 2, 3, 4).float(), requires_grad=False)
        self.filts_notemp = nn.Parameter(torch.from_numpy(filters_notemp).permute(1, 0, 2, 3, 4).float(), requires_grad=False)
        self.num_rfs = num_rfs  #Number of RFs to look out for correlations
        self.noise_val = noise_val

    def forward(self, x):
        # Define spatial extent of correlations
        corr_x = self.num_rfs * self.filts_temp.shape[3]
        corr_y = self.num_rfs * self.filts_temp.shape[4]
        num_filts = self.filts_temp.shape[0] + self.filts_notemp.shape[0]

        # Convolve filters with input image
        # x = F.relu(F.conv2d(x, self.filts))
        x_temp = F.conv3d(x, self.filts_temp)/2
        x_temp = F.relu(x_temp + self.noise_val*torch.from_numpy(np.random.randn(x_temp.size()[0], x_temp.size()[1], x_temp.size()[2], x_temp.size()[3], x_temp.size()[4])).float().cuda())
        x_notemp = F.conv3d(x[:, :, x.size()[2]-1, :, :].unsqueeze(2), self.filts_notemp)
        x_notemp = F.relu(x_notemp + self.noise_val*torch.from_numpy(np.random.randn(x_notemp.size()[0], x_notemp.size()[1], x_notemp.size()[2], x_notemp.size()[3], x_notemp.size()[4])).float().cuda())
        x = torch.cat((x_temp, x_notemp), dim=1)

        # Normalization with added eps in denominator
        x1 = torch.div(x, torch.sum(x, dim=1).unsqueeze(1) + np.finfo(float).eps)

        return x1

def train(NF, filter_len):
    model.eval()

    mypath = '/home/dvoina/ramsmatlabprogram/moving_videos_bsr/'
    onlyfiles = [f for f in listdir(mypath)]

    for i in range(len(onlyfiles)):

        print(i,onlyfiles[i])
      
        video = loadmat(mypath + onlyfiles[i])
        video = video["s_modified"]

        data_dict = {}
        # Put all BSDS images in the 'train' folder in the current working directory
        for batch_idx in range(np.shape(video)[0]):
            data = video[batch_idx, :, :]
            data = ToGrayscale(data)

            if (batch_idx>=filter_len-1):
                # Extract out image portion of data
                Data = np.zeros((1, 1, filter_len, np.shape(data)[0], np.shape(data)[1]))
                for j in range(filter_len-1):
                    Data[0, 0, j, :, :] = data_dict[str(j)]
                Data[0, 0, filter_len-1, :, :] = data

                Data = torch.from_numpy(Data).float()

                if cuda:
                    Data = Data.cuda()

                with torch.no_grad():
                    Data = Variable(Data)  # convert into pytorch variables

                    x1 = model(Data)  # forward inference
                    x1 = x1.view(1,NF,x1.size()[3],x1.size()[4])
                
                    fn_array.append(x1.cpu().numpy())  # load back to cpu and convert to numpyq


                for j in range(filter_len-2):
                    data_dict[str(j)] = data_dict[str(j+1)]
                j = filter_len - 2
                data_dict[str(j)] = data
            else:
                data_dict[str(batch_idx)] = data

    return np.asarray(fn_array)

filters34_2 = loadmat('data_filts_px3_v2.mat')

filters34_temp = np.array(filters34_2['data_filts2'][0,0:16].tolist())
#filters34_temp = np.array(filters34_2['data_filts2'][0,0:8].tolist())
filters34_temp = np.expand_dims(filters34_temp, axis=0)
filters34_temp = np.transpose(filters34_temp, (0,1,4,2,3))

filters34_notemp = np.array(filters34_2['data_filts2'][0,16:34].tolist())
#filters34_notemp = np.array(filters34_2['data_filts2'][0,8:24].tolist())
filters34_notemp = np.expand_dims(filters34_notemp, axis=1)
filters34_notemp = np.expand_dims(filters34_notemp, axis=0)

filter_len = np.shape(filters34_temp)[2]

# Let's zero mean the filters (make use of numpy broadcasting)
filters34_temp = np.transpose(np.transpose(filters34_temp, (0,2,3,4,1))-filters34_temp.reshape(1,filters34_temp.shape[1],-1).mean(axis=2), (0,4,1,2,3))
filters34_notemp = np.transpose(np.transpose(filters34_notemp, (0,2,3,4,1))-filters34_notemp.reshape(1,filters34_notemp.shape[1],-1).mean(axis=2), (0,4,1,2,3))

NF = np.shape(filters34_temp)[1] + np.shape(filters34_notemp)[1]

# Training settings
cuda = torch.cuda.is_available() # disables using the GPU and cuda if False
batch_size = 1 # input batch size for training (TODO: figure out how to group images with similar orientation)

# Create a new instance of the network
model = Net(filters34_temp, filters34_notemp, noise_val = 0, num_rfs=3)

if cuda:
    model.cuda()

fn_array = []
filt_avgs = train(NF, filter_len)
XTrain = filt_avgs

np.save("XTrain_forPython_34filters_noNoise_reviewSimple.npy", XTrain)
