#possibly change no_folders
#change weights
#change fn folders

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
import math

fr_pervid = 49 #19 #how many frames are in a video after convolution w/ filters has been done!
tau = 2 #1

N = 50 #20; %how many frames the video has

end_fr = 47 #18; %how many frames are there after looking in the past + doing filter (ie. how many frames does XTrain_Prep_forPython have?)
delta_t = 1 #how many frames the filter has-1?!

N_x = 167 #161
N_y = 167 #161

filters34_2 = loadmat('data_filts_px3_v2.mat')

filters34_temp = np.array(filters34_2['data_filts2'][0,0:16].tolist())
filters34_temp = np.expand_dims(filters34_temp, axis=0)
filters34_temp = np.transpose(filters34_temp, (0,1,4,2,3))

filters34_notemp = np.array(filters34_2['data_filts2'][0,16:34].tolist())
filters34_notemp = np.expand_dims(filters34_notemp, axis=1)
filters34_notemp = np.expand_dims(filters34_notemp, axis=0)

filters34_temp = np.transpose(np.transpose(filters34_temp, (0,2,3,4,1))-filters34_temp.reshape(1,filters34_temp.shape[1],-1).mean(axis=2), (0,4,1,2,3))
filters34_notemp = np.transpose(np.transpose(filters34_notemp, (0,2,3,4,1))-filters34_notemp.reshape(1,filters34_notemp.shape[1],-1).mean(axis=2), (0,4,1,2,3))

filter_len = np.shape(filters34_temp)[2]
NF_expl = [np.shape(filters34_temp)[1],np.shape(filters34_notemp)[1]]
NF = NF_expl[0] + NF_expl[1]

# 18filters, normal model, mov+stat
XTrain = np.load('/home/dvoina/simple_vids/repository_project1/XTrain_forPython_34filters_NoiseG2.5_reviewSimple.npy')
XTrain_real = np.load('/home/dvoina/simple_vids/repository_project1/XTrain_forPython_34filters_noNoise_reviewSimple.npy')

# 18filters, rnn model, approx
W_stat = np.load('/home/dvoina/simple_vids/results/W_43x43_34filters_static_simple_3px_ReviewSparse2.npy')
W_mov = np.load('/home/dvoina/simple_vids/results/W_43x43_34filters_moving_simple_3px_tau2_ReviewSparse.npy')

W_s2p = np.copy(W_stat); W_s2p[W_s2p>0] = 0

W_stat_matrix_torch = torch.from_numpy(W_stat).float()
W_mov_matrix_torch = torch.from_numpy(W_mov).float()
W_s2p_torch = torch.from_numpy(W_s2p).float()

Wp2v = np.load('/home/dvoina/simple_vids/results/Wp2v_34_param_simple3px_sst2vip_18_Nf5_review2.npy')
Wv2p = np.load('/home/dvoina/simple_vids/results/Wv2p_34_param_simple3px_sst2vip_18_Nf5_review2.npy')
Wv2s = np.load('/home/dvoina/simple_vids/results/Wv2s_34_param_simple3px_sst2vip_18_Nf5_review2.npy')

Wp2v = torch.from_numpy(np.squeeze(Wp2v[-3,2,:,:,:,:]))
Wv2p = torch.from_numpy(np.squeeze(Wv2p[-3,2,:,:,:,:]))
Wv2s = torch.from_numpy(np.squeeze(Wv2s[-3,2,:,:,:,:]))


dir_ref = '/home/dvoina/simple_vids/moving_videos_bsr_jumps_simple_3px_valid/'
#address = '/home/dvoina/simple_vids/moving_videos_bsr_jumps_simple_8px_test/'
address = '/home/dvoina/simple_vids/moving_videos_bsr_jumps_simple_3px_valid/'

folder_list = listdir(address)
no_folders = np.shape(folder_list)[0]

#######

c1 = 3; c2 = 3; c3 = 3;
n1 = int(math.floor(c1/2)); n2 = int(math.floor(c2/2)); n3 = int(math.floor(c3/2));
Nf = 34; Nf2 = 5

class Activities(torch.nn.Module):
    def __init__(self):
        super(Activities, self).__init__()
        self.conv_p2v = torch.nn.Conv2d(Nf, Nf2, c1, bias=None, padding=0)
        self.conv_v2s = torch.nn.Conv2d(Nf2, Nf, c2, bias=None, padding=0)
        self.conv_s2p = torch.nn.Conv2d(Nf, Nf, 43, bias=None, padding=0)  # conv for sure
        self.conv_v2p = torch.nn.Conv2d(Nf2, Nf, c3, bias=None, padding=0)
        self.conv_mov = torch.nn.Conv2d(Nf, Nf, 43, bias=None, padding=0)  # conv for sure
        self.conv_stat = torch.nn.Conv2d(Nf, Nf, 43, bias=None, padding=0)

    def forward(self, x, x_prev):

        x1 = self.conv_p2v(x_prev)
        y1 = -self.conv_v2p(x1)

        y2 = -self.conv_v2s(x1)
        y3 = self.conv_s2p(y2)

        xmov = self.conv_mov(x_prev)
        xstat = self.conv_stat(x_prev)

        if (n1+n3!=0):
            xmov = xmov[:,:,n3+n1:-n3-n1, n3+n1:-n3-n1]
            xstat = xstat[:,:,n3+n1:-n3-n1, n3+n1:-n3-n1]

        y1 = y1[:,:,21:-21, 21:-21]
        x = x[:,:,21+n3+n1:-21-n3-n1,21+n3+n1:-21-n3-n1]

        ones = torch.ones(x.size()[0], x.size()[1], x.size()[2], x.size()[3]).cuda()
        a_static = x*(ones + xstat)
        a_moving = x*(ones + xmov)

        xapprox = xstat + y1 + y3
        a_approx = x*(ones + xapprox)

        return a_static, a_moving, a_approx, xstat, xmov
        #return x*xstat, x*xmov, x*xapprox

class inverse(torch.nn.Module):
    def __init__(self):
        super(inverse, self).__init__()
        self.conv_inv = torch.nn.Conv2d(1, 1, 15, bias=None, padding=0)

    def forward(self, x):
        y = self.conv_inv(x)
        return y

cuda = torch.cuda.is_available()
model = Activities()
model.conv_s2p.weight.data.copy_(W_s2p_torch)
model.conv_mov.weight.data.copy_(W_mov_matrix_torch)
model.conv_stat.weight.data.copy_(W_stat_matrix_torch)
model.conv_p2v.weight.data.copy_(Wp2v)
model.conv_v2p.weight.data.copy_(Wv2p)
model.conv_v2s.weight.data.copy_(Wv2s)

inv = inverse()

if cuda:
    model.cuda()
    inv.cuda()

def mean2(x):
    y = np.sum(x) / np.size(x);
    return y

def corr2(a,b):
    a = a - mean2(a)
    b = b - mean2(b)

    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum());
    return r

#alpha_array = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5,100]
#no_folders = 1
alpha_array = [1]
Rtot_moving = np.zeros((np.shape(alpha_array)[0], no_folders, end_fr-delta_t))
R_moving = np.zeros((np.shape(alpha_array)[0], no_folders))
Rtot_static = np.zeros((np.shape(alpha_array)[0], no_folders, end_fr-delta_t))
R_static = np.zeros((np.shape(alpha_array)[0], no_folders))
Rtot_real = np.zeros((np.shape(alpha_array)[0], no_folders, end_fr-delta_t))
R_real = np.zeros((np.shape(alpha_array)[0], no_folders))
Rtot_ff = np.zeros((np.shape(alpha_array)[0], no_folders, end_fr-delta_t))
R_ff = np.zeros((np.shape(alpha_array)[0], no_folders))
Rtot_approx = np.zeros((np.shape(alpha_array)[0], no_folders, end_fr-delta_t))
R_approx = np.zeros((np.shape(alpha_array)[0], no_folders))

A_static2 = np.zeros((no_folders, end_fr, NF, N_x-42-2*n1-2*n3, N_y-42-2*n1-2*n3))
A_moving2 = np.zeros((no_folders, end_fr, NF, N_x-42-2*n1-2*n3, N_y-42-2*n1-2*n3))
A_approx2 = np.zeros((no_folders, end_fr, NF, N_x-42-2*n1-2*n3, N_y-42-2*n1-2*n3))
A_static_max = np.zeros((no_folders))
A_moving_max = np.zeros((no_folders))
A_approx_max = np.zeros((no_folders))

#for alpha in [0.5, 0.75, 1, 1.25, 1.5,100]:
for index, alpha in enumerate(alpha_array):
    print(alpha)
    for i in range(no_folders):
    #for j in range(1,2):
        #i=j
        print(i)
        filename = folder_list[i][:-4]

        XTrain_main = XTrain[i*fr_pervid+tau : i*fr_pervid + fr_pervid, 0, :, :, :]
        XTrain_prev = XTrain[i*fr_pervid : i*fr_pervid + fr_pervid - tau, 0, :, :, :]
        XTrain_main_real = XTrain_real[i * fr_pervid + tau: i * fr_pervid + fr_pervid, 0, :, :, :]

        XTrain_main = torch.from_numpy(XTrain_main).squeeze(1)
        XTrain_prev = torch.from_numpy(XTrain_prev).squeeze(1)
        XTrain_main_real = torch.from_numpy(XTrain_main_real).squeeze(1)

        if cuda:
            x = XTrain_main.cuda()
            x_prev = XTrain_prev.cuda()

        A_static, A_moving, A_approx, xstat, xmov = model(x, x_prev)
        A_static = A_static.cpu().detach().numpy(); A_moving = A_moving.cpu().detach().numpy(); A_approx = A_approx.cpu().detach().numpy();
        A_static[A_static<0] = 0; A_moving[A_moving<0] = 0; A_approx[A_approx<0] = 0;

        A_static_max[i] = np.max(xstat.cpu().detach().numpy()); A_moving_max[i] = np.max(xmov.cpu().detach().numpy()); A_approx_max[i] = np.max(A_approx);

        Recon_moving_initial = np.zeros((np.shape(A_moving)[0]+1, np.shape(A_moving)[1], XTrain_main.size()[2] - 42-2*n1-2*n3, XTrain_main.size()[3] - 42-2*n1-2*n3))
        Recon_static_initial = np.zeros((np.shape(A_static)[0]+1, np.shape(A_static)[1], XTrain_main.size()[2] - 42-2*n1-2*n3, XTrain_main.size()[3] - 42-2*n1-2*n3))
        Recon_approx_initial = np.zeros((np.shape(A_approx)[0]+1, np.shape(A_approx)[1], XTrain_main.size()[2] - 42-2*n1-2*n3, XTrain_main.size()[3] - 42-2*n1-2*n3))
        Recon_real_initial = np.zeros((np.shape(XTrain_main_real)[0]+1, np.shape(XTrain_main_real)[1], np.shape(XTrain_main_real)[2] - 42-2*n1-2*n3, np.shape(XTrain_main_real)[3] - 42-2*n1-2*n3))
        Recon_ff_initial = np.zeros((np.shape(XTrain_main)[0]+1, np.shape(XTrain_main)[1], np.shape(XTrain_main)[2] - 42-2*n1-2*n3, np.shape(XTrain_main)[3] - 42-2*n1-2*n3))

        for frame in range(end_fr):
            for f in range(NF):

                if f < NF_expl[0]:
                    inv_filt = torch.from_numpy(np.fliplr(np.flipud(filters34_temp[0,f,0,:,:])).copy()).unsqueeze(0).unsqueeze(0)
                    inv.conv_inv.weight.data.copy_(inv_filt.cuda())

                    activity = torch.from_numpy(np.pad(XTrain_main[frame, f, 21+n1+n3:-21-n1-n3, 21+n1+n3:-21-n1-n3], (7, 7), 'symmetric')).unsqueeze(0).unsqueeze(0)
                    Recon_ff_initial[frame, f, :, :] = inv(activity.cuda()).cpu().detach().numpy()
                    activity = torch.from_numpy(np.pad(A_moving[frame,f,:,:], (7,7), 'symmetric')).unsqueeze(0).unsqueeze(0).float()
                    Recon_moving_initial[frame,f,:,:] = inv(activity.cuda()).cpu().detach().numpy()
                    activity = torch.from_numpy(np.pad(A_static[frame, f, :, :], (7, 7), 'symmetric')).unsqueeze(0).unsqueeze(0)
                    Recon_static_initial[frame, f, :, :] = inv(activity.cuda()).cpu().detach().numpy()
                    activity = torch.from_numpy(np.pad(A_approx[frame, f, :, :], (7, 7), 'symmetric')).unsqueeze(0).unsqueeze(0)
                    Recon_approx_initial[frame, f, :, :] = inv(activity.cuda()).cpu().detach().numpy()
                    activity = torch.from_numpy(np.pad(XTrain_main_real[frame, f, 21+n1+n3:-21-n1-n3, 21+n1+n3:-21-n1-n3], (7, 7), 'symmetric')).unsqueeze(0).unsqueeze(0)
                    Recon_real_initial[frame, f, :, :] = inv(activity.cuda()).cpu().detach().numpy()
                else:
                    inv_filt = torch.from_numpy(np.fliplr(np.flipud(filters34_notemp[0, f - NF_expl[0], 0, :, :])).copy()).unsqueeze(0).unsqueeze(0)
                    inv.conv_inv.weight.data.copy_(inv_filt.cuda())

                    activity = torch.from_numpy(np.pad(XTrain_main[frame, f, 21+n1+n3:-21-n1-n3, 21+n1+n3:-21-n1-n3], (7, 7), 'symmetric')).unsqueeze(0).unsqueeze(0)
                    Recon_ff_initial[frame+delta_t, f, :, :] = inv(activity.cuda()).cpu().detach().numpy()
                    activity = torch.from_numpy(np.pad(A_moving[frame, f, :, :], (7, 7), 'symmetric')).unsqueeze(0).unsqueeze(0).float()
                    Recon_moving_initial[frame+delta_t,f,:,:] = inv(activity.cuda()).cpu().detach().numpy()
                    activity = torch.from_numpy(np.pad(A_static[frame, f, :, :], (7, 7), 'symmetric')).unsqueeze(0).unsqueeze(0)
                    Recon_static_initial[frame+delta_t, f, :, :] = inv(activity.cuda()).cpu().detach().numpy()
                    activity = torch.from_numpy(np.pad(A_approx[frame, f, :, :], (7, 7), 'symmetric')).unsqueeze(0).unsqueeze(0)
                    Recon_approx_initial[frame+delta_t, f, :, :] = inv(activity.cuda()).cpu().detach().numpy()
                    activity = torch.from_numpy(np.pad(XTrain_main_real[frame, f, 21+n1+n3:-21-n1-n3, 21+n1+n3:-21-n1-n3], (7, 7), 'symmetric')).unsqueeze(0).unsqueeze(0)
                    Recon_real_initial[frame+delta_t, f, :, :] = inv(activity.cuda()).cpu().detach().numpy()

        Recon_moving_final = np.mean(Recon_moving_initial[delta_t:-delta_t,:,:,:], axis = 1)
        #Recon_moving_final = np.sum(Recon_moving_initial[delta_t:-delta_t, :, :, :]*weights[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis], axis=2)
        Recon_moving_final = np.squeeze(Recon_moving_final)

        Recon_static_final = np.mean(Recon_static_initial[delta_t:-delta_t, :, :, :], axis=1)
        #Recon_static_final = np.sum(Recon_static_initial[delta_t:-delta_t, :, :, :]*weights[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis], axis=2)
        Recon_static_final = np.squeeze(Recon_static_final)

        Recon_approx_final = np.mean(Recon_approx_initial[delta_t:-delta_t, :, :, :], axis=1)
        Recon_approx_final = np.squeeze(Recon_approx_final)

        Recon_real_final = np.mean(Recon_real_initial[delta_t:-delta_t, :, :, :], axis=1)
        #Recon_real_final = np.sum(Recon_real_initial[delta_t:-delta_t, :, :, :]*weights[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis], axis=2)
        Recon_real_final = np.squeeze(Recon_real_final)

        Recon_ff_final = np.mean(Recon_ff_initial[delta_t:-delta_t, :, :, :], axis=1)
        #Recon_ff_final = np.sum(Recon_ff_initial[delta_t:-delta_t, :, :, :]*weights[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis], axis=2)
        Recon_ff_final = np.squeeze(Recon_ff_final)

        vid = loadmat(dir_ref+filename)
        video = vid['s_modified']

        r_ff = np.zeros((1,end_fr-delta_t))
        r_static = np.zeros((1,end_fr-delta_t))
        r_moving = np.zeros((1, end_fr - delta_t))
        r_real = np.zeros((1, end_fr-delta_t))
        r_approx = np.zeros((1, end_fr-delta_t))
        step = N - end_fr

        for frame in range(end_fr-delta_t):

            img = np.squeeze(video[frame+step, :, :])
            img = img - np.mean(img)

            if np.max(img) != 0:
                img = img/np.max(img)

            img = img[21+n1+n3:-21-n1-n3, 21+n1+n3:-21-n1-n3]
            #r_moving[0,frame] = corr2(img, np.squeeze(Recon_moving_final[frame,:,:]))
            #r_static[0,frame] = corr2(img, np.squeeze(Recon_static_final[frame, :, :]))
            #r_real[0,frame] = corr2(img, Recon_real_final[frame, :, :])
            #r_ff[0,frame] = corr2(img, Recon_ff_final[frame, :, :])
            #r_approx[0,frame] = corr2(img, Recon_approx_final[frame,:,:])

            r_ff[0, frame] = corr2(Recon_real_final[frame, :, :], Recon_ff_final[frame, :, :])
            r_moving[0, frame] = corr2(Recon_real_final[frame, :, :], Recon_moving_final[frame, :, :])
            r_static[0, frame] = corr2(Recon_real_final[frame, :, :], Recon_static_final[frame, :, :])
            r_approx[0, frame] = corr2(Recon_real_final[frame, :, :], Recon_approx_final[frame, :, :])

        #i = 0

        A_static2[i,:,:,:,:] = A_static
        A_moving2[i,:,:,:,:] = A_moving
        A_approx2[i,:,:,:,:] = A_approx

        Rtot_moving[index, i, :] = r_moving
        R_moving[index, i] = np.mean(r_moving)
        Rtot_static[index, i, :] = r_static
        R_static[index, i] = np.mean(r_static)
        Rtot_approx[index, i, :] = r_approx
        R_approx[index, i] = np.mean(r_approx)
        #Rtot_real[index, i, :] = r_real
        #R_real[index, i] = np.mean(r_real)
        Rtot_ff[index, i] = r_ff
        R_ff[index, i] = np.mean(r_ff)
        Rtot_approx[index, i] = r_approx
        R_approx[index, i] = np.mean(r_approx)

print(np.mean(R_ff[0,:]))
print(np.mean(R_static[0,:]))
print(np.mean(R_moving[0,:]))
print(np.mean(R_approx[0,:]))

Rtot = np.concatenate((Rtot_ff, Rtot_static, Rtot_moving, Rtot_approx), axis = 0)
np.save('Rtot_34filters_3px_G05_tau2i_'+filename+'.npy', Rtot)

Recon_real = Recon_real_final[np.newaxis,:,:,:]
Recon_ff = Recon_ff_final[np.newaxis,:,:,:]
Recon_static = Recon_static_final[np.newaxis,:,:,:]
Recon_moving = Recon_moving_final[np.newaxis,:,:,:]
Recon_approx = Recon_approx_final[np.newaxis,:,:,:]
Recon = np.concatenate((Recon_real, Recon_ff, Recon_static, Recon_moving, Recon_approx), axis = 0)

np.save('Recon'+filename+'_tau2i_G05_34.npy', Recon)

