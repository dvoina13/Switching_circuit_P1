from __future__ import print_function
import torch
import numpy as np
import scipy.io
import math
from os import listdir
import matplotlib.pyplot as plt
import os
import torch
import argparse
from scipy.stats import ks_2samp

parser = argparse.ArgumentParser()
parser.add_argument("--vid_path", type=str, default='/home/dvoina/simple_vids/moving_videos_bsr_jumps_simple_3px_valid',
                    required=False, help="Set path for image dataset")
parser.add_argument("--Nf2", type=int, default=5, required=False, help="number of VIP neurons/u.s.")
parser.add_argument("--tau", type=int, default=2, required=False, help="tau/synaptic delay")

args = parser.parse_args()
vid_path = args.vid_path
tau = args.tau

fr_pervid = 49  # how many frames are in a video after convolution w/ filters has been done!
Nf2 = args.Nf2

XTrain = np.load('/home/dvoina/simple_vids/relevant/XTrain_forPython_34filters_noNoise_reviewSimple.npy')
address = vid_path
folder_list = listdir(address)
no_folders = np.shape(folder_list)[0]

W_stat = np.load('/home/dvoina/simple_vids/results/W_43x43_34filters_static_simple_3px_ReviewSparse2.npy')
W_mov = np.load('/home/dvoina/simple_vids/results/W_43x43_34filters_moving_simple_3px_tau' + str(tau) + '_ReviewSparse.npy')

Wp2v_load = np.load('/home/dvoina/simple_vids/results/Wp2v_34_param_simple3px_sst2vip_Nf' + str(Nf2) + '_review2.npy')
Wv2p_load = np.load('/home/dvoina/simple_vids/results/Wv2p_34_param_simple3px_sst2vip_Nf' + str(Nf2) + '_review2.npy')
Wv2s_load = np.load('/home/dvoina/simple_vids/results/Wv2s_34_param_simple3px_sst2vip_Nf' + str(Nf2) + '_review2.npy')

Wp2v = Wp2v_load[-3, 2, :, :, :, :]
Wv2p = -Wv2p_load[-3, 2, :, :, :, :]
Wv2s = -Wv2s_load[-3, 2, :, :, :, :]

#Wp2v = Wp2v_load[-1, -3, :, :, :, :]
#Wv2p = -Wv2p_load[-1, -3, :, :, :, :]
#Wv2s = -Wv2s_load[-1, -3, :, :, :, :]

W_s2p = np.copy(W_stat)
W_s2p[W_s2p > 0] = 0

W_moving_minus_static = W_mov - W_stat

W_stat_matrix_torch = torch.from_numpy(W_stat).float()
W_mov_matrix_torch = torch.from_numpy(W_mov).float()
W_s2p_torch = torch.from_numpy(W_s2p).float()
W_moving_minus_static_torch = torch.from_numpy(W_moving_minus_static).float()
Wp2v_torch = torch.from_numpy(Wp2v).float()
Wv2p_torch = torch.from_numpy(Wv2p).float()
Wv2s_torch = torch.from_numpy(Wv2s).float()

folder_list = listdir(address)
no_folders = np.shape(folder_list)[0]

input_gaussian = torch.load("/home/dvoina/simple_vids/input_gaussian.npy")
input_constant = torch.load("/home/dvoina/simple_vids/input_constant.npy")
bar_H = torch.load("/home/dvoina/simple_vids/bar_H.npy")
bar_V = torch.load("/home/dvoina/simple_vids/bar_V.npy")
bar_iso_H = torch.load("/home/dvoina/simple_vids/bar_iso_H.npy")
bar_iso_V = torch.load("/home/dvoina/simple_vids/bar_iso_V.npy")
bar_cross_HV = torch.load("/home/dvoina/simple_vids/bar_cross_HV.npy")
bar_cross_VH = torch.load("/home/dvoina/simple_vids/bar_cross_VH.npy")

input_gaussian = torch.load("/home/dvoina/simple_vids/input_gaussian_static.npy")
input_constant = torch.load("/home/dvoina/simple_vids/input_constant_static.npy")

print("compute Activities static and Activities moving")

#no_folders = 1
A_static = np.zeros((no_folders,1,47,34,125,125))
A_moving = np.zeros((no_folders,1,47,34,125,125))
A_approx = np.zeros((no_folders,47,34,121,121))

for i in range(no_folders):
    print(i)
    #XTrain_main = XTrain[i*fr_pervid+2:i*fr_pervid+fr_pervid,0,:,:,:]
    #XTrain_prev = XTrain[i*fr_pervid:i*fr_pervid+fr_pervid-2,0,:,:,:]

    XTrain_main = XTrain[i*fr_pervid+tau:i*fr_pervid+fr_pervid, 0, :, :, :]
    XTrain_prev = XTrain[i*fr_pervid:i*fr_pervid+fr_pervid-tau, 0, :, :, :]

    #XTrain_main = input_gaussian[i*fr_pervid+tau:i*fr_pervid+fr_pervid,:,:]
    #XTrain_prev = input_gaussian[i*fr_pervid:i*fr_pervid+fr_pervid-tau,:,:]

    #XTrain_main = input_constant[i,:,:,:,:]
    #XTrain_prev = input_constant[i,:,:,:,:]

    #XTrain_main = bar_cross_VH
    #XTrain_prev = bar_cross_VH

    XTrain_main = torch.from_numpy(XTrain_main).squeeze(1)
    XTrain_prev = torch.from_numpy(XTrain_prev).squeeze(1)

    y_mov = torch.nn.functional.conv2d(XTrain_prev, W_mov_matrix_torch)
    y_stat = torch.nn.functional.conv2d(XTrain_main, W_stat_matrix_torch)

    A_moving2 = XTrain_main[:, :, 21:-21, 21:-21] * (
                torch.ones(XTrain_main.size()[0], XTrain_main.size()[1], XTrain_main.size()[2] - 42,
                XTrain_main.size()[3] - 42) + y_mov)
    A_static2 = XTrain_main[:, :, 21:-21, 21:-21] * (
                torch.ones(XTrain_main.size()[0], XTrain_main.size()[1], XTrain_main.size()[2] - 42,
                XTrain_main.size()[3] - 42) + y_stat)

    A_moving2 = A_moving2.numpy()
    A_static2 = A_static2.numpy()

    A_vip = torch.nn.functional.conv2d(XTrain_main, Wp2v_torch)

    A_moving2 = A_moving2[np.newaxis, :, :, :, :]
    A_static2 = A_static2[np.newaxis, :, :, :, :]
    Activities2 = np.concatenate((A_static2, A_moving2), axis=0)

    A_static[i,:,:,:,:,:] = A_static2
    A_moving[i,:,:,:,:,:] = A_moving2

    np.save('/home/dvoina/simple_vids/Activities_movstat_34filters_simple_tau' + str(tau) + "/"+ folder_list[i] + '_activities_noNoise.npy', Activities2)
    #np.save('/home/dvoina/simple_vids/XTrain_main_34filters_simple_tau' + str(tau) + '/'+ folder_list[i] + '_xmain_noNoise.npy', XTrain_main)
    np.save('/home/dvoina/simple_vids/Activities_vip_34_noNoise_new_from_fc/' + folder_list[i] + '_activities.npy', A_vip)

    del A_moving2, A_static2, Activities2, y_mov, y_stat, XTrain_main, XTrain_prev

print(np.mean(A_static))
print(np.mean(A_moving))

print(ks_2samp(A_static.flatten(), A_moving.flatten()))


print("compute approx Activities and Activities of VIP/SST and VIP/SST contributions")

for i in range(no_folders):
    print(i)

    XTrain_main = XTrain[i * fr_pervid + tau:i * fr_pervid + fr_pervid, 0, :, :, :]
    XTrain_prev = XTrain[i * fr_pervid:i * fr_pervid + fr_pervid - tau, 0, :, :, :]

    #XTrain_main = input_gaussian[i*fr_pervid+tau:i*fr_pervid+fr_pervid,:,:]
    #XTrain_prev = input_gaussian[i*fr_pervid:i*fr_pervid+fr_pervid-tau,:,:]

    #XTrain_main = bar_H
    #XTrain_prev = bar_H

    XTrain_main = torch.from_numpy(XTrain_main).squeeze(1)
    XTrain_prev = torch.from_numpy(XTrain_prev).squeeze(1)

    y_stat = torch.nn.functional.conv2d(XTrain_prev, W_stat_matrix_torch)

    h1 = torch.nn.functional.conv2d(XTrain_prev, Wp2v_torch)
    x1 = torch.nn.functional.conv2d(h1, Wv2p_torch)
    h2 = torch.nn.functional.conv2d(h1, Wv2s_torch)
    x2 = torch.nn.functional.conv2d(h2, W_s2p_torch)

    x1 = x1[:, :, 21:-21, 21:-21]

    A_approx2 = XTrain_main[:, :, 23:-23, 23:-23] * (
            torch.ones(XTrain_main.size()[0], XTrain_main.size()[1], XTrain_main.size()[2] - 46,
                       XTrain_main.size()[3] - 46) + y_stat[:, :, 2:-2, 2:-2] + x1 + x2)

    A_approx[i,:,:,:,:] = A_approx2

    """
    A_vip = torch.nn.functional.conv2d(A_approx2, Wp2v_torch)
    C_vip = torch.nn.functional.conv2d(A_vip, Wv2p_torch)

    A_sst = XTrain_main[:, :, 25:-25, 25:-25] + torch.nn.functional.conv2d(
        torch.nn.functional.conv2d(A_approx2, Wp2v_torch), Wv2s_torch)
    C_sst = torch.nn.functional.conv2d(A_sst, W_s2p_torch)
    """

    A_vip1 = torch.nn.functional.conv2d(A_approx2, Wp2v_torch)
    A_sst1 = XTrain_main[:, :, 25:-25, 25:-25] + torch.nn.functional.conv2d(
        torch.nn.functional.conv2d(A_approx2, Wp2v_torch), Wv2s_torch)

    np.save('/home/dvoina/simple_vids/Activities_vip_34_noNoise_new/' + folder_list[i] + '_activities.npy',
            A_vip1)
    np.save('/home/dvoina/simple_vids/Activities_approx_34_noNoise_new/' + folder_list[i] + '_activities_noNoise.npy',
            A_approx2.numpy())

    del A_approx2, XTrain_main, XTrain_prev, A_vip1

