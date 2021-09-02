import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pywt
import pywt.data
from scipy.stats import pearsonr
import pylab
from scipy.io import loadmat
import scipy.stats
import statsmodels.api as sm

A_vip = []
A_stat = []
A_mov = []
A_approx = []
f2_array = []
f3_array = []
f4_array = []

files_list = os.listdir("/home/dvoina/simple_vids/Activities_vip_34_noNoise_new_from_fc")
vid_path = '/home/dvoina/simple_vids/moving_videos_bsr_jumps_simple_3px_valid'
avgpool = torch.nn.AvgPool2d(5, stride=9, padding=0)

for j, file in enumerate(files_list):
    print(j)
    a = np.load("/home/dvoina/simple_vids/Activities_vip_34_noNoise_new_from_fc/" + file)
    a2 = a
    # /np.resize(np.max(np.abs(a), axis=1), (47,1,165,165))

    A_vip.append(a2)

# files_list = os.listdir("Activities_movstat_34filters_simple_tau2")
files_list = os.listdir("/home/dvoina/simple_vids/Activities_approx_34_noNoise_new")

for j, file in enumerate(files_list):
    print(j)
    A = np.load("/home/dvoina/simple_vids/Activities_approx_34_noNoise_new/" + file)
    A2 = A / np.resize(np.max(np.abs(A), axis=1), (47, 1, 121, 121))

    A_approx.append(A2)
    # A_stat.append(A[0,:,:,:,:])
    # A_mov.append(A[1,:,:,:,:])

files_list = os.listdir("/home/dvoina/simple_vids/Activities_movstat_34filters_simple_tau2")

for j, file in enumerate(files_list):
    print(j)
    A = np.load("/home/dvoina/simple_vids/Activities_movstat_34filters_simple_tau2/" + file)
    print(A.shape)
    A_stat_ = A[0, :, :, :, :] / np.resize(np.max(np.abs(A[0, :, :, :, :]), axis=1), (47, 1, 125, 125))
    A_mov_ = A[1, :, :, :, :] / np.resize(np.max(np.abs(A[1, :, :, :, :]), axis=1), (47, 1, 125, 125))

    A_stat.append(A_stat_)
    A_mov.append(A_mov_)

files_list = os.listdir("/home/dvoina/simple_vids/moving_videos_bsr_jumps_simple_3px_valid")
for j, file in enumerate(files_list):
    print(file)
    vid = loadmat(vid_path + '/' + file)

    for frame in range(47):
        coeffs2 = pywt.dwt2(vid["s_modified"][3 + frame, :, :], 'db4')
        f1, (f2, f3, f4) = coeffs2

        f2_avg = avgpool(torch.from_numpy(f2).unsqueeze(0)).squeeze()
        f3_avg = avgpool(torch.from_numpy(f3).unsqueeze(0)).squeeze()
        f4_avg = avgpool(torch.from_numpy(f4).unsqueeze(0)).squeeze()

        f2_array.append(f2_avg)
        f3_array.append(f3_avg)
        f4_array.append(f4_avg)

f2_array = torch.stack(f2_array)
f3_array = torch.stack(f3_array)
f4_array = torch.stack(f4_array)


f2_array = np.resize(np.array(f2_array), (4700, 10*10))
f3_array = np.resize(np.array(f3_array), (4700, 10*10))
f4_array = np.resize(np.array(f4_array), (4700, 10*10))

features_array = np.concatenate((f2_array, f3_array, f4_array), axis=1)

A_vip = np.array(A_vip)
A_vip = np.resize(A_vip, (4700, 5*165*165))
A_stat = np.array(A_stat)
A_stat = np.resize(A_stat, (4700, 34*125*125))
A_mov = np.array(A_mov)
A_mov = np.resize(A_mov, (4700, 34*125*125))
A_approx = np.array(A_approx)
A_approx = np.resize(A_approx, (4700,34,121,121))

"""
results = np.zeros((5,123, 123, 300))
ci_results = np.zeros((5,123,123, 300, 2))
pval_results = np.zeros((5,123,123, 300))

x = 0

for i in range(5):
    for s1 in range(x, x + 12):
        for s2 in range(x, x + 12):
            print(i, s1, s2)
            glm_binom = sm.GLM(np.squeeze(A_vip[:, i, s1, s2]), features_array, family=sm.families.Binomial())
            res = glm_binom.fit()

            results[i, s1 - x, s2 - x, :] = res.params
            ci_results[i, s1 - x, s2 - x, :, :] = res.conf_int(alpha=0.05, cols=None)
            pval_results[i, s1 - x, s2 - x, :] = res.pvalues

x = 0
results2 = np.zeros((34, 121, 121, 300))
ci_results2 = np.zeros((34, 121, 121, 300, 2))
pval_results2 = np.zeros((34, 121, 121, 300))

for i in range(34):
    for s1 in range(x + 2, x + 12):
        for s2 in range(x + 2, x + 12):
            print(i, s1, s2)
            glm_binom = sm.GLM(np.squeeze(A_approx[:, i, s1, s2]), features_array, family=sm.families.Binomial())
            # family=sm.families.binomial)
            res = glm_binom.fit()

            results2[i, s1 - 2 - x, s2 - 2 - x, :] = res.params
            ci_results2[i, s1 - 2 - x, s2 - 2 - x, :, :] = res.conf_int(alpha=0.05, cols=None)
            pval_results2[i, s1 - 2 - x, s2 - 2 - x, :] = res.pvalues

results_flat = np.reshape(results[:,1:11,1:11,:], (5*10*10, 300))
results2_flat = np.reshape(results2[:,0:10,0:10,:], (34*10*10, 300))

corr_coeff_regression = results_flat.dot(results2_flat.T) #vip vs pyr
"""



results = np.zeros((10000, 300))
ci_results = np.zeros((10000, 300, 2))
pval_results = np.zeros((10000, 300))

start = 0
end = start + 10000

for i in range(start, end + 1):
    print(i)

    glm_binom = sm.GLM(A_vip[:, i], features_array, family=sm.families.Poisson())
    res = glm_binom.fit()

    results[i - start, :] = res.params
    ci_results[i - start, :, :] = res.conf_int(alpha=0.05, cols=None)
    pval_results[i - start, :] = res.pvalues

plt.figure()
plt.plot(range(10000), np.mean(results[0:10000,0:100], axis=1))
plt.xlabel("neurons")
plt.savefig("avg_neurons_vs_horiz.pdf", format="pdf", bbox_inches="tight")
#plt.plot(range(10000), np.mean(ci_results[0:10000,0:100,0], axis=1))
#plt.plot(range(10000), np.mean(ci_results[0:10000,0:100,1], axis=1))


plt.figure()
plt.plot(range(10000), np.mean(results[0:10000,100:200], axis=1))
plt.xlabel("neurons")
plt.savefig("avg_neurons_vs_vert.pdf", format="pdf", bbox_inches="tight")

plt.figure()
plt.plot(range(10000), np.mean(results[0:10000,200:300], axis=1))
plt.xlabel("neurons")

print(np.mean(np.abs(results[:,0:100])))
print(np.mean(np.abs(results[:,100:200])))
print(np.mean(np.abs(results[:,200:300])))

res1 = results[:,0:100]
res2 = results[:,0:200]
res3 = results[:,0:300]

res1 = res1.flatten()
res1_pos = res1[res1>=0]
res1_neg = res1[res1<=0]
print("res1", np.mean(res1_pos), np.mean(res1_neg))

res2 = res2.flatten()
res2_pos = res2[res2>=0]
res2_neg = res2[res2<=0]
print("res2", np.mean(res2_pos), np.mean(res2_neg))

res3 = res3.flatten()
res3_pos = res3[res3>=0]
res3_neg = res3[res3<=0]
print("res3", np.mean(res3_pos), np.mean(res3_neg))


print(np.mean(results[:,0:100]))
print(np.mean(results[:,100:200]))
print(np.mean(results[:,200:300]))

plt.bar([0, 1, 2], [np.mean(results[:,0:100]), np.mean(results[:,100:200]), np.mean(results[:,200:300])])
plt.savefig("compare_horiz_vert.pdf", format="pdf", bbox_inches="tight")



mean_f1 = np.mean(results[:,0:100], axis=0)
mean_f2 = np.mean(results[:,100:200], axis=0)
mean_f3 = np.mean(results[:,200:300], axis=0)

#plt.figure()
plt.plot(range(100), mean_f1)
plt.xlabel("features")
#plt.figure()
plt.plot(range(100), mean_f2)
plt.xlabel("features")
#plt.figure()
plt.plot(range(100), mean_f3)
plt.xlabel("features")
plt.legend(["horizontal", "vertical", "diagonal"])

plt.savefig("compare_features_vs_H_V_D.pdf", format="pdf", bbox_inches="tight")


plt.scatter(np.mean(results[:,0:100], axis=1), np.mean(results[:,100:200], axis=1))
plt.plot(np.arange(-0.15,0.5, 0.01), np.arange(-0.15,0.5, 0.01))
plt.savefig("scatter_plot_H_vs_V.pdf", format="pdf", bbox_inches="tight")

