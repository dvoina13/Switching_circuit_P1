import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.io import loadmat

Nf2 = 5

#load W_static, load W_moving
W_stat = np.load('/home/dvoina/simple_vids/results/W_43x43_34filters_static_simple_3px_ReviewSparse1.npy')
tau=1
W_mov = np.load('/home/dvoina/simple_vids/results/W_43x43_34filters_moving_simple_3px_tau' + str(tau) + '_ReviewSparse.npy')

#load W_v2p, W_v2s, W_p2v
Wp2v_load = np.load('/home/dvoina/simple_vids/results/Wp2v_34_param_simple3px_sst2vip_18_Nf' + str(Nf2) + '_review2.npy')
Wv2p_load = np.load('/home/dvoina/simple_vids/results/Wv2p_34_param_simple3px_sst2vip_18_Nf' + str(Nf2) + '_review2.npy')
Wv2s_load = np.load('/home/dvoina/simple_vids/results/Wv2s_34_param_simple3px_sst2vip_18_Nf' + str(Nf2) + '_review2.npy')

Wp2v = torch.from_numpy(Wp2v_load[-3,2,:,:,:,:]).float()
Wv2p = torch.from_numpy(Wv2p_load[-3,2,:,:,:,:]).float()
Wv2s = torch.from_numpy(Wv2s_load[-3,2,:,:,:,:]).float()

#compare mean of W_vip->sst and W_vip->pyr
plt.figure()
plt.bar([1,2], [-np.mean(Wv2s_load[-3,2,:,:,:,:]), -np.mean(Wv2p_load[-3,2,:,:,:,:])])
plt.xticks([])

#plot histograms of weights to and from VIP
plt.figure()
plt.hist(Wv2p_load[-3,2,:,:,:,:].flatten(), bins=100)
plt.savefig("Wv2p_sst2vip_Wv2p.pdf", format="pdf", bbox_inches="tight")

plt.figure()
plt.hist(Wv2s_load[-3,2,:,:,:,:].flatten(), bins=100)
plt.savefig("Wv2s_sst2vip_Wv2s.pdf", format="pdf", bbox_inches="tight")

plt.figure()
plt.hist(Wp2v_load[-3,2,:,:,:,:].flatten(), bins=100)

def compute_conv_product():
    w_product = torch.zeros(34, 34, 5, 5)
    xx = np.array([-1, 0, 1])
    yy = np.array([[-1, 0, 1]])
    ss = xx + yy.T

    for i in range(34):
        for k in range(34):

            for x in range(-1, 2):
                for y in range(-1, 2):

                    s = x + y
                    x_coord = np.where(ss == s)[0]
                    y_coord = np.where(ss == s)[1]

                    for ind in range(len(x_coord)):
                        w_product[i, k, x + 2, y + 2] += Wv2p[i, :, x_coord[ind], y_coord[ind]].dot(
                            Wp2v[:, k, x_coord[ind], y_coord[ind]])

    return w_product


#consider the convolution product W_p2v2p = W_v2p * W_p2v
#take averages over spatial dimension, over pre-, post-synaptic filters

w_product = compute_conv_product()

w_product_avg_SPavg = w_product.numpy().mean(axis=(2,3))
w_product_avg_pre = w_product.numpy().mean(axis=(1,2,3))
w_product_avg_post = w_product.numpy().mean(axis=(0,2,3))

mean_conn_pyr_v2p_pre = Wv2p_load[-3,2,:,:,:,:].mean(axis=(1,2,3))
mean_conn_pyr_v2s_pre = Wv2s_load[-3,2,:,:,:,:].mean(axis=(1,2,3))
mean_conn_pyr_p2v_pre = Wp2v_load[-3,2,:,:,:,:].mean(axis=(0,2,3))

mean_conn_pyr_v2p_SPavg = Wv2p_load[-3,2,:,:,:,:].mean(axis=(2,3))
mean_conn_pyr_v2s_SPavg = Wv2s_load[-3,2,:,:,:,:].mean(axis=(2,3))

mean_conn_vip_v2p_post = Wv2p_load[-3,2,:,:,:,:].mean(axis=(0,2,3))
mean_conn_vip_v2s_post = Wv2s_load[-3,2,:,:,:,:].mean(axis=(0,2,3))
mean_conn_vip_p2v_post = Wp2v_load[-3,2,:,:,:,:].mean(axis=(1,2,3))

mean_conn_pyr_mov_pre = W_mov.mean(axis=(1,2,3))
mean_conn_pyr_mov_post = W_mov.mean(axis=(0,2,3))
mean_conn_pyr_stat_pre = W_stat.mean(axis=(1,2,3))
mean_conn_pyr_stat_post = W_stat.mean(axis=(0,2,3))
mean_conn_pyr_mov_SPavg = W_mov.mean(axis=(2,3))
mean_conn_pyr_stat_SPavg = W_stat.mean(axis=(2,3))

mean_conn_mov_stat_pre = (W_mov - W_stat).mean(axis=(1,2,3))
mean_conn_mov_stat_SPavg = (W_mov - W_stat).mean(axis=(2,3))
mean_conn_mov_stat_post = (W_mov - W_stat).mean(axis=(0,2,3))

#relevant plots (from paper)
plt.figure()
plt.plot(range(34), mean_conn_pyr_v2p_pre)
plt.xlabel("filters")
plt.ylabel("avg connection strength")
plt.title("VIP")
plt.figure()
plt.plot(range(34), mean_conn_pyr_v2s_pre)
plt.xlabel("filters")
plt.ylabel("avg connection strength")
plt.title("VIP->SST")
plt.figure()
plt.plot(range(34), mean_conn_mov_stat_post)
plt.xlabel("filters")
plt.ylabel("avg connection strength")
plt.figure()
plt.plot(range(34), -w_product_avg_post, "-*")

plt.figure()
plt.plot(range(34), -mean_conn_pyr_v2p_pre, "-*")
plt.plot(range(34), -mean_conn_pyr_v2s_pre, "-*")
plt.ylim([-0.8, 0])
plt.legend(["v2p", "v2s"])
plt.savefig("compare_weights_v2p_v2s.pdf", format="pdf", bbox_inches="tight")

plt.figure()
plt.plot(range(34), mean_conn_mov_stat_post)
plt.title("W moving")

#infer relevant correlations
print("correlation")
print(pearsonr(mean_conn_mov_stat_post.flatten(), -mean_conn_pyr_v2p_pre.flatten()))
print(pearsonr(mean_conn_mov_stat_post.flatten(), -mean_conn_pyr_v2s_pre.flatten()))

#display most relevant filters

filters34_2 = loadmat('data_filts_px3_v2.mat')

filters34_temp = np.array(filters34_2['data_filts2'][0,0:16].tolist())
filters34_temp = np.expand_dims(filters34_temp, axis=0)
filters34_temp = np.transpose(filters34_temp, (0,1,4,2,3))

filters34_notemp = np.array(filters34_2['data_filts2'][0,16:34].tolist())
filters34_notemp = np.expand_dims(filters34_notemp, axis=1)
filters34_notemp = np.expand_dims(filters34_notemp, axis=0)

#CONCLUSION: most negative inhibitory filters have a vertical/diagonal bias; in top 10 there are 2H filters;
#4V filters

print(np.sort(-w_product_avg_post))
print(np.argsort(-w_product_avg_post))

plt.figure()
plt.imshow(filters34_temp[0,12,0,:,:])
plt.axis("off")
plt.savefig("filter_lowest1-0_w_product.pdf", format="pdf", bbox_inches="tight")

print(np.sort(mean_conn_mov_stat_post))
print(np.argsort(mean_conn_mov_stat_post))

plt.figure()
plt.imshow(filters34_notemp[0,16-16,0,:,:])
plt.axis("off")

print(np.sort(-mean_conn_pyr_v2p_pre))
ind_arr = np.argsort(-mean_conn_pyr_v2p_pre)
print(ind_arr)

plt.figure()
plt.imshow(filters34_temp[0,3,0,:,:])
plt.axis("off")


print(np.sort(mean_conn_pyr_mov_post))
print(np.argsort(mean_conn_pyr_mov_post))

print("average negative moving weight for top 3 negative is: ", mean_conn_pyr_mov_post)
plt.figure()
plt.imshow(filters34_notemp[0,25-16,0,:,:])
plt.axis("off")



vertical_f_spt = np.array([3, 4, 11, 12])
horizontal_f_spt = np.array([1, 6, 9, 14])
diagonal_f_spt = np.array([0, 2, 5, 7, 8, 10, 13, 15])

vertical_f_sp = np.array([5,6,13,14])
horizontal_f_sp = np.array([3,8,11,16])
diagonal_f_sp = np.array([2,4,7,9,10,12,15,17])
oneBlob_f = np.array([0,1])

print("mean V for -w_product_avg_post")
print((np.mean(-w_product_avg_post[vertical_f_sp]) + np.mean(-w_product_avg_post[vertical_f_spt])) / 2,
      np.mean(-w_product_avg_post[vertical_f_sp]), np.mean(-w_product_avg_post[vertical_f_spt]))
print("mean H for -w_product_avg_post")
print((np.mean(-w_product_avg_post[horizontal_f_sp]) + np.mean(-w_product_avg_post[horizontal_f_spt])) / 2,
      np.mean(-w_product_avg_post[horizontal_f_sp]), np.mean(-w_product_avg_post[horizontal_f_spt]))

print("mean V for mean_conn_mov_stat_post")
print((np.mean(mean_conn_mov_stat_post[vertical_f_sp]) + np.mean(mean_conn_mov_stat_post[vertical_f_spt])) / 2,
      np.mean(mean_conn_mov_stat_post[vertical_f_sp]), np.mean(mean_conn_mov_stat_post[vertical_f_spt]))
print("mean H for mean_conn_mov_stat_post")
print((np.mean(mean_conn_mov_stat_post[horizontal_f_sp]) + np.mean(mean_conn_mov_stat_post[horizontal_f_spt])) / 2,
      np.mean(mean_conn_mov_stat_post[horizontal_f_sp]), np.mean(mean_conn_mov_stat_post[horizontal_f_spt]))

print("mean V for -mean_conn_pyr_v2p")
print((np.mean(-mean_conn_pyr_v2p_pre[vertical_f_sp]) + np.mean(-mean_conn_pyr_v2p_pre[vertical_f_spt])) / 2,
      np.mean(-mean_conn_pyr_v2p_pre[vertical_f_sp]), np.mean(-mean_conn_pyr_v2p_pre[vertical_f_spt]))
print("mean H for -mean_conn_pyr_v2p")
print((np.mean(-mean_conn_pyr_v2p_pre[horizontal_f_sp]) + np.mean(-mean_conn_pyr_v2p_pre[horizontal_f_spt])) / 2,
      np.mean(-mean_conn_pyr_v2p_pre[horizontal_f_sp]), np.mean(-mean_conn_pyr_v2p_pre[horizontal_f_spt]))
from scipy import stats

tStat, pValue = stats.ttest_ind(-mean_conn_pyr_v2p_pre[vertical_f_sp], -mean_conn_pyr_v2p_pre[horizontal_f_sp], equal_var=False)
print("tstat, pval", tStat, pValue)
tStat, pValue = stats.ttest_ind(-mean_conn_pyr_v2p_pre[vertical_f_spt], -mean_conn_pyr_v2p_pre[horizontal_f_spt],
                                equal_var=False)
print("tstat, pval", tStat, pValue)

print("mean V for W moving")
print((np.mean(mean_conn_pyr_mov_post[vertical_f_sp]) + np.mean(mean_conn_pyr_mov_post[vertical_f_spt])) / 2,
      np.mean(mean_conn_pyr_mov_post[vertical_f_sp]), np.mean(mean_conn_pyr_mov_post[vertical_f_spt]))
print("mean H for W moving")
print((np.mean(mean_conn_pyr_mov_post[horizontal_f_sp]) + np.mean(mean_conn_pyr_mov_post[horizontal_f_spt])) / 2,
      np.mean(mean_conn_pyr_mov_post[horizontal_f_sp]), np.mean(mean_conn_pyr_mov_post[horizontal_f_spt]))

print("mean V for W static")
print((np.mean(mean_conn_pyr_stat_post[vertical_f_sp]) + np.mean(mean_conn_pyr_stat_post[vertical_f_spt])) / 2,
      np.mean(mean_conn_pyr_stat_post[vertical_f_sp]), np.mean(mean_conn_pyr_stat_post[vertical_f_spt]))
print("mean H for W static")
print((np.mean(mean_conn_pyr_stat_post[horizontal_f_sp]) + np.mean(mean_conn_pyr_stat_post[horizontal_f_spt])) / 2,
      np.mean(mean_conn_pyr_stat_post[horizontal_f_sp]), np.mean(mean_conn_pyr_stat_post[horizontal_f_spt]))

# CONCLUSIONS: V filters are NOT inhibited on average for W moving, W static (on the contrary there is excitation!),
# BUT they are for W VIP-PYR and W_moving-W_static, which proves the moving context indeed changes the frequency
# of vertical features (i.e. fewer vertical features)

