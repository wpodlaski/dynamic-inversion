# linear regression example

import numpy as np
import matplotlib.pyplot as plt
from sim_tools import backprop_error, get_maxreal_eig, SSA_opt, get_mat_angle, get_vec_angle


##########################
# 1 hyper-parameters
##########################
model_name = ['BP', 'FA', 'PBP', 'NDI', 'NDI-2', 'NDI-3', 'DI', 'DI-2', 'DI-3']
model_type = ['BP', 'FA', 'PBP', 'NDI', 'NDI', 'NDI', 'DI', 'DI', 'DI']
fback_type = ['n/a', 'random', 'n/a', 'neg-transpose', 'neg-transpose', 'optim',
                                    'neg-transpose', 'neg-transpose', 'optim']
leak_vals = [0., 0., 0., 0., 0.001, 0.01, 0., 0.001, 0.01]  # alpha
eta = 1e-2  # learning rate
norm_delW = False  # FOR FIG. 3b, SET THIS TO TRUE
if norm_delW:
    num_iters = 4000
else:
    num_iters = 2000

n_input = 30
n_hidden = 20
n_output = 10


##############################
# 2 create artificial dataset
##############################
T = np.random.uniform(-1.0, 1.0, (n_output, n_input))
x = np.random.normal(0.0, 1.0, (n_input, num_iters))
y_true = np.dot(T, x)


##########################
# 3 initialize weights
##########################
W0_tmp = np.random.uniform(-0.01, 0.01, (n_hidden, n_input))
W1_tmp = np.random.uniform(-0.01, 0.01, (n_output, n_hidden))
B1_tmp = np.random.uniform(-0.01, 0.01, (n_hidden, n_output))
B1_tmp_lrg = np.random.uniform(-0.5, 0.5, (n_hidden, n_output))

B1_opt = B1_tmp.copy()
if 'optim' in fback_type:  # optimized feedback
    B1_opt = None
    while B1_opt is None:
        W1_tmp = np.random.uniform(-0.01, 0.01, (n_output, n_hidden))
        B1_opt = SSA_opt(W1_tmp, B1_tmp, 0.0)

W0, W1 = {}, {}
B1 = {}
for i in range(len(model_name)):
    W0[model_name[i]] = W0_tmp.copy()
    W1[model_name[i]] = W1_tmp.copy()
    B1[model_name[i]] = {
        'random': B1_tmp_lrg.copy(),
        'optim': B1_opt.copy(),
        'neg-transpose': -W1_tmp.T.copy(),
        'n/a': np.array([])}[fback_type[i]]

alpha = {m: leak_vals[idx] for idx,m in enumerate(model_name)}


##########################
# 4 run training
##########################
error = {m: np.zeros((num_iters,)) for m in model_name}
stability = {m: np.zeros((num_iters, 1)) for m in model_name}
angles = {m: np.zeros((num_iters, 3)) for m in model_name}  # compare B with -W^T, and deltas

# loop through training examples
input_order = np.random.permutation(num_iters)
for i in range(num_iters):
    if np.mod(i, 100) == 0:
        print(i)

    # choose random input
    x_i = x[:, input_order[i]]

    # loop through all models
    for m_idx, m in enumerate(model_name):

        # A) forward pass
        a1 = np.dot(W0[m], x_i)
        h1 = a1
        dh1 = np.ones_like(h1)
        y = np.dot(W1[m], h1)
        dy = np.ones_like(y)
        e = y_true[:, input_order[i]] - y
        y_tilde = y_true[:, input_order[i]]

        # B) backward pass
        del2 = e
        del1 = backprop_error(model_type[m_idx], W1[m], B1[m], alpha[m], dh1, del2)

        # C) weight updates
        if np.linalg.norm(del1) > 10.:
            del1 /= (np.linalg.norm(del1)/10.)

        del_W0 = np.outer(del1, x_i.T)
        del_W1 = np.outer(del2, h1.T)

        if norm_delW:
            del_W0 /= np.linalg.norm(del_W0)
            del_W1 /= np.linalg.norm(del_W1)

        if i>0:  # skip first weight update so that each model begins with the same error
            W0[m] += eta * del_W0
            W1[m] += eta * del_W1

        # D) calculate error across entire training set
        y_total = np.dot(W1[m], np.dot(W0[m], x))
        error[m][i] = np.sum((y_true - y_total)**2)

        # E) extra stuff -- calculate stability and angles
        if model_type[m_idx] == 'DI':
            angles[m][i, 0] = get_mat_angle(B1[m], -W1[m])
            del1_ndi = backprop_error('NDI', W1[m], B1[m], alpha[m], dh1, del2)
            del1_pbp = backprop_error('PBP', W1[m], B1[m], alpha[m], dh1, del2)
            angles[m][i, 1] = get_vec_angle(del1, del1_ndi)
            angles[m][i, 2] = get_vec_angle(del1, del1_pbp)

            stability[m][i] = get_maxreal_eig(W1[m], B1[m], alpha[m])
            while (stability[m][i] > 0.):
                B1[m] = SSA_opt(W1[m], B1[m].copy(), alpha[m])
                stability[m][i] = get_maxreal_eig(W1[m], B1[m], alpha[m])


##########################
# 5 plot results
##########################

# normalize the error
n_error = {m:[] for m in model_name}
e_max = np.max([np.max(e) for e in error.values()])
print(e_max)
for m in model_name:
    n_error[m] = error[m] / e_max

cmap = ['#000000', '#bdbdbd', '#b15928', '#1f78b4', '#33a02c',
        '#6a3d9a', '#e31a1c', '#ff7f00', '#ffd92f', '#CCA700']
colors = [0, 1, 4, 3, 6, 9, 3, 6, 9]
a_vals = [1,1,1,1,1,1,0.4,0.4,0.4]

f = plt.figure(figsize=(12, 2.5), dpi=150, constrained_layout=True)
widths = [1.0, 0.75, 0.75]
gs = f.add_gridspec(nrows=2, ncols=3, width_ratios=widths)
ax1 = f.add_subplot(gs[:, 0])
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
for i in range(len(model_name)):
    plt.plot(range(0, num_iters), n_error[model_name[i]],c=cmap[colors[i]],alpha=a_vals[i])
plt.yscale('log')
plt.xticks([0, num_iters/2, num_iters], [0, int(num_iters/2), num_iters])
plt.ylabel('Error (NSE)')
plt.xlabel('No. examples')
ax1.legend(model_name, bbox_to_anchor=(1.1, 1.05))

ax2 = f.add_subplot(gs[0, 1])
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
for i in range(len(model_name)):
    if model_type[i]=='DI':
        plt.plot(range(0, num_iters), stability[model_name[i]], c=cmap[colors[i]])
plt.plot(range(0, num_iters), np.zeros((num_iters,)), 'k--')
plt.xlim([-10, 500])
plt.xticks([0, 200, 400], [])
plt.ylabel('$\lambda_{max}$')
ax3 = f.add_subplot(gs[1, 1])
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
for i in range(len(model_name)):
    if model_type[i]=='DI':
        plt.plot(range(0, num_iters), angles[model_name[i]][:, 0], c=cmap[colors[i]])
plt.ylabel('$B\sphericalangle W^T$')
plt.xlabel('No. examples')
plt.xticks([0, 200, 400], [0, 200, 400])
plt.xlim([-10, 500])
plt.yticks([0, 45], [0, 45])
ax4 = f.add_subplot(gs[0, 2])
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)
for i in range(len(model_name)):
    if model_type[i]=='DI':
        plt.plot(range(0, num_iters), angles[model_name[i]][:, 1], c=cmap[colors[i]])
plt.ylabel('$\delta_{DI}\sphericalangle\delta_{NDI}$')
plt.xticks([0, 200, 400], [])
plt.xlim([-10, 500])
plt.yticks([0, 45, 90], [0, 45, 90])
plt.ylim([-5, 90])
ax5 = f.add_subplot(gs[1, 2])
ax5.spines['right'].set_visible(False)
ax5.spines['top'].set_visible(False)
for i in range(len(model_name)):
    if model_type[i]=='DI':
        plt.plot(range(0, num_iters), angles[model_name[i]][:, 2], c=cmap[colors[i]])
plt.ylabel('$\delta_{DI}\sphericalangle\delta_{PBP}$')
plt.xlim([-10, 500])
plt.xticks([0, 200, 400], [0, 200, 400])
plt.yticks([0, 45, 90], [0, 45, 90])
plt.ylim([-5, 90])
plt.xlabel('No. examples')

plt.show()
