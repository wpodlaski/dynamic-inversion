# nonlinear regression example

import numpy as np
import matplotlib.pyplot as plt
from sim_tools import (backprop_error, get_maxreal_eig, SSA_opt, get_mat_angle, get_vec_angle,
                    jitted_run_2L_tanh_di_net, jitted_run_2L_linear_di_net, jitted_run_2L_linearized_di_net)


##########################
# 1 hyper-parameters
##########################
model_name = ['BP', 'FA', 'FA-2', 'PBP', 'NDI', 'NDI-2', 'NDI-3',
              'DI', 'DI-2', 'DI-3', 'SDI', 'SDI-2', 'SDI-3']
model_type = ['BP', 'FA', 'FA', 'PBP', 'NDI', 'NDI', 'NDI',
                'DI', 'DI', 'DI', 'SDI', 'SDI', 'SDI']
fback_type = ['n/a', 'random', 'neg-transpose', 'n/a', 'neg-transpose', 'neg-transpose', 'optim', 'neg-transpose',
              'neg-transpose', 'optim', 'neg-transpose', 'neg-transpose', 'optim']
leak_vals = [0., 0., 0., 0., 0.01, 0.001, 0.01, 0.01, 0.001, 0.01, 0.01, 0.001, 0.01]
eta = 1e-2
num_iters = 10000  # set this to 20000 for exact replication of Fig. 3

n_input = 30
n_hidden1 = 20
n_hidden2 = 10
n_output = 10


##############################
# 2 create artificial dataset
##############################
W0_T = np.random.uniform(-0.01, 0.01, (n_hidden1, n_input))
W1_T = np.random.uniform(-0.01, 0.01, (n_hidden2, n_hidden1))
W2_T = np.random.uniform(-0.01, 0.01, (n_output, n_hidden2))
b0_T = np.random.uniform(-0.01, 0.01, (n_hidden1,))
b1_T = np.random.uniform(-0.01, 0.01, (n_hidden2,))
b2_T = np.random.uniform(-0.01, 0.01, (n_output,))
x = np.random.normal(0.0, 1.0, (n_input, num_iters))
y_true = np.dot(W2_T, np.tanh(np.dot(W1_T, np.tanh(np.dot(W0_T, x) + np.tile(b0_T, (num_iters, 1)).T))
                              + np.tile(b1_T, (num_iters, 1)).T)) + np.tile(b2_T, (num_iters, 1)).T


##########################
# 3 initialize weights
##########################
W0_tmp = np.random.uniform(-0.01, 0.01, (n_hidden1, n_input))
W1_tmp = np.random.uniform(-0.01, 0.01, (n_hidden2, n_hidden1))
W2_tmp = np.random.uniform(-0.01, 0.01, (n_output, n_hidden2))
b0_tmp = np.random.uniform(-0.01, 0.01, (n_hidden1,))
b1_tmp = np.random.uniform(-0.01, 0.01, (n_hidden2,))
b2_tmp = np.random.uniform(-0.01, 0.01, (n_output,))
B1_tmp = np.random.uniform(-0.01, 0.01, (n_hidden1, n_hidden2))
B2_tmp = np.random.uniform(-0.01, 0.01, (n_hidden2, n_output))
B_tmp = np.random.uniform(-0.01, 0.01, (n_hidden1, n_output))
B1_tmp_lrg = np.random.uniform(-0.5, 0.5, (n_hidden1, n_hidden2))
B2_tmp_lrg = np.random.uniform(-0.5, 0.5, (n_hidden2, n_output))
B_tmp_lrg = np.random.uniform(-0.5, 0.5, (n_hidden1, n_output))
B1_opt = B1_tmp.copy()
B2_opt = B2_tmp.copy()
B_opt = B_tmp.copy()
if 'optim' in fback_type:
    B1_opt = None
    B2_opt = None
    B_opt = None
    while B2_opt is None:
        B2_opt = SSA_opt(W2_tmp, B2_tmp, 0.0,lr=5e-1,maxiters=2000,miniters=100)
    while B1_opt is None:
        B1_opt = SSA_opt(W1_tmp, B1_tmp, 0.0,lr=5e-1,maxiters=2000,miniters=100)
    #while B_opt is None:
    #    B_opt = SSA_opt(np.dot(W2_tmp, W1_tmp), B_tmp, 0.0,lr=5e-1,maxiters=2000,miniters=100)
W0, W1, W2 = {}, {}, {}
b0, b1, b2 = {}, {}, {}
B1, B2, B = {}, {}, {}
for i in range(len(model_name)):
    W0[model_name[i]] = W0_tmp.copy()
    W1[model_name[i]] = W1_tmp.copy()
    W2[model_name[i]] = W2_tmp.copy()
    b0[model_name[i]] = b0_tmp.copy()
    b1[model_name[i]] = b1_tmp.copy()
    b2[model_name[i]] = b2_tmp.copy()
    B1[model_name[i]], B2[model_name[i]], B[model_name[i]] = {
        'random': (B1_tmp_lrg.copy(), B2_tmp_lrg.copy(), B_tmp_lrg.copy()),
        'optim': (B1_opt.copy(), B2_opt.copy(), -np.dot(B1_opt,B2_opt)),
        'neg-transpose': (-W1_tmp.T.copy(), -W2_tmp.T.copy(), -np.dot(W1_tmp.T, W2_tmp.T)),
        'n/a': (np.array([]), np.array([]), np.array([]))}[fback_type[i]]

alpha = {m: leak_vals[idx] for idx, m in enumerate(model_name)}


##########################
# 4 run training
##########################
error = {m: np.zeros((num_iters,)) for m in model_name}
stability = {m: np.zeros((num_iters, 2)) for m in model_name}
angles = {m: np.zeros((num_iters, 4)) for m in model_name}  # compare B with -W^T, and deltas

# loop through training examples
input_order = np.random.permutation(num_iters)
for i in range(num_iters):
    if np.mod(i, 100) == 0:
        print('Iter: %d / %d'%(i,num_iters))

    # choose random input
    x_i = x[:, input_order[i]]

    # loop through all models
    for m_idx, m in enumerate(model_name):

        # A) forward pass
        a1 = np.dot(W0[m], x_i) + b0[m]
        h1 = np.tanh(a1)
        dh1 = 1. - np.tanh(a1)**2
        a2 = np.dot(W1[m], h1) + b1[m]
        h2 = np.tanh(a2)
        dh2 = 1. - np.tanh(a2)**2
        y = np.dot(W2[m], h2) + b2[m]
        e = y_true[:, input_order[i]] - y

        # B) backward pass
        del3 = e
        if model_type[m_idx] == 'SDI':
            (del1, del2) = jitted_run_2L_tanh_di_net(n_hidden1, n_hidden2, n_output,
                                                      W1[m], W2[m], B[m], del3, alpha[m])#, dh1, dh2)
        else:
            del2 = backprop_error(model_type[m_idx], W2[m], B2[m], alpha[m], dh2, del3, 'linear')
            del2 = np.multiply(del2,dh2)
            del1 = backprop_error(model_type[m_idx], W1[m], B1[m], alpha[m], dh1, del2, 'linear')
            del1 = np.multiply(del1,dh1)

        # C) weight updates
        if i>0:  # skip first weight update so that each model begins with the same error
            if np.linalg.norm(del1) > 1.:
                del1 /= (np.linalg.norm(del1)/1.)
            if np.linalg.norm(del2) > 1.:
                del2 /= (np.linalg.norm(del2)/1.)

            del_W0 = np.outer(del1, x_i.T)
            del_W1 = np.outer(del2, h1.T)
            del_W2 = np.outer(del3, h2.T)
            W0[m] += eta * del_W0
            W1[m] += eta * del_W1
            W2[m] += eta * del_W2
            b0[m] += eta * del1
            b1[m] += eta * del2
            b2[m] += eta * del3

        # D) calculate error across entire training set
        y_total = (np.dot(W2[m], np.tanh(np.dot(W1[m], np.tanh(np.dot(W0[m], x) + np.tile(b0[m], (num_iters, 1)).T)) 
                        + np.tile(b1[m], (num_iters, 1)).T)) + np.tile(b2[m], (num_iters, 1)).T)
        error[m][i] = np.sum((y_true - y_total)**2)

        # E) extra stuff -- calculate stability and angles
        if model_type[m_idx] == 'DI':
            angles[m][i, 0] = get_mat_angle(B1[m], -W1[m])
            angles[m][i, 1] = get_mat_angle(B2[m], -W2[m])
            del2_ndi = backprop_error('NDI', W2[m], B2[m], alpha[m], dh2, del3)
            del1_ndi = backprop_error('NDI', W1[m], B1[m], alpha[m], dh1, del2_ndi)
            angles[m][i, 2] = get_vec_angle(del1, del1_ndi)
            angles[m][i, 3] = get_vec_angle(del2, del2_ndi)

            stability[m][i, 0] = get_maxreal_eig(W1[m], B1[m], alpha[m])
            while (stability[m][i, 0] > 0.):
                B1[m] = SSA_opt(W1[m], B1[m].copy(), alpha[m],lr=5e-1,maxiters=2000,miniters=100)
                stability[m][i, 0] = get_maxreal_eig(W1[m], B1[m], alpha[m])
            stability[m][i, 1] = get_maxreal_eig(W2[m], B2[m], alpha[m])
            while (stability[m][i, 1] > 0.):
                B1[m] = SSA_opt(W2[m], B2[m].copy(), alpha[m],lr=5e-1,maxiters=2000,miniters=100)
                stability[m][i, 1] = get_maxreal_eig(W2[m], B2[m], alpha[m])

        if model_type[m_idx] == 'SDI':
            angles[m][i, 0] = get_mat_angle(B1[m], -W1[m])
            angles[m][i, 1] = get_mat_angle(B2[m], -W2[m])
            del2_ndi = backprop_error('NDI', W2[m], B2[m], alpha[m], dh2, del3)
            del1_ndi = backprop_error('NDI', W1[m], B1[m], alpha[m], dh1, del2_ndi)
            angles[m][i, 2] = get_vec_angle(del1, del1_ndi)
            angles[m][i, 3] = get_vec_angle(del2, del2_ndi)

            stability[m][i, 0] = get_maxreal_eig(W1[m], B1[m], alpha[m])
            while (stability[m][i, 0] > 0.):
                B1[m] = SSA_opt(W1[m], B1[m].copy(), alpha[m],lr=5e-1,maxiters=2000,miniters=100)
                stability[m][i, 0] = get_maxreal_eig(W1[m], B1[m], alpha[m])
            stability[m][i, 1] = get_maxreal_eig(W2[m], B2[m], alpha[m])
            while (stability[m][i, 1] > 0.):
                B1[m] = SSA_opt(W2[m], B2[m].copy(), alpha[m],lr=5e-1,maxiters=2000,miniters=100)
                stability[m][i, 1] = get_maxreal_eig(W2[m], B2[m], alpha[m])


##########################
# 5 plot results
##########################

# normalize the error
n_error = {m: [] for m in model_name}
e_max = np.max([np.max(e) for e in error.values()])
for m in model_name:
    n_error[m] = error[m] / e_max

cmap = ['#000000', '#bdbdbd', '#b15928', '#1f78b4', '#33a02c',
        '#6a3d9a', '#e31a1c', '#ff7f00', '#ffd92f', '#CCA700']
colors = [0, 1, 1, 4, 3, 6, 9, 3, 6, 9, 3, 6, 9]
a_vals = [1, 1, 0.4, 1, 0.4, 0.4, 0.4, 1, 1, 1, 1, 1, 1]
s_vals = ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', ':', ':', ':']

f = plt.figure(figsize=(12, 3), dpi=150, constrained_layout=True)
widths = [1.0, 0.75, 0.75]
gs = f.add_gridspec(nrows=2, ncols=3, width_ratios=widths)
ax1 = f.add_subplot(gs[:, 0])
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
for i in range(len(model_name)):
    plt.plot(range(0, num_iters), n_error[model_name[i]], s_vals[i], c=cmap[colors[i]], alpha=a_vals[i])
plt.yscale('log')
plt.xticks([0, num_iters/2, num_iters], [0, int(num_iters/2), num_iters])
plt.ylabel('Error (NSE)')
plt.xlabel('No. examples')
ax1.legend(model_name, bbox_to_anchor=(1.1, 1.05))

ax2 = f.add_subplot(gs[0, 1])
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
for i in range(len(model_name)):
    if model_type[i] == 'DI':
        plt.plot(range(0, num_iters), np.abs(stability[model_name[i]][:,0]), c=cmap[colors[i]])
        plt.plot(range(0, num_iters), np.abs(stability[model_name[i]][:,1]), '--', c=cmap[colors[i]])
plt.yscale('log')
plt.gca().invert_yaxis()
plt.xlim([-10, 500])
plt.xticks([0, 200, 400], [])
plt.yticks([10**-3,10**-2],['$-10^{-3}$','$-10^{-2}$'])
#plt.ylim([0.0008,0.012])
plt.ylabel('$\lambda_{max}$')
ax3 = f.add_subplot(gs[1, 1])
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
for i in range(len(model_name)):
    if model_type[i] == 'DI':
        plt.plot(range(0, num_iters),angles[model_name[i]][:, 0], c=cmap[colors[i]])
        plt.plot(range(0, num_iters),angles[model_name[i]][:, 1], '--', c=cmap[colors[i]])
plt.ylabel('$B\sphericalangle W^T$')
plt.xlabel('No. examples')
plt.xticks([0, 200, 400], [0, 200, 400])
plt.xlim([-10, 500])
plt.yticks([0, 45], [0, 45])
ax4 = f.add_subplot(gs[0, 2])
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)
for i in range(len(model_name)):
    if model_type[i] == 'DI':
        plt.plot(range(0, num_iters),angles[model_name[i]][:, 2], c=cmap[colors[i]])
        plt.plot(range(0, num_iters),angles[model_name[i]][:, 3], '--',c=cmap[colors[i]])
plt.ylabel('$\delta_{DI}\sphericalangle\delta_{NDI}$')
plt.xticks([0, 200, 400], [])
plt.xlim([-10, 500])
plt.yticks([0, 15], [0, 15])
#plt.ylim([-5, 50])
ax5 = f.add_subplot(gs[1, 2])
ax5.spines['right'].set_visible(False)
ax5.spines['top'].set_visible(False)
for i in range(len(model_name)):
    if model_type[i] == 'SDI':
        plt.plot(range(0, num_iters),angles[model_name[i]][:, 2], c=cmap[colors[i]])
        plt.plot(range(0, num_iters),angles[model_name[i]][:, 3], '--',c=cmap[colors[i]])
plt.ylabel('$\delta_{SDI}\sphericalangle\delta_{NDI}$')
plt.xlim([-10, 500])
plt.xticks([0, 200, 400], [0, 200, 400])
plt.yticks([0, 45], [0, 45])
plt.ylim([-2, 55])
plt.xlabel('No. examples')

plt.tight_layout()

f.savefig('nonlinear_regression.pdf')

plt.show()
