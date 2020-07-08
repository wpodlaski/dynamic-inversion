# linear regression example

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ortho_group
from sim_tools import backprop_error, get_maxreal_eig, SSA_opt, get_mat_angle, get_vec_angle
from mlxtend.data import loadlocal_mnist


##########################
# 1 hyper-parameters
##########################
model_name = ['BP', 'FA', 'PBP', 'NDI', 'NDI-2']
model_type = ['BP', 'FA', 'PBP', 'NDI', 'NDI']
fback_type = ['n/a', 'random', 'n/a', 'neg-transpose', 'neg-transpose']
leak_vals = [0., 0., 0., 0., 0.01]
w_init_type = 'normal'  # 'orthogonal'
eta = 1e-6
w_decay = 1e-10

batch_size = 100
num_epochs = 20
save_rate = 10 # number of batches
num_iters = int(60000. / (save_rate*batch_size))
total_itrs = num_epochs*num_iters

n_input = 784
n_hidden1 = 500
n_hidden2 = 250
n_hidden3 = 30
n_hidden4 = 250
n_hidden5 = 500
n_output = 784


##############################
# 2 load MNIST dataset
##############################
X_train, y_train = loadlocal_mnist(
                images_path='data/train-images-idx3-ubyte', 
                labels_path='data/train-labels-idx1-ubyte')
X_test, y_test = loadlocal_mnist(
                images_path='data/t10k-images-idx3-ubyte', 
                labels_path='data/t10k-labels-idx1-ubyte')
X_train = X_train.T
X_test = X_test.T
input_dim = X_train.shape[0]
n_train_iters = X_train.shape[1]
n_test_iters = X_test.shape[1]

# normalize to make data between 0 and 1
X_train = X_train / 255.
X_test = X_test / 255.


##########################
# 3 initialize weights
##########################
if w_init_type=='normal':
    W0_tmp = np.random.uniform(-0.01,0.01,(n_hidden1,n_input))
    W1_tmp = np.random.uniform(-0.01,0.01,(n_hidden2,n_hidden1))
    W2_tmp = np.random.uniform(-0.01,0.01,(n_hidden3,n_hidden2))
    W3_tmp = np.random.uniform(-0.01,0.01,(n_hidden4,n_hidden3))
    W4_tmp = np.random.uniform(-0.01,0.01,(n_hidden5,n_hidden4))
    W5_tmp = np.random.uniform(-0.01,0.01,(n_output,n_hidden5))
elif w_init_type=='orthogonal':
    W0_tmp = ortho_group.rvs(n_input)[:n_hidden1,:]
    W1_tmp = ortho_group.rvs(n_hidden1)[:n_hidden2,:]
    W2_tmp = ortho_group.rvs(n_hidden2)[:n_hidden3,:]
    W3_tmp = ortho_group.rvs(n_hidden4)[:,:n_hidden3]
    W4_tmp = ortho_group.rvs(n_hidden5)[:,:n_hidden4]
    W5_tmp = ortho_group.rvs(n_output)[:,:n_hidden5]
b0_tmp = np.zeros((n_hidden1,))
b1_tmp = np.zeros((n_hidden2,))
b2_tmp = np.zeros((n_hidden3,))
b3_tmp = np.zeros((n_hidden4,))
b4_tmp = np.zeros((n_hidden5,))
b5_tmp = np.zeros((n_output,))
B1_tmp = np.random.uniform(-0.01,0.01,(n_hidden1,n_hidden2))
B2_tmp = np.random.uniform(-0.01,0.01,(n_hidden2,n_hidden3))
B3_tmp = np.random.uniform(-0.01,0.01,(n_hidden3,n_hidden4))
B4_tmp = np.random.uniform(-0.01,0.01,(n_hidden4,n_hidden5))
B5_tmp = np.random.uniform(-0.01,0.01,(n_hidden5,n_output))
B1_opt = B1_tmp.copy()
if 'optim' in fback_type:
    B1_opt = None
    while B1_opt is None:
        B1_opt = SSA_opt(W1_tmp,B1_tmp.copy(),0.01)
    B2_opt = None
    while B2_opt is None:
        B2_opt = SSA_opt(W2_tmp,B2_tmp.copy(),0.01)
    B3_opt = None
    while B3_opt is None:
        B3_opt = SSA_opt(W3_tmp,B3_tmp.copy(),0.01)
    B4_opt = None
    while B4_opt is None:
        B4_opt = SSA_opt(W4_tmp,B4_tmp.copy(),0.01)
    B5_opt = None
    while B5_opt is None:
        B5_opt = SSA_opt(W5_tmp,B5_tmp.copy(),0.1)
else:
    B1_opt = B1_tmp.copy()
    B2_opt = B2_tmp.copy()
    B3_opt = B3_tmp.copy()
    B4_opt = B4_tmp.copy()
    B5_opt = B5_tmp.copy()

W0, W1, W2, W3, W4, W5 = {}, {}, {}, {}, {}, {}
b0, b1, b2, b3, b4, b5 = {}, {}, {}, {}, {}, {}
B1, B2, B3, B4, B5 = {}, {}, {}, {}, {}
for i in range(len(model_name)):
    W0[model_name[i]] = W0_tmp.copy()
    W1[model_name[i]] = W1_tmp.copy()
    W2[model_name[i]] = W2_tmp.copy()
    W3[model_name[i]] = W3_tmp.copy()
    W4[model_name[i]] = W4_tmp.copy()
    W5[model_name[i]] = W5_tmp.copy()
    b0[model_name[i]] = b0_tmp.copy()
    b1[model_name[i]] = b1_tmp.copy()
    b2[model_name[i]] = b2_tmp.copy()
    b3[model_name[i]] = b3_tmp.copy()
    b4[model_name[i]] = b4_tmp.copy()
    b5[model_name[i]] = b5_tmp.copy()
    B1[model_name[i]], B2[model_name[i]], B3[model_name[i]], B4[model_name[i]], B5[model_name[i]] = {
        'random': (B1_tmp.copy(),B2_tmp.copy(),B3_tmp.copy(),B4_tmp.copy(),B5_tmp.copy()),
        'optim': (B1_opt.copy(),B2_opt.copy(),B3_opt.copy(),B4_opt.copy(),B5_opt.copy()),
        'transpose': (W1_tmp.T.copy(),W2_tmp.T.copy(),W3_tmp.T.copy(),W4_tmp.T.copy(),W5_tmp.T.copy()),
        'pseudoinv': (np.linalg.pinv(W1_tmp),np.linalg.pinv(W2_tmp),np.linalg.pinv(W3_tmp),np.linalg.pinv(W4_tmp),np.linalg.pinv(W5_tmp)),
        'neg-transpose': (-W1_tmp.T.copy(),-W2_tmp.T.copy(),-W3_tmp.T.copy(),-W4_tmp.T.copy(),-W5_tmp.T.copy()),
        'neg-pseudoinv': (-np.linalg.pinv(W1_tmp),-np.linalg.pinv(W2_tmp),-np.linalg.pinv(W3_tmp),-np.linalg.pinv(W4_tmp),-np.linalg.pinv(W5_tmp)),
        'sign': (np.sign(W1_tmp.T),np.sign(W2_tmp.T),np.sign(W3_tmp.T),np.sign(W4_tmp.T),np.sign(W5_tmp.T)),
        'neg-sign': (-np.sign(W1_tmp.T),-np.sign(W2_tmp.T),-np.sign(W3_tmp.T),-np.sign(W4_tmp.T),-np.sign(W5_tmp.T)),
        'n/a': (np.array([]),np.array([]),np.array([]),np.array([]),np.array([]))}[fback_type[i]]

alpha = {m: leak_vals[idx] for idx,m in enumerate(model_name)}


##########################
# 4 run training
##########################
test_error = {m: [] for m in model_name}
train_error = {m: [] for m in model_name}
err_itrs = {m: [] for m in model_name}
stability = {m: np.zeros((num_iters, 1)) for m in model_name}
angles = {m: np.zeros((num_iters, 3)) for m in model_name}  # compare B with -W^T, and deltas

# loop through training examples
c=-1
for ep in range(num_epochs):

    print("Epoch Nr. %d"%ep)
    
    # jumble data
    permutation = np.random.permutation(60000)
    X_train = X_train[:,permutation]

    # loop through training examples
    for i in range(0,num_iters):
        c+=1

        if np.mod(i,10) == 0:
            print("%d of %d (epoch %d)" %(i,num_iters,ep))

        for j in range(save_rate):

            idx = i*save_rate*batch_size + j*batch_size
            X_batch = X_train[:,idx:(idx+batch_size)]

            # loop through all models
            for m_idx, m in enumerate(model_name):

                # A) forward pass
                b0_tile = np.tile(b0[m],(batch_size,1)).T
                b1_tile = np.tile(b1[m],(batch_size,1)).T
                b2_tile = np.tile(b2[m],(batch_size,1)).T
                b3_tile = np.tile(b3[m],(batch_size,1)).T
                b4_tile = np.tile(b4[m],(batch_size,1)).T
                b5_tile = np.tile(b5[m],(batch_size,1)).T

                a1 = np.dot(W0[m],X_batch) + b0_tile
                h1 = np.tanh(a1)
                dh1 = 1. - np.tanh(a1)**2

                a2 = np.dot(W1[m],h1) + b1_tile
                h2 = np.tanh(a2)
                dh2 = 1. - np.tanh(a2)**2

                a3 = np.dot(W2[m],h2) + b2_tile
                h3 = a3
                dh3 = np.ones_like(a3)
                
                a4 = np.dot(W3[m],h3) + b3_tile
                h4 = np.tanh(a4)
                dh4 = 1. - np.tanh(a4)**2
                
                a5 = np.dot(W4[m],h4) + b4_tile
                h5 = np.tanh(a5)
                dh5 = 1. - np.tanh(a5)**2

                a6 = np.dot(W5[m],h5) + b5_tile
                y = a6
                dy = np.ones_like(a6)

                e = X_batch - y

                # B) backward pass
                del6 = np.multiply(dy,e)
                del5 = backprop_error(model_type[m_idx], W5[m], B5[m], alpha[m], dh5, del6, 'tanh')
                del4 = backprop_error(model_type[m_idx], W4[m], B4[m], alpha[m], dh4, del5, 'tanh')
                del3 = backprop_error(model_type[m_idx], W3[m], B3[m], alpha[m], dh3, del4, 'linear')
                del2 = backprop_error(model_type[m_idx], W2[m], B2[m], alpha[m], dh2, del3, 'tanh')
                del1 = backprop_error(model_type[m_idx], W1[m], B1[m], alpha[m], dh1, del2, 'tanh')

                # C) weight updates
                del_W0 = np.dot(del1, X_batch.T)
                del_W1 = np.dot(del2, h1.T)
                del_W2 = np.dot(del3, h2.T)
                del_W3 = np.dot(del4, h3.T)
                del_W4 = np.dot(del5, h4.T)
                del_W5 = np.dot(del6, h5.T)
                W0[m] += eta * del_W0 - w_decay * W0[m]
                W1[m] += eta * del_W1 - w_decay * W1[m]
                W2[m] += eta * del_W2 - w_decay * W2[m]
                W3[m] += eta * del_W3 - w_decay * W3[m]
                W4[m] += eta * del_W4 - w_decay * W4[m]
                W5[m] += eta * del_W5 - w_decay * W5[m]
                b0[m] += eta * np.sum(del1,1) - w_decay * b0[m]
                b1[m] += eta * np.sum(del2,1) - w_decay * b1[m]
                b2[m] += eta * np.sum(del3,1) - w_decay * b2[m]
                b3[m] += eta * np.sum(del4,1) - w_decay * b3[m]
                b4[m] += eta * np.sum(del5,1) - w_decay * b4[m]
                b5[m] += eta * np.sum(del6,1) - w_decay * b5[m]

        # D) calculate error across training and test sets
        for m_idx,m in enumerate(model_name):
            y_total_test = ((np.dot(W5[m],np.tanh(np.dot(W4[m],np.tanh(np.dot(W3[m],(np.dot(W2[m],
                            np.tanh(np.dot(W1[m],np.tanh(np.dot(W0[m],X_test) 
                            + np.tile(b0[m],(n_test_iters,1)).T)) + np.tile(b1[m],(n_test_iters,1)).T)) 
                            + np.tile(b2[m],(n_test_iters,1)).T)) + np.tile(b3[m],(n_test_iters,1)).T))
                            + np.tile(b4[m],(n_test_iters,1)).T)) + np.tile(b5[m],(n_test_iters,1)).T))
            test_error[m].append(np.mean((X_test - y_total_test)**2))
            y_total_train = ((np.dot(W5[m],np.tanh(np.dot(W4[m],np.tanh(np.dot(W3[m],(np.dot(W2[m],
                            np.tanh(np.dot(W1[m],np.tanh(np.dot(W0[m],X_train) 
                            + np.tile(b0[m],(n_train_iters,1)).T)) + np.tile(b1[m],(n_train_iters,1)).T)) 
                            + np.tile(b2[m],(n_train_iters,1)).T)) + np.tile(b3[m],(n_train_iters,1)).T))
                            + np.tile(b4[m],(n_train_iters,1)).T)) + np.tile(b5[m],(n_train_iters,1)).T))
            train_error[m].append(np.mean((X_train - y_total_train)**2))
            print("%s, %d test error = %f"%(m,i,test_error[m][-1]))
            print("%s, %d train error = %f"%(m,i,train_error[m][-1]))
            err_itrs[m].append(c)                


##########################
# 5 plot results
##########################

cmap = ['#000000', '#bdbdbd', '#b15928', '#1f78b4', '#33a02c',
        '#6a3d9a', '#e31a1c', '#ff7f00', '#ffd92f', '#CCA700']
colors = [0, 1, 4, 3, 9]

f = plt.figure(figsize=(3, 2.5), dpi=150, constrained_layout=True)
gs = f.add_gridspec(nrows=1, ncols=1)
ax1 = f.add_subplot(gs[0, 0])
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
for i in range(len(model_name)):
    plt.plot(err_itrs[model_name[i]], test_error[model_name[i]],c=cmap[colors[i]])
plt.xticks([0,0.2,0.4,0.6,0.8], [0, 20, 40, 60, 80])
plt.ylabel('Test Error (%)')
plt.xlabel('No. examples')
ax1.legend(model_name, bbox_to_anchor=(1.1, 1.05))

plt.show()
