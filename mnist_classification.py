# linear regression example

import numpy as np
import matplotlib.pyplot as plt
from sim_tools import backprop_error, get_maxreal_eig, SSA_opt, get_mat_angle, get_vec_angle
from mlxtend.data import loadlocal_mnist


##########################
# 1 hyper-parameters
##########################
model_name = ['BP', 'FA', 'PBP', 'NDI', 'NDI-2']
model_type = ['BP', 'FA', 'PBP', 'NDI', 'NDI']
fback_type = ['n/a', 'random', 'n/a', 'neg-transpose', 'neg-transpose']
leak_vals = [0., 0., 0., 0., 0.01]
eta = 1e-3
w_decay = 1e-6
num_epochs = 15
num_iters = 60000
save_itrs = 1000
total_itrs = num_epochs*num_iters

n_input = 784
n_hidden = 1000
n_output = 10


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

# normalize to obtain zero mean and unity standard deviation
X_mean = np.mean(X_train)
X_std = np.std(X_train)
X_train = (X_train - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

# create one-hot labels
y_train_1H = np.zeros((10,len(y_train)))
y_train_1H[y_train,np.arange(len(y_train),dtype=np.int)] = 1.
y_test_1H = np.zeros((10,len(y_test)))
y_test_1H[y_test,np.arange(len(y_test),dtype=np.int)] = 1.


##########################
# 3 initialize weights
##########################
W0_tmp = np.random.uniform(-0.01, 0.01, (n_hidden, n_input))
W1_tmp = np.random.uniform(-0.01, 0.01, (n_output, n_hidden))
b0_tmp = np.zeros((n_hidden,))
b1_tmp = np.zeros((n_output,))
B1_tmp = np.random.uniform(-0.01, 0.01, (n_hidden, n_output))
B1_tmp_lrg = np.random.uniform(-0.5, 0.5, (n_hidden, n_output))

B1_opt = B1_tmp.copy()
if 'optim' in fback_type:  # optimized feedback
    B1_opt = None
    while B1_opt is None:
        W1_tmp = np.random.uniform(-0.01, 0.01, (n_output, n_hidden))
        B1_opt = SSA_opt(W1_tmp, B1_tmp, 0.0)

W0, W1 = {}, {}
b0, b1 = {}, {}
B1 = {}
for i in range(len(model_name)):
    W0[model_name[i]] = W0_tmp.copy()
    W1[model_name[i]] = W1_tmp.copy()
    b0[model_name[i]] = b0_tmp.copy()
    b1[model_name[i]] = b1_tmp.copy()
    B1[model_name[i]] = {
        'random': B1_tmp_lrg.copy(),
        'optim': B1_opt.copy(),
        'neg-transpose': -W1_tmp.T.copy(),
        'n/a': np.array([])}[fback_type[i]]

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
    
    # loop through training examples
    input_order = np.random.permutation(num_iters)
    for i in range(0,num_iters):
        c+=1
        if np.mod(i,100) == 0:
            print(i)

        # choose random input
        x_i = X_train[:,input_order[i]]

        # loop through all models
        for m_idx, m in enumerate(model_name):

            # A) forward pass
            a1 = np.dot(W0[m],x_i) + b0[m]
            h1 = np.tanh(a1)
            dh1 = 1. - np.tanh(a1)**2
            a2 = np.dot(W1[m],h1) + b1[m]
            y = np.exp(a2) / np.sum(np.exp(a2))
            e = y_train_1H[:,input_order[i]] - y

            if np.any(np.isnan(y)):
                y[np.isnan(y)] = 1.

            # B) backward pass
            del2 = e
            del1 = backprop_error(model_type[m_idx], W1[m], B1[m], alpha[m], dh1, del2, 'tanh')

            # C) weight updates
            del_W0 = np.outer(del1, x_i.T)
            del_W1 = np.outer(del2, h1.T)
            W0[m] += eta * del_W0 - w_decay * W0[m]
            W1[m] += eta * del_W1 - w_decay * W1[m]
            b0[m] += eta * del1 - w_decay * b0[m]
            b1[m] += eta * del2 - w_decay * b1[m]

            # D) calculate error across training and test sets
            if np.mod(i,save_itrs)==0:
                a2_total_test = (np.dot(W1[m],np.tanh(np.dot(W0[m],X_test)+np.tile(b0[m],(n_test_iters,1)).T))
                            + np.tile(b1[m],(n_test_iters,1)).T)
                test_error[m].append(np.count_nonzero(y_test - np.argmax(a2_total_test,axis=0))/len(y_test))
                a2_total_train = (np.dot(W1[m],np.tanh(np.dot(W0[m],X_train)+np.tile(b0[m],(n_train_iters,1)).T))
                            + np.tile(b1[m],(n_train_iters,1)).T)
                train_error[m].append(np.count_nonzero(y_train - np.argmax(a2_total_train,axis=0))/len(y_train))
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
