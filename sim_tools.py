
import math
import numpy as np
from scipy.linalg import solve_continuous_lyapunov
from numba import njit

rcond = -1
tsteps = 10000
dt = 0.1

# ERROR BACKPROPAGATION FUNCTIONS

def run_linear_di_net(dx, dy, W, B, y_hat, alpha, dt=dt, tsteps=tsteps):
    rx = np.zeros((dx,))
    ry = np.zeros((dy,))
    for t in range(1,tsteps):
        ry = ry + dt*(-alpha*ry + np.dot(W,rx) - y_hat)
        rx = rx + dt*(-rx + np.dot(B,ry))
    return rx

def run_tanh_di_net(dx, dy, W, B, y_hat, alpha, dt=dt, tsteps=tsteps):
    rx = np.zeros((dx,))
    ry = np.zeros((dy,))
    for t in range(1,tsteps):
        ry = ry + dt*(-alpha*ry + np.dot(W,np.tanh(rx)) - y_hat)
        rx = rx + dt*(-rx + np.dot(B,ry))
    return rx

# for single-loop dynamic inversion (SDI)
def run_2L_tanh_di_net(dx, dy, dz, W, W2, B, z_hat, alpha, dt=dt, tsteps=tsteps):
    rx = np.zeros((dx,))
    ry = np.zeros((dy,))
    rz = np.zeros((dz,))
    for t in range(1,tsteps):
        rx = rx + dt*(-rx + np.dot(B,rz))
        ry = ry + dt*(-ry + np.dot(W,np.tanh(rx)))
        rz = rz + dt*(-alpha*rz + np.dot(W2,np.tanh(ry)) - z_hat)
    return (rx,ry)

# use numba to speed up dynamic inversion algorithms
jitted_run_linear_di_net = njit()(run_linear_di_net)
jitted_run_tanh_di_net = njit()(run_tanh_di_net)
jitted_run_2L_tanh_di_net = njit()(run_2L_tanh_di_net)

def backprop_error(m,W,B,alpha,dh,delta0,tfunc_str='linear'):
	if m=='BP':
		return np.multiply( np.dot(W.T, delta0), dh)
	elif m=='FA':
		return np.multiply( np.dot(B, delta0), dh)
	elif m=='PBP':
		[del_tmp,_,_,_] = np.linalg.lstsq(W,delta0,rcond=rcond)
		return np.multiply(del_tmp,dh)
	elif m=='NDI' and W.shape[0]<=W.shape[1]:
		[del_tmp,_,_,_] = np.linalg.lstsq(np.dot(W,B)-alpha*np.eye(W.shape[0]),delta0,rcond=rcond)
		return np.multiply(np.dot(B,del_tmp), dh)
	elif m=='NDI' and W.shape[0]>W.shape[1]:
		[del_tmp,_,_,_] = np.linalg.lstsq(np.dot(B,W)-alpha*np.eye(W.shape[1]),np.dot(B,delta0),rcond=rcond)
		return np.multiply(del_tmp, dh)
	elif m=='DI':
		if tfunc_str=='linear':
			return jitted_run_linear_di_net(W.shape[1], W.shape[0], W, B, delta0, alpha, dt, tsteps)
		elif tfunc_str=='tanh':
			return jitted_run_tanh_di_net(W.shape[1], W.shape[0], W, B, delta0, alpha, dt, tsteps)
	else:
		raise Exception('Model type not recognized.')


# EXTRA HELPER FUNCTIONS

def get_maxreal_eig(W,B,alpha):
    dims = W.shape
    M = np.block([[-np.eye(dims[1]),B],[W,-alpha*np.eye(dims[0])]])
    return np.max(np.real(np.linalg.eigvals(M)))


# Smoothed spectral abscissa (SSA) optimization (see Vanbiervliet et al. 2009, Hennequin et al. 2014)
def SSA_opt(W,B,alpha,lr=5e-2,maxiters=200,miniters=10):
    
    # calculate max eigenvalue
    d1,d2 = W.shape
    M = np.block([[-np.eye(d2),B],[W,-alpha*np.eye(d1)]])
    s = np.max(np.real(np.linalg.eigvals(M)))
    print('Optimizing feedback weights. lambda init = ',s)
    
    itr = 0
    while (miniters>itr or (s>-0.01 and itr<maxiters)):
    
        # calculate gradient using Q and P
        Q = solve_continuous_lyapunov((M-s*np.eye(d1+d2)).T, 2*np.eye(d1+d2))
        P = solve_continuous_lyapunov((M-s*np.eye(d1+d2)), 2*np.eye(d1+d2))
        grad = np.dot(Q,P)
        grad /= np.trace(grad)

        # update only the B matrix
        gradB = grad[:d2,d2:]
        B -= lr * gradB
        
        itr += 1
        
        # calculate max eigenvalue
        M = np.block([[-np.eye(d2),B],[W,-alpha*np.eye(d1)]])
        s = np.max(np.real(np.linalg.eigvals(M)))
        
    print('lambda final = ',s)
        
    if s<0.:
        return B
    else:
        return None
    

def get_mat_angle(A,B):
    x = np.trace(np.dot(A,B))/np.linalg.norm(A)/np.linalg.norm(B)
    if x < -1.:
        x = -1.
    elif x > 1.:
        x = 1.
    return math.degrees(np.arccos(x))

def get_vec_angle(a,b):
    x = np.dot(a,b)/np.linalg.norm(a)/np.linalg.norm(b)
    if x < -1.:
        x = -1.
    elif x > 1.:
        x = 1.
    return math.degrees(np.arccos(x))

