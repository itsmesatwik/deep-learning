import numpy as np

import h5py
# import time
import copy
from random import randint

############################
#   SINGLE HIDDEN LAYER    #
#     MULTIPLE KERNELS     #
############################



# Elementwise ReLU nonlinearity to produce the hidden layer
def hidden_layer(Z):
    return np.maximum(Z,0,Z)

# Convolution of matrix X with the filter K
def convolution(X, K, stride):
    Z = np.matlib.zeros((X.shape[0]-K.shape[0]+1, X.shape[1]-K.shape[1]+1))
    for m in range(0, conv.shape[0]-1, stride):
        for n in range(0, conv.shape[1]-1, stride):
            X_slice = X[m:m+K.shape[0]-1, n:n+K.shape[1]-1]
            Z[m,n] = np.vdot(X_slice, K)

    return Z



def hidden_linear(H, W, bk):
    U = np.zeros((W.shape[0]))
    for k in range(W.shape[0]):
        U[k] = np.vdot(W[k], H) + bk[k]

    return U

# [000,,,1,,,,00000]
def e(elem, K):
    ret = np.zeros(K)
    ret[elem] = 1
    return ret

# Softmax Function
def softmax(vec_):
    vec = vec_.A
    #assuming vec is 1-d
    exp_vec = np.exp(vec)
    vsum = np.float32(1/np.float32(sum(exp_vec)))
    exp_vec *= vsum
    return exp_vec

def sigma_prime(Z):
    for i in range(len(Z)):
        if Z[i] >= 0:
            Z[i] = 1
        else:
            Z[i] = 0
    return Z


def partial_K(sigma_prime, delta, X):
    return convolution(X, (sigma_prime*delta), 1)


def partial_U(soft, e_Y):
    return -1*(e_Y-soft)


def partial_W(partial_b, H):
    ret = np.zeros((partial_b.shape[0], H.shape[0], H.shape[1]))
    for i in range(ret.shape[0]):
        ret[i] = partial_b[i]*H
    return ret

def delta(W, partial_u):
    delt = np.zeros(W.shape[1], W.shape[2])
    for i in range(partial_u.shape[0]):
        delt += partial_u[i]*W[i]
    return delt


def param_update(param, ALPHA, grad):
    return param - (ALPHA*grad)

    
    




MNIST_data = h5py.File('../MNISTdata.hdf5', 'r')

x_train = np.float32(MNIST_data['x_train'][:])
y_train = np.int32(np.array(MNIST_data['y_train'][:, 0]))
x_test = np.float32(MNIST_data['x_test'][:])
y_test = np.int32(np.array(MNIST_data['y_test'][:, 0]))

MNIST_data.close()

#######################################################################

# IMPLEMENT Neural Network

num_inputs = 28*28
input_dim = 28
num_outputs = 10
filter_dim = 5
K = np,random.randn(filter_dim, filter_dim) / np.sqrt(filter_dim)
b = np.random.randn(num_outputs) / np.sqrt(num_outputs)
W = np.random.randn(num_outputs, input_dim - filter_dim + 1, input_dim - filter_dim + 1) / np.sqrt(input_dim - filter_dim + 1)
EPOCH = 2
ALPHA = 0.003

for ep in range(EPOCH):
    print(ep)
    shuffle = np.arange(x_train.shape[0])
    np.random.shuffle(shuffle)
    shuffle_x = x_train[shuffle]
    shuffle_y = y_train[shuffle]
    for i in range(len(shuffle_x)):

        X = (shuffle_x[i])

        # FORWARD STEP
        Z = convolution(X, K, 1)
        sigma_z = sigma(Z)
        H = hidden_layer(Z)
        U = hidden_linear(H,W,b)
        soft_x = softmax(U)
        e_y = e(shuffle_y[i], num_outputs)

        # CALCULATE PARTIAL DERIVATIVES
        par_u = partial_U(soft_x, e_y)
        par_w = partial_W(par_u, H)
        delta = delta()
        par_k = partial_K(sigma_prime(Z), delta, X)

        # UPDATE PARAMETERS

        C = param_update(C, ALPHA, par_c)
        W = param_update(W, ALPHA, par_w)
        b1 = param_update(b1, ALPHA, par_b1)
        b2 = param_update(b2, ALPHA, par_b2)





#######################################################################

# Test Data

total_correct = 0

for n in range(len(x_test)):
    y = y_test[n]
    x = x_test[n][:]
    Z = linear_step(W, x, b1)
    H = hidden_layer(Z)
    U = hidden_linear(H, W, b)
    soft_x = softmax(U)
    prediction = np.argmax(soft_x)
    if (prediction == y):
        total_correct += 1


print (total_correct/np.float(len(x_test)))