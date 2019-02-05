import numpy as np

import h5py
# import time
import copy
from random import randint


# Softmax Function

def softmax(vec):
    #assuming vec is 1-d
    exp_vec = np.exp(vec)
    vsum = np.float32(1/np.float32(sum(exp_vec)))
    exp_vec *= vsum

    return exp_vec 

# Linear Step i.e Linear transformation of X
# return Wx + b1

def linear_step(W, x, b1):
    Wx = np.matmul(W,x)
    return Wx + b1

# Elementwise ReLU nonlinearity to produce the hidden layer
def hidden_layer(Z):
    return np.maximum(Z,0,Z)

# [000,,,1,,,,00000]
def e(elem, K):
    ret = np.zeros(K)
    ret[elem] = 1
    return ret

def cross_entropy_error(vec, Y):
    return -1*(np.log(vec[Y]))

def partial_U(soft, Y):
    return -1*(Y-soft)

def partial_b2(partial_u):
    return partial_u

def partial_C(partial_u, H):
    return np.matmul(partial_u, np.transpose(H))

def partial_b1(delta, Z):
    return

def partial_W(p_b1, X):
    return np.matmul(p_b1 ,np.transpose(X))

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
num_outputs = 10
dH = 50
C = np.random.randn(num_outputs,dH) / np.sqrt(dH)
W = np.random.randn(dH, num_inputs) / np.sqrt(num_inputs)
b1 = np.random.randn(dH) / np.sqrt(dH)
b2 = np.random.randn(num_outputs) / np.sqrt(num_outputs)
EPOCH = 10
ALPHA = 1

for ep in range(EPOCH):
    shuffle = np.arange(x_train.shape[0])
    np.random.shuffle(shuffle)
    shuffle_x = x_train[shuffle]
    shuffle_y = y_train[shuffle]
    for i in range(len(shuffle_x)):
        Z = linear_step(W, shuffle_x[i], b1)
        H = hidden_layer(Z)
        U = linear_step(C, H, b2)
        soft_x = softmax(shuffle_x[i])
        e_y = e(shuffle_y[i], num_outputs)
        par_U = soft_x - e_y
        




#######################################################################

# Test Data

total_correct = 0

for n in range(len(x_test)):
    y = y_test[n]
    x = x_test[n][:]
    #p = matrix_mult(theta_0, x)
    prediction = np.argmax(p)
    if (prediction == y):
        total_correct += 1


print (total_correct/np.float(len(x_test)))