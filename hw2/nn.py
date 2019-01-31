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
    return np.sum(Wx, b1, axis=0)

# Elementwise ReLU nonlinearity to produce the hidden layer
def hidden_layer(Z):
    return np.maximum(Z,0,Z)

# [000,,,1,,,,00000]
def e(elem, K):
    ret = numpy.zeros(K)
    ret[elem] = 1
    return ret

def rho(vec, Y):
    return -1*(np.log(vec[Y]))

def partial_U()





# load MNIST data
MNIST_data = h5py.File('../MNISTdata.hdf5', 'r')

x_train = np.float32(MNIST_data['x_train'][:])
y_train = np.int32(np.array(MNIST_data['y_train'][:, 0]))
x_test = np.float32(MNIST_data['x_test'][:])
y_test = np.int32(np.array(MNIST_data['y_test'][:, 0]))

MNIST_data.close()

#######################################################################

# IMPLEMENT Neural Network




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