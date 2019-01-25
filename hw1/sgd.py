import numpy as np

import h5py
import time 
import copy 
from random import randint


# SOFFTMAX FUNCTION
def f_softmax(vec):
    sum = 0
    ret = []
    for i in vec:
        sum += np.exp(i)
    for i in vec:
        ret.append(np.exp(i)/sum)
    return ret

# E(Y) FUNCTION
def e_func(elem):
    ret = [0]*10
    ret[elem] = 1
    return ret


# MULTIPLYING THETA AND X_VEC
def matrix_mult(theta, x):
    ret = [0]*10
    for i in range(theta):
        for j in range(theta[0]):
            ret[i] += theta[i][j]*x[j]

    return ret


def loss_grad(theta, x, y):
    e_vec = e_func(y)
    theta_x = matrix_mult(theta,x)
    softmax_vec = f_softmax(theta_x)
    intermed_vec = [0]*10
    for i in range(10):
        intermed_vec[i] = e_vec[i] - softmax_vec[i]
    final_vec = [[0]*10]*784
    for i in range(784):
        for j in range(10):
            final_vec[i][j]









#load MNIST data
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')


x_train = np.float32(MNIST_data['x_train'][:])
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32(MNIST_data['x_test'][:])
y_test = np.int32(np.array(MNIST_data['y_test'][:,0]))

MNIST_data.close()


#######################################################################


#Implementing stochastic grad descent algorithm

num_inputs = 28*28

#number of outputs

num_outputs = 10

model = {}

model['W1'] = np.random.randn(num_outputs, num_inputs)/np.sqrt(num_inputs)
model_grads = copy.deepcopy(model)


#######################################################################



