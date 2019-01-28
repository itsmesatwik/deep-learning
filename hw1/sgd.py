import numpy as np

import h5py
# import time
import copy
from random import randint


# SOFFTMAX FUNCTION
def f_softmax(vec):
    ret = [0] * 10
    for i in vec:
        vec[i] = np.exp(vec[i])
    sum_ = sum(vec)
    for i in vec:
        ret[i] = vec[i] / sum_
    return ret


# E(Y) FUNCTION
def e_func(elem):
    ret = [0] * 10
    ret[elem] = 1
    return ret


# MULTIPLYING THETA AND X_VEC
def matrix_mult(theta, x):
    ret = [0] * 10
    for i in range(10):
        for j in range(784):
            ret[i] += x[j] * theta[i][j]

    return ret


def matrix_sum(x, y):
    sum = [[0]*10 for _ in range(784)]
    for i in range(784):
        for j in range(10):
            sum[i][j] = x[i][j]+y[i][j]

    return sum


def loss_grad(theta, x, y):
    e_vec = e_func(y)
    theta_x = matrix_mult(theta, x)
    softmax_vec = f_softmax(theta_x)
    #intermed_vec = [0] * 10
    for i in range(10):
        e_vec[i] -= softmax_vec[i]
    final_vec = [[0] * 10 for _ in range(784)]
    for i in range(784):
        for j in range(10):
            final_vec[i][j] = e_vec[j] * x[i]

    return final_vec


def theta_update(theta, grad, ALPHA):
    updated_theta = [[0] * 10 for _ in range(784)]
    for i in range(784):
        for j in range(10):
            updated_theta[i][j] = theta[i][j] - ALPHA * grad[i][j]

    return updated_theta


def mini_batch(x, y, theta, M_):
    grad_sum = [[0]*784 for _ in range(10)]
    for i in range(M_):
        new_grad = loss_grad(theta, x[i], y[i])
        grad_sum = matrix_sum(grad_sum, new_grad)

    for i in range(10):
        for j in range(784):
            grad_sum[i][j] = grad_sum[i][j] / M_
    return grad_sum


# load MNIST data
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')

x_train = np.float32(MNIST_data['x_train'][:])
y_train = np.int32(np.array(MNIST_data['y_train'][:, 0]))
x_test = np.float32(MNIST_data['x_test'][:])
y_test = np.int32(np.array(MNIST_data['y_test'][:, 0]))

MNIST_data.close()

#######################################################################


# Implementing stochastic grad descent algorithm

num_inputs = 28 * 28

# number of outputs

num_outputs = 10

model = {'W1': np.random.randn(num_outputs, num_inputs) / np.sqrt(num_inputs)}

model_grads = copy.deepcopy(model)

r = np.random.random_sample()  # COMPLETE THIS
theta_0 = [[r] * 784 for _ in range(10)]  # THETA INITIALIZED HERE

ALPHA_VAL = 0.003

EPOCH = 100
M = len(y_train) / 100

for ep in range(EPOCH):
    shuffle = np.arange(x_train.shape[0])
    np.random.shuffle(shuffle)
    shuffle_x = x_train[shuffle]
    shuffle_y = y_train[shuffle]

    gradient_desc = mini_batch(shuffle_x, shuffle_y, theta_0, M)

 #   gradient_desc = loss_grad(theta_0, shuffle_x, shuffle_y)
    theta_0 = theta_update(theta_0, gradient_desc, ALPHA_VAL)

#######################################################################
