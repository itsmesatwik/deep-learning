import numpy as np

import h5py
# import time
import copy
from random import randint


# Softmax Function

def softmax(vec):
    #assuming vec is 1-d
    exp_vec = np.exp(vec)
    vsum = sum(exp_vec)
    for i in range(len(vec)):
        exp_vec[i] /= np.float32(vsum)

    return exp_vec


def linear_step(W, x, b1):
    Wx = np.matmul(W,x)
    return np.sum(Wx, b1, axis=0)



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