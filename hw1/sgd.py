import numpy as np

import h5py
import time 
import copy 
from random import randint

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



