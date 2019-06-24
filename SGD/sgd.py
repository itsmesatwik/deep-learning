import numpy as np

import h5py
# import time
import copy
from random import randint



def print_digit(digit):
    count = 0
    digit_str = ""
    for i in range(28):
        for j in range(28):
            if (digit[count] > 0.4):
                digit_str += '#'
            elif (digit[count] > 0.78):
                digit_str += '+'
            else:
                digit_str += '.'
            count += 1
        digit_str += '\n'
    return(digit_str)


# SOFFTMAX FUNCTION
def f_softmax(vec):
    ret = np.exp(vec)
    sum_ = sum(ret)
    for i in range(10):
        ret[i] /= np.float(sum_)
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
            ret[i] += (x[j] * theta[i][j])

    return ret


def matrix_sum(x, y):
    sum = [[0]*784 for _ in range(10)]
    for i in range(10):
        for j in range(784):
            sum[i][j] = x[i][j]+y[i][j]

    return sum


def loss_grad(theta, x, y):
    e_vec = e_func(y)
    theta_x = matrix_mult(theta, x)
    softmax_vec = f_softmax(theta_x)
    #intermed_vec = [0] * 10
    for i in range(10):
        e_vec[i] -= softmax_vec[i]
    final_vec = [[0] * 784 for _ in range(10)]
    for i in range(10):
        for j in range(784):
            final_vec[i][j] = e_vec[i] * x[j]

    return final_vec


def theta_update(theta, grad, ALPHA):
    #updated_theta = [[0] * 784 for _ in range(10)]
    for i in range(10):
        for j in range(784):
            theta[i][j] += (ALPHA * grad[i][j])

    return theta


# def mini_batch(x, y, theta,batch_pt):
#     grad_sum = [[0]*784 for _ in range(10)]
#     for i in range(10):
#         #print(i+batch_pt)
#         new_grad = loss_grad(theta, x[min((i + batch_pt), len(x))][:], y[min((i + batch_pt), len(x))])
#         grad_sum = matrix_sum(grad_sum, new_grad)

#     for i in range(10):
#         for j in range(784):
#             grad_sum[i][j] = (grad_sum[i][j] / np.float(10))
#     return grad_sum


# load MNIST data
MNIST_data = h5py.File('../MNISTdata.hdf5', 'r')

x_train = np.float32(MNIST_data['x_train'][:])
y_train = np.int32(np.array(MNIST_data['y_train'][:, 0]))
x_test = np.float32(MNIST_data['x_test'][:])
y_test = np.int32(np.array(MNIST_data['y_test'][:, 0]))


MNIST_data.close()

#######################################################################


#print(len(x_train[0]))

# Implementing stochastic grad descent algorithm

num_inputs = 28 * 28

# number of outputs

num_outputs = 10

#model = {'W1': np.random.randn(num_outputs, num_inputs) / np.sqrt(num_inputs)}

#model_grads = copy.deepcopy(model)

theta_0 = np.random.randn(num_outputs,num_inputs) / np.sqrt(num_inputs)  # COMPLETE THIS

ALPHA_VAL = 0.003
EPOCH = 1
#print(EPOCH)
for ep in range(EPOCH):
    #print(ep)
    batch_len = 10
    shuffle = np.arange(x_train.shape[0])
    np.random.shuffle(shuffle)
    shuffle_x = x_train[shuffle]
    shuffle_y = y_train[shuffle]

    grad_sum = [[0]*784 for _ in range(10)]
    for i in range(3000):
        #print(i+batch_pt)
        new_grad = loss_grad(theta_0, shuffle_x[i][:], shuffle_y[i])
        theta_0 = theta_update(theta_0, new_grad, ALPHA_VAL)
        #grad_sum = matrix_sum(grad_sum, new_grad)

    # for bc in range(0, len(x_test), batch_len):


    #     gradient_desc = mini_batch(shuffle_x, shuffle_y, theta_0, bc)

    #     #   gradient_desc = loss_grad(theta_0, shuffle_x, shuffle_y)
    #     theta_0 = theta_update(theta_0, gradient_desc, ALPHA_VAL)


#######################################################################

# Test Data

total_correct = 0

with open("outputs.txt", 'w') as outfile:
    for n in range( len(x_test)):
        y = y_test[n]
        x = x_test[n][:]
        p = matrix_mult(theta_0, x)
        prediction = np.argmax(p)
        if (prediction == y):
            total_correct += 1
        else:
            outfile.write(print_digit(x_test[n]))
            outfile.write("\nPrediction: {}, Label: {}\n".format(prediction, y))

print (total_correct/np.float(len(x_test)))