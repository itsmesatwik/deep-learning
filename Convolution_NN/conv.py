import numpy as np

import h5py
import time
import copy
from random import randint


############################
#   SINGLE HIDDEN LAYER    #
#     MULTIPLE KERNELS     #
############################


# Elementwise ReLU nonlinearity to produce the hidden layer
def sigma(Z):
    return np.maximum(Z, 0, Z)


# Convolution of matrix X with the filter K
def convolution(X, K, stride):
    rs = cs = X.itemsize
    X_slices = np.lib.stride_tricks.as_strided(X, shape=(
    K.shape[0], K.shape[1], X.shape[0] - K.shape[0] + 1, X.shape[1] - K.shape[1] + 1),
                                               strides=(X.shape[0] * rs, cs, rs, X.shape[1] * cs))
    X_s = X_slices.reshape(X_slices.shape[2] * X_slices.shape[3], X_slices.shape[0] * X_slices.shape[1])
    K_dot = K.reshape(K.shape[0] * K.shape[1], K.shape[2])
    Conv = np.dot(X_s, K_dot)
    Conv = Conv.reshape(X_slices.shape[2], X_slices.shape[3], Conv.shape[1])
    # Z = np.zeros((X.shape[0]-K.shape[0]+1, X.shape[1]-K.shape[1]+1, K.shape[2]))
    # for m in range(0, Z.shape[0]-1, stride):
    #    for n in range(0, Z.shape[1]-1, stride):
    #        for k in range(0, Z.shape[2]):
    #            Z[m,n,k] = np.vdot(X[m:m+K.shape[0], n:n+K.shape[1]], K[:,:,k])
    return Conv


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
def softmax(vec):
    # assuming vec is 1-d
    exp_vec = np.exp(vec)
    vsum = np.float32(1 / np.float32(sum(exp_vec)))
    exp_vec *= vsum
    return exp_vec


def sigma_prime(Z):
    return (1 / 1 + np.exp(-1 * Z))


def partial_K(sigma_prime, delta, X):
    return convolution(X, (sigma_prime * delta), 1)


def partial_U(soft, e_Y):
    return -1 * (e_Y - soft)


def partial_W(partial_b, H):
    ret = np.zeros((partial_b.shape[0], H.shape[0], H.shape[1], H.shape[2]))
    for i in range(ret.shape[0]):
        ret[i] = partial_b[i] * H
    return ret


def delta(W, partial_u):
    delt = np.zeros((W.shape[1], W.shape[2], W.shape[3]))
    for i in range(partial_u.shape[0]):
        delt += partial_u[i] * W[i]
    return delt


def param_update(param, ALPHA, grad):
    return param - (ALPHA * grad)


# def final_param(param):
#     sum = param[0]
#     for i in range(1, param.shape[0]):
#         sum += param[i]

#     return sum


MNIST_data = h5py.File('../MNISTdata.hdf5', 'r')

x_train = np.float32(MNIST_data['x_train'][:])
y_train = np.int32(np.array(MNIST_data['y_train'][:, 0]))
x_test = np.float32(MNIST_data['x_test'][:])
y_test = np.int32(np.array(MNIST_data['y_test'][:, 0]))


MNIST_data.close()


#######################################################################
def NN(ALPHA, filter_dim, num_filters):
    print(ALPHA, filter_dim, num_filters)
    # IMPLEMENT Neural Network

    num_inputs = 28 * 28
    input_dim = 28
    num_outputs = 10
    K = (2 * np.random.random_sample((filter_dim, filter_dim, num_filters)) - 1) / 1000
    b = (2 * np.random.random_sample((num_outputs)) - 1) / 1000
    W = (2 * np.random.random_sample(
        (num_outputs, input_dim - filter_dim + 1, input_dim - filter_dim + 1, num_filters)) - 1) / 1000
    accuracy = 0.0
    runtime = time.time()
    epoch = 1
    acc_0 = 0
    for ep in range(1, 15):
        start = time.time()
        print(ep)
        shuffle = np.arange(x_train.shape[0])
        np.random.shuffle(shuffle)
        shuffle_x = x_train[shuffle]
        shuffle_y = y_train[shuffle]
        for i in range(len(shuffle_x)):
            X = (shuffle_x[i]).reshape((28, 28))

            # FORWARD STEP
            Z = convolution(X, K, 1)
            H = sigma(Z)
            U = hidden_linear(H, W, b)
            soft_x = softmax(U)
            e_y = e(shuffle_y[i], num_outputs)

            # CALCULATE PARTIAL DERIVATIVES
            par_u = partial_U(soft_x, e_y)
            par_w = partial_W(par_u, H)
            delt = delta(W, par_u)
            par_k = partial_K(sigma_prime(Z), delt, X)

            # UPDATE PARAMETERS

            K = param_update(K, ALPHA, par_k)
            W = param_update(W, ALPHA, par_w)
            b = param_update(b, ALPHA, par_u)

        total_correct = 0
        for n in range(len(x_test)):
            y = y_test[n]
            x = x_test[n][:]
            x = x.reshape((28, 28))
            Z = convolution(x, K, 1)
            H = sigma(Z)
            U = hidden_linear(H, W, b)
            soft_x = softmax(U)
            prediction = np.argmax(soft_x)
            if (prediction == y):
                total_correct += 1
        accuracy = (total_correct / np.float(len(x_test)))
        print("Accuracy: ", accuracy)
        if (epoch == 1):
            acc_0 = accuracy

        if (accuracy > 0.94):
            break
        if (accuracy < acc_0):
            print("Param reset at epoch ", epoch)
            K = (2 * np.random.random_sample((filter_dim, filter_dim, num_filters)) - 1) / 1000
            b = (2 * np.random.random_sample((num_outputs)) - 1) / 1000
            W = (2 * np.random.random_sample(
                (num_outputs, input_dim - filter_dim + 1, input_dim - filter_dim + 1, num_filters)) - 1) / 1000
        epoch += 1
    # EPOCH ENDED
    print("Accuracy Achieved at Epoch: ", epoch)
    print("Time Taken Per Epoch: ", (time.time() - start))
    print("Total run time: ", (time.time() - runtime))
    print("Accuracy: ", accuracy)


NN(0.0007, 5, 16)
