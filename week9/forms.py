import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(out):
    return out * (1 - out)  # takes the OUTPUT


def tanh(x):
    return np.tanh(x)


def tanh_deriv(out):
    return 1 - out**2  # takes the OUTPUT


def relu(x):
    return (x > 0) * x


def relu_deriv(out):
    return out > 0  # takes the OUTPUT


def softmax(z): # z.shape == (batch, k)
    z = z - np.max(z, axis=1, keepdims=True) # subtract row max
    exp_z = np.exp(z) # exp(onentiate)
    return exp_z / exp_z.sum(axis=1, keepdims=True)  # divide by row sum row-sum
