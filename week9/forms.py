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


def softmax(z):
    return np.exp(z - np.max(z)) / np.sum(
        np.exp(z - np.max(z)), axis=1, keepdims=True
    )  # subtract row max, exp, / row-sum; (batch, k)
