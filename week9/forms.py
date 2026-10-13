import numpy as np

def sigmoid(x): ... # 1 / (1 + e^-x)
def sigmoid_deriv(out): ... # out * (1 - out) -- takes the OUTPUT
def tanh(x): ... # np.tanh(x)
def tanh_deriv(out): ... # 1 - out**2 -- takes the OUTPUT
def relu(x): ... # (x > 0) * x
def relu_deriv(out): ... # (out > 0)
def softmax(z): ... # subtract row max, exp, / row-sum; (batch, k)