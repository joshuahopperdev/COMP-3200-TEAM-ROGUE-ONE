import numpy as np
from tensor import Tensor


class Layer:
    def __init__(self):
        self.parameters = []

    def get_parameters(self):
        return self.parameters


class Linear(Layer):
    def __init__(self, n_in, n_out):
        super().__init__()
        W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
        self.weight = Tensor(W, autograd=True)
        self.bias   = Tensor(np.zeros(n_out), autograd=True)
        self.parameters.append(self.weight)
        self.parameters.append(self.bias)

    def forward(self, input):
        return input.__mm__(self.weight) + self.bias.expand(0, len(input.data))


class Sequential(Layer):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def get_parameters(self):
        parameters = []
        for layer in self.layers:
            parameters.extend(layer.get_parameters())
        return parameters
