import numpy as np
from tensor import Tensor


class Layer:
    def __init__(self):
        self.parameters = []

    def get_parameters(self):
        return self.parameters


class Linear(Layer):
    def __init__(self, n_in, n_out, bias = True):
        super().__init__()
        W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
        self.weight = Tensor(W, autograd=True)
        if bias:
            self.bias = Tensor(np.zeros(n_out), autograd=True)
        else:
            self.bias = None
        self.parameters.append(self.weight)
        if bias:
            self.parameters.append(self.bias)

    def forward(self, input):
        if self.bias:
            return input.mm(self.weight) + self.bias.expand(0, len(input.data))
        else:
            return input.mm(self.weight)

class Embedding(Layer):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        
        self.weight = Tensor((np.random.rand(vocab_size, dim) - 0.5) / dim, autograd=True)
        self.parameters.append(self.weight)

    def forward(self, input):
        return self.weight.index_select(input)

class CrossEntropyLoss:
    def __init__(self):
        super().__init__()
        
    def forward(self, input, target):
        return input.cross_entropy(target)

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


class Tanh(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        return input.tanh()

class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.sigmoid()

class MSELoss(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return ((pred-target)*(pred-target)).sum(0)

