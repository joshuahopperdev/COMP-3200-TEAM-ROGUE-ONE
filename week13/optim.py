class SGD:
    def __init__(self, parameters, alpha):
        self.parameters = parameters
        self.alpha = alpha

    def step(self, zero=True):
        for p in self.parameters:
            p.data -= p.grad.data * self.alpha
            if zero:
                p.grad.data *= 0
