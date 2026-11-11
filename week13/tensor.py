import numpy as np

class Tensor:
    def __init__(self, data, autograd=False, creators=None, creation_op=None, id=None):
        self.data = np.array(data)
        self.creators = creators
        self.creation_op = creation_op
        self.grad = None
        self.autograd = autograd

        if id:
            self.id = id
        else:
            self.id = np.random.randint(0,100000)

        self.children = {}
        if creators is not None:
            for c in creators:
                if self.id not in c.children:
                    c.children[self.id] = 1
                else:
                    c.children[self.id] += 1

    def __repr__(self):
        pass

    def __str__(self):
        return str(self.data)

    def all_children_accounted_for(self):
        for id in self.children:
            if self.children[id] != 0:
                return False
        return True

    def backward(self, grad = None, grad_origin = None):
        pass


    def __add__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data + other.data, creators = [self, other], creation_op = "add")
        else:
            return Tensor(self.data + other.data)




