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
        return str(self.data)

    def __str__(self):
        return str(self.data)

    def all_children_accounted_for(self):
        for id in self.children:
            if self.children[id] != 0:
                return False
        return True

    def backward(self, grad = None, grad_origin = None):
        # If self.autograd is false, return a plain tensor takes no part in the walk
        if not self.autograd:
            return

        # If grad is None, start from ones
        if grad is None:
            grad = Tensor(np.ones_like(self.data))

        # If grad_origin is given, a child is reporting. 
        # If that child’s counter is already zero, raise Exception("cannot backprop more than once"); 
        # otherwise decrement it
        if grad_origin:
            if self.children[grad_origin.id] == 0:
                raise Exception("cannot backprop more than once")
            else:
                self.children[grad_origin.id] -= 1

        # Accumulate into self.grad with += rather than overwriting.
        if self.grad is None:
            self.grad = grad
        elif self.grad:
            self.grad += grad

        # Only recurse into creators once all_children_accounted_for() is true, or grad_origin is None
        if self.all_children_accounted_for() or grad_origin is None:
            if self.creation_op is None:
                return
            elif self.creation_op == "add":
                self.creators[0].backward(grad = self.grad, grad_origin = self)
                self.creators[1].backward(grad = self.grad, grad_origin = self)
            elif self.creation_op == "neg":
                self.creators[0].backward(-self.grad, self)
            elif self.creation_op == "sub":
                self.creators[0].backward(self.grad, self)
                self.creators[1].backward(self.grad, self)
            elif self.creation_op == "mul":
                self.creators[0].backward(self.grad * self.creators[1], self)
                self.creators[1].backward(self.grad * self.creators[0], self)
            elif self.creation_op == "mm":
                c0, c1 = self.creators
                c0.backward(Tensor(self.grad.data.dot(c1.data.T)), self)
                c1.backward(Tensor(c0.data.T.dot(self.grad.data)), self)
            elif self.creation_op == "transpose":
                self.creators[0].backward(self.grad.T, self)
            elif "sum" in self.creation_op:
                dim = int(self.creation_op.split("_")[1])
                ds = self.creators[0].data.shape[dim]
                self.creators[0].backward(self.grad.expand(dim, ds), self)
            elif "expand" in self.creation_op:
                dim = int(self.creation_op.split("_")[1])
                self.creators[0].backward(self.grad.sum(dim), self)

    def __add__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data + other.data, autograd = True, creators = [self, other], creation_op = "add")
        else:
            return Tensor(self.data + other.data)

    def __sub__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data - other.data, autograd = True, creators = [self, other], creation_op = "sub")
        else:
            return Tensor(self.data - other.data)

    def __mul__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data * other.data, autograd = True, creators = [self, other], creation_op = "mul")
        else:
            return Tensor(self.data * other.data)

    def __neg__(self):
        if self.autograd:
            return Tensor(self.data * -1, autograd = True, creators = [self], creation_op = "neg")
        else:
            return Tensor(self.data * -1)

    def __matmul__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data.dot(other.data), autograd = True, creators = [self, other], creation_op = "mm")
        else:
            return Tensor(self.data.dot(other.data))

    @property
    def T(self):
        return self.transpose()

    def mm(self, other):
        return self @ other
    
    def transpose(self):
        if self.autograd:
            return Tensor(self.data.T, autograd = True, creators = [self], creation_op = "transpose")
        else:
            return Tensor(self.data.T)

    def sum(self, dim):
        if self.autograd:
            return Tensor(np.sum(self.data, dim), autograd = True, creators = [self], creation_op = f"sum_{dim}")
        else:
            return Tensor(np.sum(self.data, dim))

    def expand(self, dim, n):
        trans_cmd = list(range(0, len(self.data.shape)))
        trans_cmd.insert(dim, len(self.data.shape))
        new_data = self.data.repeat(n).reshape(list(self.data.shape) + [n]).transpose(trans_cmd)
        
        if not self.autograd:
            return Tensor(new_data)
        
        return Tensor(new_data, autograd = True, creators = [self], creation_op = f"expand_{dim}")
