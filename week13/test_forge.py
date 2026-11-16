import numpy as np
from fontTools.misc.bezierTools import epsilon

from tensor import Tensor
from layers import Sequential, Linear, Tanh, Sigmoid
from optim import SGD

def test_accumulation():
    _, b, _, _, _, f = get_test_tensors()
    f.backward(Tensor(np.array([1,1,1,1,1])))
    assert np.all(b.grad.data == [2,2,2,2,2]), f"b's gradient must accumulate, got {b.grad.data}"

def test_add_grad():
    a, b, c, d = get_test_tensors()[:4]
    g = (a + b) + (c + d)
    g.backward(get_ones(g))
    for x in a, b, c, d:
        assert np.all(x.grad.data == get_ones(x)), f"Leaves need a gradient of ones, got {x.grad.data}"

def test_numeric_grad():
    x = Tensor([2.0, 3.0], autograd=True)
    y = (x*x).sum(0)
    y.backward(get_ones(y))
    assert y.grad.data == [4,6], f"y's gradient must be [4,6], got {y.grad.data}"
    f = lambda n: (n*n).sum()
    epsilon, tolerance = 1e-5, 1e-4
    assert np.allclose([(f(x + epsilon*ei) - f(x - epsilon*ei)) / 2*epsilon for ei in np.array([[0,1],[1,0]])], tolerance), f"Central-difference estimate failed"

def test_step_dec_loss():
    np.random.seed(0)
    data = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]], autograd=True)
    target = Tensor([[0], [1], [0], [1]], autograd=True)

    model = Sequential([Linear(2, 3), Linear(3, 1)])
    optim = SGD(parameters=model.get_parameters(), alpha=0.01)
    prev_loss = None
    for _ in range(10):
        pred = model.forward(data)
        loss = ((pred - target) * (pred - target)).sum(0)
        loss.backward()
        optim.step()
        if prev_loss is not None:
            assert loss < prev_loss, f"Loss must be strictly below the one before, got loss {loss} and previous loss {prev_loss}"
        prev_loss = loss

def test_shape_param_count():
    model = Sequential([Linear(2,3), Tanh(), Linear(3,1), Sigmoid()])
    params = model.get_parameters()
    assert len(params) == 4, f"Model must have 4 Tensors (2 weights + 2 biases), got {len(params)}"

def test_wait():
    pass

def test_mm():
    pass

def test_whole_grad():
    pass

# -- Helpers --
def get_test_tensors() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    a = Tensor([1, 2, 3, 4, 5], autograd=True)
    b = Tensor([2, 2, 2, 2, 2], autograd=True)
    c = Tensor([5, 4, 3, 2, 1], autograd=True)

    d = a + b
    e = b + c
    f = d + e

    return a, b, c, d, e, f

def get_ones(ns: Tensor):
    return np.ones_like(ns.data)

if __name__ == "__main__":
    tests = [name for name in dir() if name.startswith("test_")]
    for test_name in sorted(tests):
        test_func = globals()[test_name]
        try:
            test_func()
            print(f" PASS: {test_name}")
        except AssertionError as e:
            print(f" FAIL (AssertionError): {test_name} -- {e}")
