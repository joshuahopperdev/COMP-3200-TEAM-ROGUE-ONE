import numpy as np

from tensor import Tensor
from layers import Sequential, Linear, Tanh, Sigmoid, MSELoss
from optim import SGD

def test_accumulation():
    _, b, _, _, _, f = get_test_tensors()
    f.backward(Tensor(np.array([1,1,1,1,1])))
    assert np.all(b.grad.data == np.array([2,2,2,2,2])), f"b's gradient must accumulate, got {b.grad.data}"

def test_add_grad():
    a, b, c, d = get_test_tensors()[:4]
    g = (a + b) + (c + d)
    g.backward(get_ones(g))
    for x in a, b, c, d:
        assert np.all(x.grad.data == get_ones(x).data), f"Leaves need a gradient of ones, got {x.grad.data}"

def test_numeric_grad():
    x = Tensor([2.0, 3.0], autograd=True)
    y = (x*x).sum(0)
    y.backward(Tensor(np.array([1, 1])))
    assert np.all(x.grad.data == np.array([4,6])), f"y's gradient must be [4,6], got {y.grad.data}"
    f = lambda n: (n*n).sum(0)
    epsilon, tolerance = 1e-5, 1e-4
    cdiff_est = [(f(x.data + epsilon*ei) - f(x.data - epsilon*ei)) / 2*epsilon for ei in np.array([[0,1],[1,0]])]
    assert np.allclose(cdiff_est, 0, atol=tolerance, rtol=tolerance), f"Central-difference estimate failed, got {cdiff_est}"

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
            assert np.all(loss.data < prev_loss.data), f"Loss must be strictly below the one before, got loss {loss} and previous loss {prev_loss}"
        prev_loss = loss

def test_shape_param_count():
    model = Sequential([Linear(2,3), Tanh(), Linear(3,1), Sigmoid()])
    params = model.get_parameters()
    assert len(params) == 4, f"Model must have 4 Tensors (2 weights + 2 biases), got {len(params)}"

def test_wait():
    a, _, c = get_test_tensors()[:3]
    x = Tensor([1, 2, 3, 4, 5], autograd=True)
    y = Tensor([1, 1, 1, 1, 1], autograd=True)
    s = x + y
    loss = (s + a) + (s + c)
    loss.backward(Tensor(np.array([1,1,1,1,1])))
    assert np.all(x.grad.data == np.array([2,2,2,2,2])), f"x's gradient must be [2,2,2,2,2], got {x.grad.data}"

def test_mm():
    X = Tensor([[1., 2.], [3., 4.]], autograd=True)
    W = Tensor([[0.5, -0.5], [1., 0.24]], autograd=True)
    A = Tensor([[1., 0.], [2., 1.]], autograd=True)
    B = Tensor([[0., 1.], [1., 3.]], autograd=True)
    h = X.mm(W)                                     # made by an op...
    (h.mm(A) + h.mm(B)).sum(0).sum(0).backward()    # ...and used twice
    assert W.grad.data.tolist() == [[8.0, 28.0], [12.0, 42.0]], f"W's gradient must match the example, got {W.grad.data.tolist()}"

def test_whole_grad():
    np.random.seed(0)
    data = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]], autograd=True)  # after the seed,
    target = Tensor([[0], [1], [0], [1]], autograd=True)  # before the model
    model = Sequential([Linear(2, 3), Tanh(), Linear(3, 1), Sigmoid()])
    criterion = MSELoss()
    optim = SGD(parameters=model.get_parameters(), alpha=1)
    for i in range(10):
        pred = model.forward(data)
        loss = criterion.forward(pred, target)
        loss.backward()
        optim.step()

    # predict and get the loss on the trained model before nudging a weight entry
    loss = lambda: criterion.forward(model.forward(data), target)
    l = loss()

    # setup
    epsilon, tolerance = 1e-5, 1e-4
    # "one_hot" or essentially e1, the first standard basis
    # vector for the first weight matrix scaled by the epsilon
    one_hot = np.zeros_like(model.get_parameters()[0].data)
    one_hot[0] += epsilon

    # down
    model.get_parameters()[0] -= Tensor(one_hot)
    ld = loss()

    # up
    # double the epsilon to negate the negative nudge
    # and nudge it the other way--ie positive
    model.get_parameters()[0] += Tensor(2*one_hot)
    lu = loss()

    cdiff_est = (lu - ld).data / 2*epsilon
    assert np.abs(cdiff_est) <= tolerance, f"Central difference failed, got {cdiff_est}"

# -- Helpers --
def get_test_tensors() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    a = Tensor([1, 2, 3, 4, 5], autograd=True)
    b = Tensor([2, 2, 2, 2, 2], autograd=True)
    c = Tensor([5, 4, 3, 2, 1], autograd=True)

    d = a + b
    e = b + c
    f = d + e

    return a, b, c, d, e, f

def get_ones(ns: Tensor) -> Tensor:
    return Tensor(np.ones_like(ns.data))

if __name__ == "__main__":
    tests = [name for name in dir() if name.startswith("test_")]
    for test_name in sorted(tests):
        test_func = globals()[test_name]
        try:
            test_func()
            print(f" PASS: {test_name}")
        except AssertionError as e:
            print(f" FAIL (AssertionError): {test_name} -- {e}")
