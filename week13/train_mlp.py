import numpy as np
from tensor import Tensor
from layers import Linear, Sequential
from optim import SGD

# setup
np.random.seed(0)
data = Tensor([[0,0],[0,1],[1,0],[1,1]], autograd=True)
target = Tensor([[0],[1],[0],[1]], autograd=True)

# Using the model
print("-- Training the MLP --")
model = Sequential([Linear(2,3), Linear(3,1)])
optim = SGD(parameters=model.get_parameters(), alpha=0.01)
for i in range(10):
    pred = model.forward(data)
    loss = ((pred - target) * (pred - target)).sum(0)
    loss.backward()
    optim.step()
    print(loss)

# Using the bare engine
print("\n-- Training the Bare Engine --")
w = [Tensor(np.random.rand(2,3), autograd=True),
     Tensor(np.random.rand(3,1), autograd=True)]
for i in range(10):
    pred = data.mm(w[0]).mm(w[1])
    loss = ((pred - target) * (pred - target)).sum(0)
    loss.backward()
    for w_ in w:
        w_.data -= w_.grad.data * 0.1
        w_.grad.data *= 0
    print(loss)
