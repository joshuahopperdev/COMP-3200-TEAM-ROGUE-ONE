import numpy as np
from tensor import Tensor
from layers import Linear, Sequential, Tanh, Sigmoid, MSELoss
from optim import SGD

np.random.seed(0)
data = Tensor([[0,0],[0,1],[1,0],[1,1]], autograd=True) # after the seed,
target = Tensor([[0],[1],[0],[1]], autograd=True) # before the model
model = Sequential([Linear(2,3), Tanh(), Linear(3,1), Sigmoid()])
criterion = MSELoss()
optim = SGD(parameters=model.get_parameters(), alpha=1)

# 10 iterations: pred -> loss=criterion.forward(pred,target)
# -> loss.backward() -> optim.step()