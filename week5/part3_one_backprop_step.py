from part2_forward_hidden import forward
import numpy as np

# 4 sensings, 3 binary tells each
tells = np.array([[1, 0, 1], # foot shift, no guard drop, exhale
[0, 1, 1], # no shift, guard drop, exhale
[0, 0, 1], # only exhale
[1, 1, 1]]) # all three (the bluff)
# Ground truth: 1 = strike imminent, 0 = they will hold
strike = np.array([[1, 1, 0, 0]]).T # column vector, shape (4, 1)

np.random.seed(1)

hidden_size = 4
weights_0_1 = 2 * np.random.random((3, hidden_size)) - 1 # 3 x 4
weights_1_2 = 2 * np.random.random((hidden_size, 1)) - 1 # 4 x 1

def relu2deriv(y):
    return y > 0


# important note: the layer_2 this returns is the value before
# the update step, pre-modification of weights. The layer_2_error
# this returns, ditto

def one_step(layer_0, target, weights_0_1, weights_1_2, alpha):

    # forward pass
    layer_1, layer_2 = forward(layer_0, weights_0_1, weights_1_2)


    # layer 2 delta is easy: it's the difference between pred and true
    layer_2_delta = layer_2 - target

    # squaring scalars is also easy
    layer_2_error = layer_2_delta**2


    # this one's muddlier and I really, really need to work through
    # reverse chain rule again some time to make sure I have this down.
    # For e.g. our 1x1 layer_2_delta and our 4x1 weights_1_2.T, we'll
    # get a 1x4 scaled by layer_2_delta and the derivative of relu
    layer_1_delta = layer_2_delta @ weights_1_2.T * relu2deriv(layer_1)

    # now it's backprop time!

    updated_weights_0_1 = weights_0_1 - alpha * layer_0 * layer_1_delta
    updated_weights_1_2 = weights_1_2 - alpha * layer_1 * layer_2_delta

    
    
    return updated_weights_0_1, updated_weights_1_2, layer_2, layer_2_error