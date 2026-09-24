import numpy as np


# 4 sensings, 3 binary tells each
tells = np.array([[1, 0, 1], # foot shift, no guard drop, exhale
[0, 1, 1], # no shift, guard drop, exhale
[0, 0, 1], # only exhale
[1, 1, 1]]) # all three (the bluff)

# Ground truth: 1 = strike imminent, 0 = they will hold
strike = np.array([[1, 1, 0, 0]]).T # column vector, shape (4, 1)


def relu(input):
    return input * (input > 0)

def relu2deriv(input):
    return input > 0

# works on an arbitrary length of net, assuming relu on all but the last step
# input as row vectors, matrices as input dims x output dims
# assumes input and all entries in weight_mats are numpy arrays
# (not weight_mats itself, which is not)
def forward(input, weight_mats):
    saved = [0] * (len(weight_mats) + 1)
    saved[0] = input
    cur = input
    # 1 less loop than there are weight matrices, see below
    for i in range(len(weight_mats) - 1):
        cur = relu(cur @ weight_mats[i])
        saved[i+1] = cur
    # no relu on the last step
    cur = cur @ weight_mats[-1]
    saved[-1] = cur
    return saved


# I don't like the idea of writing a function for a particular shape; 
# this function works for any shape, but defaults to the assignment spec shape.
def train_from_diagram(tells, strike, alpha, epochs, seed, layer_lens = [3, 8, 4, 1]):
    # initialize weight matrices, sizes calculated from layer_lens.
    # so for e.g. layer_lens of (3, 4, 1), this would initialize 2
    # matrices, a 3x4 and a 4x1
    np.random.seed(seed)
    weight_mats = [0] * (len(layer_lens) - 1)
    for i in range(len(layer_lens) - 1):
        weight_mats[i] = 2*np.random.random((layer_lens[i], layer_lens[i+1])) - 1
        print(weight_mats[i])

    # make an empty error history
    # for each epoch.
    err_hist = [0] * epochs

    # for each epoch...
    for epoch in range(epochs):
        # for each tell...
        for i in range(len(tells)):
            # run a forward pass, saving all the layers
            # including layer 0, the input
            layers = forward(tells[i:i+1], weight_mats)
            # make an empty deltas list
            deltas = [0] * len(weight_mats)
            # set the last delta, since its process is a bit special
            deltas[-1] = layers[-1] - strike[i:i+1]

            # save our error before we go on, since we only need 
            # this delta to calculate it
            err_hist[epoch] = deltas[-1]**2

            # set all the other deltas (1 less iteration than there
            # are deltas, since we already set the last one)
            for i in range(len(deltas) - 1):
                deltas[-i-2] = deltas[-i-1] @ weight_mats[-i-1].T * relu2deriv(layers[-i-2])
                print(deltas[-i-2])









#forward(np.array([[0, 1, 2]]), [np.array([2, 4, 9]), np.array([3])])
#train_from_diagram(tells, strike, 0, 2, 4, layer_lens = [3, 2, 1])

