import numpy as np

def relu(input):
    return input * (input > 0)

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
    weight_mats = [0] * (len(layer_lens) - 1)
    for i in range(len(layer_lens) - 1):
        weight_mats[i] = 2*np.random.random((layer_lens[i], layer_lens[i+1])) - 1
        print(weight_mats[i])

    err_hist = [0] * epochs

#    for epoch in range(epochs):
#        for i in range(len(tells)):






#forward(np.array([[0, 1, 2]]), [np.array([2, 4, 9]), np.array([3])])
#train_from_diagram(0, 0, 0, 0, 0, layer_lens = [3, 8, 4, 1])

