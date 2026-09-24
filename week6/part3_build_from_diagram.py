
# I don't like the idea of writing a function for a particular shape; 
# this function works for any shape, but defaults to the assignment spec shape.
def train_from_diagram(tells, strike, alpha, epochs, seed, layer_lens = [3, 8, 4, 1]):
    # initialize weight matrices, sizes calculated from layer_lens.
    # so for e.g. layer_lens of (3, 4, 1), this would initialize 2
    # matrices, a 3x4 and a 4x1
    weight_mats = [0] * (len(layer_lens) - 1)
    for i in range(len(layer_lens) - 1):
        weight_mats[i] = [[0] * layer_lens[i+1] for _ in range(layer_lens[i])]
        print(weight_mats[i])

