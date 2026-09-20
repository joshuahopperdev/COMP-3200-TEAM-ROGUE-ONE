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


# derivative of relu at y > 0 is 1, else 0, so
# just return the truth value of each entry being
# greater than 0
def relu2deriv(y):
    return y > 0


# important note: the layer_2 this returns is the value before
# the update step, pre-modification of weights. The layer_2_error
# this returns, ditto

# layer_0 and target will be row vectors, specifically 1x3 and 1x1 here

def one_step(layer_0, target, weights_0_1, weights_1_2, alpha):

    # forward pass
    # also row vectors, here 1x4 and 1x1
    layer_1, layer_2 = forward(layer_0, weights_0_1, weights_1_2)

    # layer 2 delta is easy: it's the difference between pred and true
    layer_2_delta = layer_2 - target

    # squaring scalars is also easy
    layer_2_error = layer_2_delta**2


    # this one's muddlier and I really, really need to work through
    # reverse chain rule again some time to make sure I have this down.

    layer_1_delta = layer_2_delta @ weights_1_2.T * relu2deriv(layer_1)

    

    

    # now it's backprop time!

    # the weight update is alpha times the outer product of our 3 inputs and
    # our 4 deltas: each column is the relevant delta x the input set,
    # so each column of the weights (corresponding to the operations on a set
    # of inputs to get one of the outputs) will be modified by the inputs x
    # the delta relating to that output
    
    # bleh, np.outer() does of course work on both row and column vectors in 
    # any configuration...
    updated_weights_0_1 = weights_0_1 - alpha * np.outer(layer_0, layer_1_delta)

    # same logic, but more trivial because it's got more 1s
    updated_weights_1_2 = weights_1_2 - alpha * np.outer(layer_1, layer_2_delta)

    
    return updated_weights_0_1, updated_weights_1_2, layer_2, layer_2_error



def main():

    # run it! on sensing 0
    new_weights_0_1, new_weights_1_2, old_layer_2, old_layer_2_error = one_step(tells[0:1], strike[0:1], weights_0_1, weights_1_2, alpha = 0.2)

    # calculate our new layer 2 and layer 2 error
    _, new_layer_2 = forward(tells[0], new_weights_0_1, new_weights_1_2)
    new_layer_2_error = (new_layer_2 - strike[0])**2


    # prints, yay
    print(f"layer_2 before update was {old_layer_2}, and after the update was {new_layer_2}")
    print(f"Layer 2's error before update was {old_layer_2_error}, and after the update was {new_layer_2_error}\n\n")
    print(f"Weights of layer 1 before update had shape {weights_0_1.shape}")
    print(f"Weights of layer 1 after update had shape {new_weights_0_1.shape}\n")
    print(f"Weights of layer 2 before update had shape {weights_1_2.shape}")
    print(f"Weights of layer 2 after update had shape {new_weights_1_2.shape}")

if __name__ == "__main__":
    main()



# Hand-verification:
# Old weight in row 2 of hidden layer 2 is 0.75623487, new weight is 0.8192639
# alpha = 0.2,
# layer_1 = [0, 0.51828245, 0, 0]
# layer_2_delta = [-0.60805673]
# only row 2 of layer 1 matters since we're only verifying for row 2
# 0.8192639 ~= 0.75623487 - 0.2 * 0.51828245 * -0.60805663
# confirmed

# "explain in your own words why layer_1_delta uses weights_1_2.T (not
# weights_1_2) and why we multiply by relu2deriv(layer_1)."

# I originally did this with column vectors, so my more systematic
# and fun work is below, now inapplicable.
# And, um, the answer is sort of just dimensional analysis.
# If this were layer_2_delta @ weights_1_2, we'd be multiplying a
# 1x1 by a 4x1, an incoherent multiplication; we transpose so we can
# multiply a 1x1 by a 1x4 to get a 1x4.


# We multiply by relu2deriv to, well, scale by the derivative of this
# part of the function chain. In this case, it will wipe out all
# modification of weights that didn't actually provide meaningful
# informational content to our final answer. It's practically the
# same thing as a freeze step!




# expunged:

# One implied facet of this assignment was using row vectors
# for lots of stuff; I did not do so, I stuck with all column
# vectors so I could just use forward() from part 2. My explanation
# below reflects this.


# Using dimensions of 1 muddles this, so let's use a bigger example:
# 3 inputs from hidden layer 1 to our final 5 outputs, means a 3x5
# final layer weights_1_2. Our layer_2_delta will be a 5x1 matrix,
# so weights_1_2 @ layer_2_delta will be a 3x5 x 5x1 = 3x1 vector,
# with each row representing the dot product of the output deltas
# and how much each input would have affected each of them
