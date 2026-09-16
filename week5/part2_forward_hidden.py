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


# only really designed for numpy vectors, but it might in theory work on other
# things? For all elements, multiplies them by the truth value of "is it greater
# than 0", so multiplies >0 nums by 1, <0 nums by 0. Returns the same vector.
def relu(x):
    zeroed = x * (x > 0)
    return zeroed

#results experimentally verified very sloppily by using np.ones for the weights
# seems to work
def forward(layer_0, weights_0_1, weights_1_2):
    # first layer is equal to relu of the product of the transpose of the weight matrix
    # and the input. So for e.g. our 3x4 weight matrix and our 3x1 input vector
    # here, we're multiplying a 4x3 matrix by a 3x1, and relu of the dot products 
    # of each row with the vector are our 4 outputs.
    layer_1 = relu(weights_0_1.T@layer_0)
    #print(layer_1.shape)

    # second layer is just product of the transpose of the weight matrix
    # and the input, no relu; so e.g. our 4x1 weight matrix and our 4x1 input
    # vector will return their dot product, a scalar.
    layer_2 = weights_1_2.T@layer_1
    #print(layer_2.shape)

    # output both
    return layer_1, layer_2


def main():

    # at first I zipped tells and strike together for this,
    # but there's literally no point, we don't use strike
    # in this section
    
    # for each input vector...
    for i in range(len(tells)):
        # run a forward pass...
        layer_1, layer_2 = forward(tells[i], weights_0_1, weights_1_2)

        # and print it out.
        print(f"For sensing {i}, layer 1's result is {layer_1}, and layer 2's result is {layer_2}\n")
    

if __name__ == "__main__":
    main()


# comment block regarding shapes and meanings is requested here;
# I already did that and more above in the comments for forward(). 
# Shapes are (4,) and (1,).