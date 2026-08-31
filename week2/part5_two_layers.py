import numpy as np
from part4_multi_in_multi_out import vect_mat_mul
from part4_multi_in_multi_out import np_nn


# angle balance breath
ih_wgt = [[0.1, 0.2, -0.1], # -> hid[0]
[-0.1, 0.1, 0.9], # -> hid[1]
[0.1, 0.4, 0.1]] # -> hid[2]

# hid0 hid1 hid2
hp_wgt = [[0.3, 1.1, -0.3], # -> opens_left?
[0.1, 0.2, 0.0], # -> strikes_high?
[0.0, 1.3, 0.1]] # -> feints?


### a bunch of little attempts to replicate stuff myself; 
## this way I understand what it's doing when I use others' code
## elementwise multiply each value in 2 equal-length vectors, then sum
#def dot_prod(vec1, vec2):
#    # check they're the same length
#    assert len(vec1) == len(vec2)
#    # start at sum = 0
#    output = 0
#    # sequentially add the element-wise products of the vectors' values
#    for i in range(len(vec1)):
#        output += vec1[i] * vec2[i]
#
#    return output
#
#def matrix_vect_mult(matrix, vector):
#    # verify that we're multiplying an m x n matrix by an n x 1 vector
#    # by checking that the second dimension of the first row of the matrix
#    # is equal to the height of the vector (which we presume to be a vector)
#    # assumes the matrix is rectangular, that is, same width at all points
#    assert len(matrix[0]) == len(vector)
#
#    # generate output vector of 0s, as long as the height of the matrix
#    # final result of multiplying m x n matrix by n x 1 vector is m x 1 vector
#    output = [0]*len(matrix)
#
#    # see above re: length
#    for i in range(len(matrix)):
#        # the i'th value of the vector should be equal to
#        # the dot product of the entire vector with the i'th row
#        # of the matrix.
#        output[i] = dot_prod(matrix[i], vector)
#
#    return output



# works on an arbitrary number of equal-sized layers with the same size as the input
# e.g. can take 8 layers of weights for 3 inputs each and a vector of length 3
def neural_network(weights_sequence, input):
    # store starting values
    cur = input
    stored_layers = [0]*len(weights_sequence)
    #sequentially multiply starting values by the matrix of the weights
    #in each layer, proceeding to the next one

    
    for i in range(len(weights_sequence)):
        cur = vect_mat_mul(cur, weights_sequence[i])
        stored_layers[i] = cur

    return stored_layers


def main():
    weights = [ih_wgt, hp_wgt]
    blade_angle = [8.5, 9.5, 9.9, 9.0] # degrees off-vertical
    balance = [0.65, 0.80, 0.80, 0.90] # balance reading
    breath = [1.2, 1.3, 0.5, 1.0] # exhalations per second

    # trust that there are the same number of entries
    for i in range(len(blade_angle)):
        input = [blade_angle[i], balance[i], breath[i]]
        pred=neural_network(weights, input)
        for j in range(len(pred)):
            print(f"Sensing {i}, Layer {j+1}: {pred[j]}")

    # Expected for sensing 0 (rounded): hidden = [0.86, 0.295, 1.23]
    # pred = [0.2135, 0.145, 0.5065]

    print("Each hidden value is an intermediate stage, something that may or may not"
    " correspond to something we understand, storing some amount of information in some way."
    " But to not all collapse into one layer of summed linear transformations, we need"
    " something nonlinear (and non-polynomial!), applied to all these to mess with"
    " them such that you can get odd behavior.")

    #wait, why are we doing this with np.dot? This seems a little more manual than necessary.
    print("--Numpy Version--")
    np_weights = np.asarray(weights)
    np_sensings = np.asarray(list(zip(blade_angle, balance, breath)))


    print(f"{np_weights.shape} - a 2 x 3 x 3 vector, it contains 2 layers of 3 vectors of the 3 weights of that neuron.")
    print(f"{np_sensings.shape} - a 4 x 3 vector, it contains 4 sensings, each containing a vector made of the 3 variables blad_angle, balance and breath")
    for i in range(len(np_sensings)):
        input = [np_sensings[i]]

        np_hidden_layer = np_nn(input, np_weights[0])
        np_final_layer = np_nn(np_hidden_layer, np_weights[1])






if __name__ == "__main__":
    main()
