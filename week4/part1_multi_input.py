from helpers import w_sum, vect_mat_mul
import numpy as np

def ele_mul(scalar, vector, debug = False):
    """
    Element wise multiplication.
    """
    output = [0] * len(vector)
    if debug:
        print(f"Length of vector: {len(vector)}")
    for i in range(len(vector)):
        output[i] = scalar *  vector[i]

    return output

def gradient_descent_multi(input, weights, true, alpha, iterations, debug = False):
    """
    Gradient descent with multiple weights.
    """
    for iter in range(0, iterations):
        # Predict
        pred = w_sum(input, weights)

        # Compare and get error
        error = (pred - true) ** 2
        delta = pred - true

        # Learn
        weight_deltas = ele_mul(delta, input)

        for i in range(len(weight_deltas)):
            weights[i] -= alpha * weight_deltas[i]

        if debug:
            print(f"Iter: {iter+1}\tPred: {pred}\tError: {error:.6f}\tWeights: {weights}")

    return weights

def gradient_descent_multi_numpy(input, weights, true, alpha, iterations, debug = False):
    """
    Gradient descent with multiple weights, numpy version.
    """
    np_input = np.array(input)
    np_weights = np.array(weights)

    for iter in range(0,iterations):

        # Predict
        pred = np_input.dot(np_weights.T)

        # Compare and get error
        error = (pred - true) ** 2
        deltas = pred - true

        # Learn
        # print((alpha * np.outer(deltas,input)).shape) (1,3)
        # print(np_weights.shape) (3,)
        np_weights -= alpha * deltas * np_input

        if debug:
            print(f"Iter: {iter+1}\tPred: {pred}\tError: {error:.6f}\tWeights: {np_weights}")

    return np_weights


def main():
    """
    Main function for testing functions.
    """

    # Example from class
    delta = -0.14
    input = [8.5, 0.65, 1.2]
    weight_deltas = ele_mul(delta, input)
    print(weight_deltas)

    # Example from the slides
    weights = [0.1, 0.2, -0.1]
    true = 1
    alpha = 0.01
    weights = gradient_descent_multi(input, weights, true, alpha, 4, debug = True)
    # It really doesn't improve much after 4 iterations. 10 is overkill. 
    # If our alpha was smaller maybe it would be useful.
    
    np_weights = np.array([0.1, 0.2, -0.1])
    true = 1
    alpha = 0.01
    np_weights = gradient_descent_multi_numpy(input, np_weights, true, alpha, 4, debug = True)

    if np.allclose(weights, np_weights, 1e-9):
        print("From Scratch and Numpy Example are equal.")
    else:
        print("From Scratch and Numpy Example are not equal.")

    # As for which weights changed the most. Probably the first one? None of the weights seemed
    # to have changed very drastically. The first one just happened to have a change of 0.01 where
    # the others changed 0.001 or 0.003. Overall the values changed very little.



if __name__ == "__main__":
    main()