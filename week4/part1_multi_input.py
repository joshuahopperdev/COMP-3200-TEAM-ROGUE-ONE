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

def normalize(channel):
    """
    This function retuns an array normalized by the largest value in the array.
    This function assumes that the array passed in is numeric in value.
    """
    max_value = max(channel)
    result_array = []
    for i in channel:
        result_array.append(i/max_value)

    return result_array

def gradient_descent_multi(input, weights, true, alpha, iterations, debug = False):
    """
    Gradient descent with multiple weights.
    """
    error_result = []
    weight_hist_result = []
    for iter in range(0, iterations):
        # Predict
        pred = w_sum(input, weights)

        # Compare and get error
        error = (pred - true) ** 2
        error_result.append(error)
        delta = pred - true

        # Learn
        weight_deltas = ele_mul(delta, input)

        for i in range(len(weight_deltas)):
            weights[i] -= alpha * weight_deltas[i]
        weight_hist_result.append(weights.copy())

        if debug:
            print(f"Iter: {iter+1}\tPred: {pred}\tError: {error:.6f}\tWeights: {weights}")

    return weights, error_result, weight_hist_result

def gradient_descent_multi_numpy(input, weights, true, alpha, iterations, debug = False):
    """
    Gradient descent with multiple weights, numpy version.
    """
    np_error_result = []
    np_weight_hist_result = []
    np_input = np.array(input)
    np_weights = np.array(weights)

    for iter in range(0,iterations):

        # Predict
        pred = np_input.dot(np_weights.T)

        # Compare and get error
        error = (pred - true) ** 2
        np_error_result.append(error)
        deltas = pred - true

        # Learn
        # print((alpha * np.outer(deltas,input)).shape) (1,3)
        # print(np_weights.shape) (3,)
        np_weights -= alpha * deltas * np_input
        np_weight_hist_result.append(np_weights.copy())

        if debug:
            print(f"Iter: {iter+1}\tPred: {pred}\tError: {error:.6f}\tWeights: {np_weights}")

    return np_weights, np_error_result, np_weight_hist_result


def main():
    """
    Main function for testing functions.
    """
    print("\n------ Week 4 - Part 1 ------\n")
    # Example from class
    delta = -0.14
    input = [8.5, 0.65, 1.2]
    weight_deltas = ele_mul(delta, input)
    print(weight_deltas)

    # Example from the slides
    weights = [0.1, 0.2, -0.1]
    true = 1
    alpha = 0.01
    weights, _, _ = gradient_descent_multi(input, weights, true, alpha, 4, debug = True)
    # It really doesn't improve much after 4 iterations. 10 is overkill. 
    # If our alpha was smaller maybe it would be useful.
    
    np_weights = np.array([0.1, 0.2, -0.1])
    true = 1
    alpha = 0.01
    np_weights, np_errors, np_weight_hist = gradient_descent_multi_numpy(input, np_weights, true, alpha, 4, debug = True)

    # These print statements were for testing the returns from the function initially.
    # print(f"---------------------- NP Errors\n{np_errors}")
    # print(f"---------------------- NP Weight Hist\n{np_weight_hist}")

    if np.allclose(weights, np_weights, 1e-9):
        print("From Scratch and Numpy Example are equal.")
    else:
        print("From Scratch and Numpy Example are not equal.")

    # As for which weights changed the most. Probably the first one? None of the weights seemed
    # to have changed very drastically. The first one just happened to have a change of 0.01 where
    # the others changed 0.001 or 0.003. Overall the values changed very little.


    print("\n------ Week 4 - Part 1b ------\n")
    input = [8.5, 0.65, 1.2]
    weights = [0.1, 0.2, -0.1]
    true = 1
    alpha = 0.01
    print(f"Raw with alpha {alpha}")
    weights, _, _ = gradient_descent_multi(input, weights, true, alpha, 4, debug = True)
    # normalize(input)
    input = [8.5, 0.65, 1.2]
    weights = [0.1, 0.2, -0.1]
    true = 1
    alpha = 0.2
    print(f"Scaled with alpha {alpha}")
    new_input = normalize(input)
    weights, _, _ = gradient_descent_multi(new_input, weights, true, alpha, 20, debug = True)

    # Of note here, the 0.1 after 20 iterations doesn't get near the same error and pred as having a smaller
    # learning rate and larger inputs. In the real world I would still go for normalizing the inputs regardless.
    # But in this scenario, having a smaller learning rate works just fine. Largely because of the giant difference
    # in input value size.

    # Part 1b.5.a
    # The input stays the same, the weight changes. So if the input just happens to be above 1, and the weight 
    # started out above 1 as well, this number will have no way of decreasing and will go into infinity.
    
    # Part 1b.5.b
    # I unintentially answered this above before the questions. But I believe it has to do with how big the input
    # difference is that when normalized, the smaller inputs are made into much smaller decimals which don't account
    # the same as having 1 decimal point higher alpha value. When we move it near 0.2 or 0.25 over 0.01, then we
    # wee a much faster error reduction speed after 20 iterations.

    # Part 1b.5.c
    # I'm not sure if I understand the question or not, but if we are freezing some neurons or using drop out,
    # setting the weight to 0 effectively does the same thing as that doesn't have any impact on the result.

if __name__ == "__main__":
    main()