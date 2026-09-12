from helpers import w_sum, vect_mat_mul

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
    for iter in range(1, iterations):
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
            print(f"Iter: {i}\tPred: {pred}\tError: {error:.6f}")

    return weights


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
    

if __name__ == "__main__":
    main()