import numpy as np

def gradient_descent_outputs(input, weights, trues, alpha, iterations, debug = False):
    """
    This function performs gradient descent with one input but multiple outputs.
    """
    error_result = []
    weight_hist_result = []
    for iter in range(0, iterations):
        # Predict
        preds = [input * w for w in weights]

        # Compare and get error
        deltas = [preds[i] - trues[i] for i in range(len(weights))]
        errors = [d ** 2 for d in deltas]
        error_result.append(errors)

        # Learn
        weighted_deltas = [d * input for d in deltas]
        for i in range(len(weights)):
            weights[i] -= alpha * weighted_deltas[i]

        weight_hist_result.append(weights)

        if debug:
            print(f"Iter: {iter+1}\tPred: {preds}\n\tError: {errors}\n\tWeights: {weights}")

    return weights, error_result, weight_hist_result

def gradient_descent_outputs_numpy(input, weights, trues, alpha, iterations, debug = False):
    """
    This function performs gradient descent with one input but multiple outputs.
    This time with numpy.
    """
    error_result = []
    weight_hist_result = []
    for iter in range(0, iterations):
        # Predict
        preds = input * weights

        # Compare and get error
        deltas = preds - trues
        errors = [d ** 2 for d in deltas]
        error_result.append(errors)

        # Learn
        weights -= alpha * deltas * input
        weight_hist_result.append(weights)

        if debug:
            print(f"Iter: {iter+1}\tPred: {preds}\n\tError: {errors}\n\tWeights: {weights}")

    return weights, error_result, weight_hist_result


def main():
    """
    Main function for testing functions.
    """
    input = 0.65
    weights = [0.3, 0.2, 0.9]
    trues = [0.0, 1.0, 0.0]
    alpha = 0.1
    iterations = 10

    weights, _, _ = gradient_descent_outputs(input, weights, trues, alpha, iterations, debug = True)


    np_input = np.asarray([0.65])
    np_weights = np.asarray([0.3, 0.2, 0.9])
    np_trues = np.asarray([0.0, 1.0, 0.0])
    alpha = 0.1
    iterations = 10

    np_weights, _, _ = gradient_descent_outputs_numpy(np_input, np_weights, np_trues, alpha, iterations, debug = True)

    if np.allclose(weights, np_weights, 1e-9):
        print("From Scratch and Numpy Example are equal.")
    else:
        print("From Scratch and Numpy Example are not equal.")

    # Iter: 1  Error: [0.038025,             0.7569,              0.3422250000000001]
    # Iter: 10 Error: [0.017482684272196927, 0.34799852006905607, 0.15734415844977237]
    # Difference       0.021~                0.409~               0.185~
    # The error(delta^2) is much lower on the first. Still quite large on the 2nd, and somewhere in between on the 3rd.

if __name__ == "__main__":
    main()
