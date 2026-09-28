import numpy as np

# 4 sensings, 3 binary tells each, shape (4, 3)
tells = np.array([[1, 0, 1], # foot shift, no guard drop, exhale
                  [0, 1, 1], # no shift, guard drop, exhale
                  [0, 0, 1], # only exhale
                  [1, 1, 1]]) # all three (the bluff)

# Ground truth: 1 = strike imminent, 0 = they will hold
strike = np.array([[1, 1, 0, 0]]).T # column vector, shape (4, 1)

# ReLU and its derivative
relu = lambda x: np.where(x > 0, x, 0)
relu2deriv = lambda x: np.where(x > 0, 1, 0)

def train(tells, strike, alpha, epochs, hidden_size, seed, verbose = False):
    # Set random seed so results can be reproduced
    np.random.seed(seed)

    # Initialize weights randomly between -1 and 1
    weights_0_1 = 2 * np.random.random((tells.shape[1], hidden_size)) - 1 # layer 0 to layer 1, shape (tells.shape[1], hidden_size)
    weights_1_2 = 2 * np.random.random((hidden_size, 1)) - 1 # layer 1 to layer 2, shape (hidden_size, 1)

    # Track total error for each epoch
    error_history = [0.0] * epochs

    # Loop through the dataset for the given number of epochs
    for epoch in range(epochs):
        total_error = 0

        # Process each sensing sample one at a time (Stochastic GD)
        for i in range(len(tells)):
            _, _, _, _, _, weights_0_1, weights_1_2, total_error = onestep(tells, strike, i, weights_0_1, weights_1_2, alpha, total_error)

        # Save this epoch's total error
        error_history[epoch] = total_error

        # Print total squared error every 10 epochs
        # +1 so it doesn't print right away
        if (epoch + 1) % 10 == 0:
            if verbose:
                print(f"Epoch {epoch + 1:2d} | Total Error: {total_error:.6f}")

    # Return final weights and error log
    return weights_0_1, weights_1_2, error_history

def onestep(tells, strike, idx, weights_0_1, weights_1_2, alpha, total_error):
    """
    One step of the Predict-Compare-Learn loop
    Parameters:
    `tells` (in): the dataset
    `strike` (in): labels for the dataset
    `idx` (in): index with which to get the current tell/strike
    `weights_0_1` (in): weight matrix from layer_0 to layer_1
    `weights_1_2` (in): weight matrix from layer_1 to layer_2
    `alpha` (in): alpha value used to reign in weight updates
    `total_error` (in): total error for the overall epoch
    """
    layer_0 = tells[idx:idx+1]
    target = strike[idx:idx+1]

    # --- FORWARD ---
    layer_1 = relu(layer_0 @ weights_0_1) # (1, hidden_size)
    layer_2 = layer_1 @ weights_1_2 # (1, 1)

    # --- COMPARE ---
    total_error += np.sum((layer_2 - target) ** 2) # (1, 1)

    # --- BACKWARD ---
    layer_2_delta = layer_2 - target # (1, 1)
    layer_1_delta = layer_2_delta @ weights_1_2.T * relu2deriv(layer_1) # (1, hidden_size)

    # --- LEARN ---
    weights_0_1 -= alpha * layer_0.T @ layer_1_delta # (3, 1) @ (1, hidden_size) -> broadcast over shape (3, hidden_size) -> (3, hidden_size)
    weights_1_2 -= alpha * layer_1.T @ layer_2_delta # (hidden_size, 1) @ (1, 1) -> broadcast over shape (hidden_size, 1) -> (hidden_size, 1)

    return layer_0, layer_1, layer_2, layer_1_delta, layer_2_delta, weights_0_1, weights_1_2, total_error
