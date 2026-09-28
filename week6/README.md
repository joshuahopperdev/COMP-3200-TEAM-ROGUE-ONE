# Week 6: Outbound

This week was the accumulation of all the previous concepts. We have forward pass, error, back prop, weight learning.
We now have this in a loop, combining hidden layers. We have also now incorporated an understanding of shapes. This
will be the jumping off point for the rest of the semester as we learn the different rules and designs for architecture.

## Part 1: Refactored Loop

# Josiah

In last weeks module we did something similar but now we can organize the training loop rather nicely. This version
easily handles multiple hidden layers dynamically.

```py
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
```

## Part 2: Architecture

# Oliver

This section heavily focused on the diagram and visual aspects. Showing shape sizes and what the corresponding weights look like.
This was the core of this module as we were given a powerful visual representation for documentation and design puproses.

# Wider
layer_0 --[ weights_0_1 (3, 32), relu ]--> layer_1 --[ weights_1_2 (32, 1) ]--> layer_2
(1, 3) (1, 32) (1, 1)
Weights: 3x32 + 32x1 = 128

# Deeper
layer_0 --[ weights_0_1 (3, 4), relu ]--> layer_1 --[ weights_1_2 (4, 4), relu ] --> layer_2 --[ weights_2_3 (4, 4), relu ] -->layer_3 --[ weights_3_4 (4, 1) ] --> layer_4
(1, 3) (1, 4) (1, 4) (1, 4) (1, 1)
Weights: 3x4 + 4x4 + 4x4 + 4x4 + 4x1 = 64

# Multi-output
layer_0 --[ weights_0_1 (3, 8), relu ]--> layer_1 --[ weights_1_2 (8, 4) ]--> layer_2
(1, 3) (1, 8) (1, 4)
Weights: 3x8 + 8x4 = 56

# Custom
### 16 measurements from a brain activity sensor are used to classify whether a person is resting, focused, or moving
layer_0 --[ weights_0_1 (16, 32), relu ]--> layer_1 --[ weights_1_2 (32, 16) ]--> layer_2 --[ weights_2_3 (16, 3)]--> layer_3
(1, 16) (1, 32) (1, 16) (1, 3)
Weights: 16x32 + 32x16 + 16x3 = 1072

## Part 3: Build from Diagram

# Nathanael

This part was working from the diagram before and designing a neural network layout to match the specified weights
and shape size.

```py
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
def train_from_diagram(tells, strike, alpha, epochs, seed, layer_lens = [3, 8, 4, 1], verbose = False):
    # initialize weight matrices, sizes calculated from layer_lens.
    # so for e.g. layer_lens of (3, 4, 1), this would initialize 2
    # matrices, a 3x4 and a 4x1
    np.random.seed(seed)
    weight_mats = [0] * (len(layer_lens) - 1)
    for i in range(len(layer_lens) - 1):
        weight_mats[i] = 2*np.random.random((layer_lens[i], layer_lens[i+1])) - 1

    # make an empty error history
    # for each epoch.
    err_hist = [0] * epochs

    # for each epoch...
    for epoch in range(epochs):
        # for each tell...
        for i in range(len(tells)):
            # run a forward pass, saving all the layers
            # including layer 0, the input
            layers = forward(tells[i:i+1], weight_mats)
            # make an empty deltas list
            deltas = [0] * len(weight_mats)
            # set the last delta, since its process is a bit special
            deltas[-1] = layers[-1] - strike[i:i+1]

            # add in our error before we go on, since we only need 
            # this delta to calculate it
            err_hist[epoch] += deltas[-1]**2

            # set all the other deltas (1 less iteration than there
            # are deltas, since we already set the last one)
            for j in range(len(deltas) - 1):
                deltas[-j-2] = deltas[-j-1] @ weight_mats[-j-1].T * relu2deriv(layers[-j-2])

            # do the actual weight updating
            for j in range(len(weight_mats)):
                weight_mats[j] = weight_mats[j] - alpha * np.outer(layers[j], deltas[j]) 
        # print error every 30 epochs
        if epoch % 30 == 29:
            if verbose:
                print(err_hist[epoch]) 
    # run one more forward pass for each and print it
    return err_hist, [forward(input, weight_mats)[-1] for input in tells]

    print(train_from_diagram(tells, strike, 0.1, 150, 4, layer_lens = [3, 8, 4, 1])[1], verbose = True)
```

## Unit Testing

# Caleb
Unit testing results went well.
 PASS: test_deeper_net_convergence
 PASS: test_deeper_net_runs
 PASS: test_determinism
 PASS: test_one_refactor_correctness
 PASS: test_shape_sanity

Tests comparing similar functionality from the week prior matches the concepts we applied.
Testing was somewhat light as this week was heavily on diagram architecture and how to implement it.
