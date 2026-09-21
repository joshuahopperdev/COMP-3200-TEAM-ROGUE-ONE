import numpy as np

# The Trial Dataset
# 4 sensings, 3 binary tells each
tells = np.array([[1, 0, 1],  # foot shift, no guard drop, exhale
                  [0, 1, 1],  # no shift, guard drop, exhale
                  [0, 0, 1],  # only exhale
                  [1, 1, 1]]) # all three (the bluff)

# Ground truth: 1 = strike imminent, 0 = they will hold
strike = np.array([[1, 1, 0, 0]]).T # column vector, shape (4, 1)

# NOTE: Each test_ function has a try-catch block for the imported function 
#       to test and throws an ImportError if a part isn't yet implemented.

# --- Part 1: ReLU and its derivative --- #
def test_relu():
    try:
        from part2_forward_hidden import relu
        from part3_one_backprop_step import relu2deriv
    except ImportError:
        raise ImportError("Either or both relu or relu2deriv does not yet exist")

    # standard sensing from assignment
    sensing = np.array([-1, 0, 1, 2])
    # second arbitrary sensing
    sensing2 = np.array([-0.65, 2.1, -1.3, 0.01])

    # NOTE: to determine if two np.arrays were equal, I used np.all to check if broadcasting the equality
    #       operator (==) returned true for each compared element

    # ReLU
    assert np.all(relu(sensing) == np.array([0, 0, 1, 2])), f"ReLU: expected [0, 0, 1, 2], got {relu(sensing).tolist()}"
    assert np.all(relu(sensing2) == np.array([0, 2.1, 0, 0.01])), f"ReLU: expected [0, 2.1, 0, 0.01], got {relu(sensing2).tolist()}"

    # ReLU derivative: the function returned a list of booleans, which map cleanly to 0 and 1
    assert np.all(relu2deriv(sensing) == np.array([0, 0, 1, 1])), f"ReLU derivative: expected [0, 0, 1, 1], got {relu2deriv(sensing).tolist()}"
    assert np.all(relu2deriv(sensing2) == np.array([0, 1, 0, 1])), f"ReLU derivative: expected [0, 1, 0, 1], got {relu2deriv(sensing2).tolist()}"

# ---- Part 2: Single-layer failure ----- #
def test_single_layer_failure():
    try:
        from part1_single_layer_fails import single_layer_train
    except ImportError:
        raise ImportError("Single layer train function does not yet exist")

    # get the third return value -- error_history -- from the function
    _, _, error_history = single_layer_train(tells, strike, alpha=0.1, epochs=60, seed=1)

    # no error in the history should go below the sanity threshold (0.5), 
    # but the assignment seems to only care about the final total error
    assert np.all([error > 0.5 for error in error_history]), f"Total errors must be above the sanity threshold of 0.5, but instead it's {error_history}"

# ------- Part 3: Forward shapes -------- #
def test_forward_shapes():
    try:
        from part2_forward_hidden import forward
    except ImportError:
        raise ImportError("Forward function does not yet exist")

    # random seed doesn't affect shape,
    # therefore you can pick any seed
    np.random.seed(42)

    hidden_sizes = [4, 6, 3, 8]
    for i in range(4):
        forward_shape_helper(i+1, hidden_sizes[i], forward)
    
def forward_shape_helper(idx, hidden_size, forward):
    # Set up weights
    weights_0_1 = 2 * np.random.random((3, hidden_size)) - 1
    weights_1_2 = 2 * np.random.random((hidden_size, 1)) - 1

    # Get layers
    layer_1, layer_2 = forward(tells[idx-1:idx], weights_0_1, weights_1_2)

    # Actual tests
    assert layer_1.shape == (1, hidden_size), f"layer_1 (Test {idx}): expected shape (1, {hidden_size}), got {layer_1.shape}"
    assert layer_2.shape == (1, 1), f"layer_2 (Test {idx}): expected shape (1, 1), got {layer_2.shape}"    

# ------ Part 4: One backprop step ------ #
def test_backprop_step():
    try:
        # I need forward() to get layer_2,
        # which is the prediction with which
        # to calculate the pre-backprop error
        from part2_forward_hidden import forward
        from part3_one_backprop_step import one_step
    except ImportError:
        raise ImportError("One step function does not yet exist")

    ## Test 1: same seed/weight matrix as src ##
    backprop_step_helper(1, 1, 4, tells[0:1], strike[0:1], one_step, forward)

    ## Test 2: different seed/weight matrix ##
    backprop_step_helper(2, 42, 6, tells[0:1], strike[0:1], one_step, forward)

    ## Test 3: original seed/weight matrix, but different tell/strike ##
    backprop_step_helper(3, 1, 4, tells[1:2], strike[1:2], one_step, forward)

def backprop_step_helper(idx, seed, hidden_size, tell, goal, one_step, forward):
    np.random.seed(seed)
    weights_0_1 = 2 * np.random.random((3, hidden_size)) - 1
    weights_1_2 = 2 * np.random.random((hidden_size, 1)) - 1

    # One step gets us the error before updating 
    # weights, as well as the updated weights
    uweights_0_1, uweights_1_2, _, prev_err = one_step(tell, 
                                                        goal,
                                                        weights_0_1,
                                                        weights_1_2,
                                                        alpha=0.2)

    # layer_2 is synonymous with pred; we don't need layer_1
    _, after_pred = forward(tell, uweights_0_1, uweights_1_2)
    after_err = (after_pred - goal) ** 2

    # The actual test: is the error after backprop 
    # strictly less than the error before?
    assert after_err < prev_err, f"(Test {idx}) Error after backpropagation must be strictly less than the error before"

# -- Part 5: Full training convergence -- #
def test_full_train_converge():
    try:
        from part4_full_training_loop import train # the main subject of the test
        from part2_forward_hidden import relu # activation function for getting the pred
    except ImportError:
        raise ImportError("Train or ReLU function does not yet exist")

    w_0_1, w_1_2, error_history = train(tells, strike, alpha=0.2, epochs=60, hidden_size=4, seed=1)

    # Super concise way to get the pred given
    # the whole dataset and both weight matrices
    pred = relu(tells @ w_0_1) @ w_1_2

    # Assertions
    assert error_history[-1] < 0.01, f"Final total error must be below a small threshold (0.01), got {error_history[-1]}"
    assert np.all([abs(p - s) < 0.5 for p, s in zip(pred, strike)]), f"Each prediction must be on the correct side of 0.5, got {pred}"

# --------- Part 6: Determinism --------- #
def test_determinism():
    try:
        from part4_full_training_loop import train
    except ImportError:
        raise ImportError("Train function does not yet exist")

    seeds = [1, 42, 49, 1776, 7105, 10]
    for seed in seeds:
        determinism_helper(seed, train)

def determinism_helper(seed, train):
    tolerance = 1e-10
    
    fst_w_0_1, fst_w_1_2, _ = train(tells, strike, alpha=0.2, epochs=60, hidden_size=4, seed=seed)
    snd_w_0_1, snd_w_1_2, _ = train(tells, strike, alpha=0.2, epochs=60, hidden_size=4, seed=seed)

    assert np.allclose(fst_w_0_1, snd_w_0_1, rtol=tolerance, atol=tolerance), f"Input-to-hidden weights should be within 1e-10 with seed {seed}, fst={fst_w_0_1}, snd={snd_w_0_1}"
    assert np.allclose(fst_w_1_2, snd_w_1_2, rtol=tolerance, atol=tolerance), f"Hidden-to-output weights should be within 1e-10 with seed {seed}, fst={fst_w_1_2}, snd={snd_w_1_2}"


if __name__ == "__main__":
    tests = [name for name in dir() if name.startswith("test_")]
    for test_name in sorted(tests):
        test_func = globals()[test_name]
        try:
            test_func()
            print(f" PASS: {test_name}")
        except AssertionError as e:
            print(f" FAIL (AssertionError): {test_name} -- {e}")
        except ImportError as e:
            print(f" FAIL (ImportError): {test_name} -- {e}")