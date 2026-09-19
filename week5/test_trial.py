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

# -- Part 1: ReLU and its derivative --
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

# -- Part 2: Single-layer failure --
def test_single_layer_failure():
    try:
        from part1_single_layer_fails import single_layer_train
    except ImportError:
        raise ImportError("Single layer train function does not yet exist")

    # TODO: write tests when the code becomes testable

# -- Part 3: Forward shapes --
def test_forward_shapes():
    try:
        from part2_forward_hidden import forward
    except ImportError:
        raise ImportError("Forward function does not yet exist")

    # random seed doesn't affect shape,
    # therefore you can pick any seed
    np.random.seed(42)

    ## Test 1 ##

    # Set up weights
    hidden_size = 4
    weights_0_1 = 2 * np.random.random((3, hidden_size)) - 1 # 3 x 4
    weights_1_2 = 2 * np.random.random((hidden_size, 1)) - 1 # 4 x 1

    # Get layers
    layer_1, layer_2 = forward(tells[0:1], weights_0_1, weights_1_2)

    # Actual tests
    assert layer_1.shape == (1, hidden_size), f"layer_1 (Test 1): expected shape (1, {hidden_size}), got {layer_1.shape}"
    assert layer_2.shape == (1, 1), f"layer_2 (Test 1): expected shape (1, 1), got {layer_2.shape}"

    ## Test 2 ##

    # Set up weights
    hidden_size = 6
    weights_0_1 = 2 * np.random.random((3, hidden_size)) - 1 # 3 x 6
    weights_1_2 = 2 * np.random.random((hidden_size, 1)) - 1 # 6 x 1

    # Get layers
    layer_1, layer_2 = forward(tells[1:2], weights_0_1, weights_1_2)

    # Actual tests
    assert layer_1.shape == (1, hidden_size), f"layer_1 (Test 2): expected shape (1, {hidden_size}), got {layer_1.shape}"
    assert layer_2.shape == (1, 1), f"layer_2 (Test 2): expected shape (1, 1), got {layer_2.shape}"

# -- Part 4: One backprop step --
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
    np.random.seed(1)
    hidden_size = 4
    weights_0_1 = 2 * np.random.random((3, hidden_size)) - 1 # 3 x 4
    weights_1_2 = 2 * np.random.random((hidden_size, 1)) - 1 # 4 x 1

    # One step gets us the error before updating 
    # weights, as well as the updated weights
    uweights_0_1, uweights_1_2, _, prev_err = one_step(tells[0:1], 
                                                       strike[0],
                                                       weights_0_1,
                                                       weights_1_2,
                                                       alpha=0.2)

    # layer_2 is synonymous with pred; we don't need layer_1
    _, after_pred = forward(tells[0:1], uweights_0_1, uweights_1_2)
    after_err = (after_pred - strike[0]) ** 2

    # The actual test: is the error after backprop 
    # strictly less than the error before?
    assert after_err < prev_err, "(Test 1) Error after backpropagation must be strictly less than the error before"

    ## Test 2: different seed/weight matrix ##
    np.random.seed(42)
    hidden_size = 6
    weights_0_1 = 2 * np.random.random((3, hidden_size)) - 1 # 3 x 4
    weights_1_2 = 2 * np.random.random((hidden_size, 1)) - 1 # 4 x 1

    uweights_0_1, uweights_1_2, _, prev_err = one_step(tells[0:1],
                                                       strike[0],
                                                       weights_0_1,
                                                       weights_1_2,
                                                       alpha=0.2)

    _, after_pred = forward(tells[0:1], uweights_0_1, uweights_1_2)
    after_err = (after_pred - strike[0]) ** 2

    assert after_err < prev_err, "(Test 2) Error after backpropagation must be strictly less than the error before"

    ## Test 3: original seed/weight matrix, but different tell/strike ##
    np.random.seed(1)
    hidden_size = 4
    weights_0_1 = 2 * np.random.random((3, hidden_size)) - 1 # 3 x 4
    weights_1_2 = 2 * np.random.random((hidden_size, 1)) - 1 # 4 x 1

    uweights_0_1, uweights_1_2, _, prev_err = one_step(tells[0:1], 
                                                        strike[0],
                                                        weights_0_1,
                                                        weights_1_2,
                                                        alpha=0.2)

    _, after_pred = forward(tells[0:1], uweights_0_1, uweights_1_2)
    after_err = (after_pred - strike[0]) ** 2

    assert after_err < prev_err, "(Test 3) Error after backpropagation must be strictly less than the error before"

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