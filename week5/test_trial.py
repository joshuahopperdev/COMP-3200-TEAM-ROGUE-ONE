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
        raise ImportError

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