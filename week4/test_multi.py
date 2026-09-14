import numpy as np

# Each test_ function has a try-catch block for the import
# and throws an ImportError if a part isn't yet imlemented


# -- Part 1: Normalization --
def test_normalization():
    try:
        from part1_multi_input import normalize
    except ImportError:
        raise ImportError("Part 1b - normalize() does not exist yet")

    # Normalize an array of inputs
    assert normalize([4, 2, 1]) == [
        1,
        0.5,
        0.25,
    ], "Biggest value (4) should become 1, and the others (2 and 1, half and one-fourth) should be 0.5 and 0.25."


# -- Part 1: Multi-input GD --
def test_multi_in_gd():
    try:
        from part1_multi_input import gradient_descent_multi
    except ImportError:
        raise ImportError("Part 1 - gradient descent multi does not yet exist")

    # Setup
    sensing = [1, 1.5, -0.75]
    weights = [1, 0.5, 0.25]
    final_weights, errors, _ = gradient_descent_multi(sensing, weights, 1, 0.1, 10)

    # Error goes down
    for i in range(len(errors) - 2):
        assert errors[i] > errors[i + 1], "Errors should decrease over each iteration"

    # Final prediction is close to goal (which is 1)
    # get prediciton
    from helpers import w_sum

    final_pred = w_sum(sensing, final_weights)
    assert (
        abs(final_pred - 1) < .01
    ), "Final prediction should be close to the goal for an easy sensing"


# -- Part 2: Multi-output GD --
def test_multi_out_gd():
    try:
        from part2_multi_output import gradient_descent_outputs
    except ImportError:
        raise ImportError("Part 2 - gradient descent outputs does not yet exist")

    # Use data from recurring dataset
    balance = 0.65
    weights = [0.3, 0.2, 0.9]
    trues = [0.0, 1.0, 0.0]
    _, _, weight_history = gradient_descent_outputs(balance, weights, trues, 0.01, 20)

    # Iterate through weight_history, comparing each value with the next
    for i in range(len(weight_history) - 1):
        weighti = weight_history[i]
        weightip = weight_history[i + 1]
        pred = [balance * weight for weight in weighti]
        next_pred = [balance * weight for weight in weightip]

        # Use abs(pred - true) for sheer magnitude, no direction or
        # square exaggerations despite their technical correctness
        deltas = [abs(p - true) for p, true in zip(pred, trues)]
        next_deltas = [abs(p - true) for p, true in zip(next_pred, trues)]

        for j in range(3):
            assert (
                deltas[j] >= next_deltas[j]
            ), "Predictions should be moving toward the target"
            assert (
                weighti[j] == weightip[j]
                if deltas[j] == next_deltas[j]
                and pred[j] == next_pred[j]
                and pred[j] == trues[j]
                else weighti[j] != weightip[j]
            ), "Weight should change each iteration unless the prediction matches the target"


# -- Part 3: Outer Product --
def test_outer_product():
    try:
        from part3_multi_in_out import outer_product
    except ImportError:
        raise ImportError("Part 3 - outer product does not yet exist")

    assert outer_product([1, 2], [3, 4, 5]) == [
        [3, 4, 5],
        [6, 8, 10],
    ], "A len-2 list outer_prod'ed with a len-3 list produces a 2x3 matrix; [1,2]x[3,4,5] == [[3,4,5],[6,8,10]]"
    assert outer_product([6, 7, 8], [9, 10]) == [
        [54, 60],
        [63, 70],
        [72, 80],
    ], "A len-3 list outer_prod'ed with a len-2 list produces a 3x2 matrix; [6,7,8]x[9,10] == [[54,60],[63,70],[72,80]]"


# -- Part 4: Freezing --
def test_freezing():
    try:
        from part4_freeze import gradient_descent_frozen
    except ImportError:
        raise ImportError(
            "Part 4 - gradient descent frozen doesn't yet exist"
        )

    # Test 1: 1 index frozen
    help_test_freezing([1], gradient_descent_frozen)

    # Test 2: 2 indices frozen
    help_test_freezing([0, 2], gradient_descent_frozen)

def help_test_freezing(frozen, gdf):
    from helpers import w_sum
    frozen_copy = frozen.copy()
    sensing = [8.5, 0.65, 1.2]
    weights = [0.1, 0.2, -0.1]
    true = 1
    _, _, weight_history = gdf(
        sensing, weights, true, 0.3, 5, frozen
    )

    for i in range(len(weight_history) - 1):
        # Setup for checking that free weights change unless they don't need to.
        weighti = weight_history[i]
        weightip = weight_history[i + 1]
        pred = w_sum(sensing, weighti)
        next_pred = w_sum(sensing, weightip)

        # Use abs(pred - true) for sheer magnitude, no direction or
        # square exaggerations despite their technical correctness
        delta = pred - true
        next_delta = next_pred - true

        # The frozen weights must not change
        for j in range(len(weighti) - 1):
            if j in frozen_copy:
                assert (
                    weighti[j] == weightip[j]
                ), "Frozen weights must not change"
            else:
                assert (
                    weighti[j] == weightip[j]
                    if delta == next_delta
                    and pred == next_pred
                    and pred == true
                    else weighti[j] != weightip[j]
                ), "Free weights must change"


# -- Scratch vs NumPy agreement --
# Many In, One Out
def test_multi_in_agreement():
    try:
        from part1_multi_input import (
            gradient_descent_multi,
            gradient_descent_multi_numpy,
        )
    except ImportError:
        raise ImportError("Part 1 - gradient descent multi doesn't yet exist")

    from helpers import w_sum

    sensing = [8.5, 0.65, 1.2]
    weights = [0.1, 0.2, -0.1]
    true = 1.0
    alpha = 0.01
    iterations = 10

    final_weights_scratch, _, _ = gradient_descent_multi(
        sensing, weights, true, alpha, iterations
    )
    final_weights_numpy, _, _ = gradient_descent_multi_numpy(
        sensing, weights, np.float64(true), np.float64(alpha), iterations
    )

    pred_scratch = w_sum(sensing, final_weights_scratch)
    pred_numpy = np.asarray(sensing) @ final_weights_numpy.T

    assert (
        abs(pred_scratch - pred_numpy) < 1e-10
    ), "Multi-in Gradient Descent from scratch must be comparable to NumPy equivalent"


# One In, Many Out
def test_multi_out_agreement():
    try:
        from part2_multi_output import (
            gradient_descent_outputs,
            gradient_descent_outputs_numpy,
        )
    except ImportError:
        raise ImportError("Part 2 - gradient descent multiple output doesn't yet exist")

    balance = 0.65
    weights = [0.3, 0.2, 0.9]
    trues = [0.0, 1.0, 0.0]
    alpha = 0.1
    iterations = 20

    final_weights_scratch, _, _ = gradient_descent_outputs(
        balance, weights, trues, alpha, iterations
    )
    final_weights_numpy, _, _ = gradient_descent_outputs_numpy(
        balance, weights, np.float64(trues), np.float64(alpha), iterations
    )

    preds_scratch = [balance * weight for weight in final_weights_scratch]
    preds_numpy = np.float64(balance) * final_weights_numpy

    assert [
        abs(pred_scratch - pred_numpy) < 1e-10
        for pred_scratch, pred_numpy in zip(preds_scratch, preds_numpy)
    ], "Multi-out Gradient Descent from scratch must be comparable to NumPy equivalent"


# Many In, Many Out
def test_multi_in_out_agreement():
    try:
        from part3_multi_in_out import gradient_descent_full, grad_desc_numpy_full
    except ImportError:
        raise ImportError(
            "Part 3 - gradient descent multiple input and output doesn't exist yet"
        )

    from helpers import vect_mat_mul

    sensing = [8.5, 0.65, 1.2]
    weights = [[0.1, 0.1, -0.3], [0.1, 0.2, 0.0], [0.0, 1.3, 0.1]]
    trues = [0.0, 1.0, 0.0]
    alpha = 0.01
    iterations = 15

    final_weights_scratch, _, _ = gradient_descent_full(
        sensing, weights, trues, alpha, iterations
    )
    final_weights_numpy, _, _ = grad_desc_numpy_full(
        sensing, weights, trues, np.float64(alpha), iterations
    )

    preds_scratch = vect_mat_mul(sensing, final_weights_scratch)
    preds_numpy = np.asarray(sensing) @ final_weights_numpy.T

    assert [
        abs(pred_scratch - pred_numpy) < 1e-10
        for pred_scratch, pred_numpy in zip(preds_scratch, preds_numpy)
    ], "Multi-in-multi-out Gradient Descent from scratch must be comparable to NumPy equivalent"


if __name__ == "__main__":
    tests = [name for name in dir() if name.startswith("test_")]
    for test_name in sorted(tests):
        test_func = globals()[test_name]
        try:
            test_func()
            print(f" PASS: {test_name}")
        except AssertionError as e:
            print(f" FAIL: {test_name} -- {e}")
        except ImportError as e:
            print(f" FAIL: {test_name} -- {e}")
