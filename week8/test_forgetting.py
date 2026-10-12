import numpy as np
import inspect

from data import load_mnist, one_hot

(X_train, y_train), (X_test, y_test) = load_mnist()
X_val, y_val = X_train[1000:2000], one_hot(y_train[1000:2000])
X_train, y_train = X_train[:1000], one_hot(y_train[:1000])

def test_part_1_overfitting():
    try:
        from part1_memorize import train
    except ImportError as e:
        raise ImportError(f"Error loading train from part 1. {e}")
    _, _, train_hist, val_hist = train(X_train, y_train, X_val, y_val,
iterations=50)

    assert (train_hist[-1] > 0.95), f"Part 1: final training accuracy should be greater than 0.95; final training accuracy was {train_hist[-1]}"
    assert (train_hist[-1] - val_hist[-1] > 0.1), f"Part 1: final training accuracy should be at least 0.1 greater than final validation accuracy; final difference was {train_hist[-1]-val_hist[-1]}"


def test_parts_1234_no_test_leakage():
    try:
        from part1_memorize import train as train_1
        from part2_early_stopping import train_early_stopping as train_2
        from part3_dropout import train_dropout as train_3
        from part4_batching import train_batched as train_4
    except ImportError as e:
        raise ImportError(f"Error loading a train function. {e}")

    train_1_sig = inspect.signature(train_1)
    train_1_source = inspect.getsource(train_1)
    train_2_sig = inspect.signature(train_2)
    train_2_source = inspect.getsource(train_2)
    train_3_sig = inspect.signature(train_3)
    train_3_source = inspect.getsource(train_3)
    train_4_sig = inspect.signature(train_4)
    train_4_source = inspect.getsource(train_4)

    # follows assignment as written, but will fail if any comment anywhere
    # contains the word "X_test" or "y_test"....
    assert (not ("X_test" in str(train_1_sig))), "Part 1's signature contains the word \"X_test\""
    assert (not ("X_test" in str(train_1_source))), "Part 1's source contains the word \"X_test\""
    assert (not ("X_test" in str(train_2_sig))), "Part 2's signature contains the word \"X_test\""
    assert (not ("X_test" in str(train_2_source))), "Part 2's source contains the word \"X_test\""
    assert (not ("X_test" in str(train_3_sig))), "Part 3's signature contains the word \"X_test\""
    assert (not ("X_test" in str(train_3_source))), "Part 3's source contains the word \"X_test\""
    assert (not ("X_test" in str(train_4_sig))), "Part 4's signature contains the word \"X_test\""
    assert (not ("X_test" in str(train_4_source))), "Part 4's source contains the word \"X_test\""


    assert (not ("y_test" in str(train_1_sig))), "Part 1's signature contains the word \"y_test\""
    assert (not ("y_test" in str(train_1_source))), "Part 1's source contains the word \"y_test\""
    assert (not ("y_test" in str(train_2_sig))), "Part 2's signature contains the word \"y_test\""
    assert (not ("y_test" in str(train_2_source))), "Part 2's source contains the word \"y_test\""
    assert (not ("y_test" in str(train_3_sig))), "Part 3's signature contains the word \"y_test\""
    assert (not ("y_test" in str(train_3_source))), "Part 3's source contains the word \"y_test\""
    assert (not ("y_test" in str(train_4_sig))), "Part 4's signature contains the word \"y_test\""
    assert (not ("y_test" in str(train_4_source))), "Part 4's source contains the word \"y_test\""


def test_part_2_best_early_stopping():
    try:
        from part2_early_stopping import train_early_stopping
    except ImportError as e:
        raise ImportError(f"Error loading train_early_stopping from part 2. {e}")
    try:
        from part1_memorize import evaluate
    except ImportError as e:
        raise ImportError(f"Error loading evaluate (helper function) from part 1. {e}")

    W_0_1, W_1_2, train_hist, val_hist = train_early_stopping(X_train, y_train, X_val, y_val, iterations=60)
    forward_pass = evaluate(X_val, y_val, W_0_1, W_1_2)
    assert np.allclose(forward_pass, np.amax(val_hist), atol=1e-9), f"Part 2's returned weights do not produce the same value as val_hist's best value: {forward_pass} vs {np.amax(val_hist)}"

def test_part_3_dropout_improves():
    try:
        from part3_dropout import train_dropout
    except ImportError as e:
        raise ImportError(f"Error loading train_dropout from part 3. {e}")
    try:
        from part1_memorize import evaluate
    except ImportError as e:
        raise ImportError(f"Error loading evaluate (helper function) from part 1. {e}")

    W_0_1_drop, W_1_2_drop, _, _ = train_dropout(X_train, y_train, X_val, y_val, iterations = 50, dropout=True)
    W_0_1_ctrl, W_1_2_ctrl, _, _ = train_dropout(X_train, y_train, X_val, y_val, iterations = 50, dropout=False)

    drop_train_acc = evaluate(X_train, y_train, W_0_1_drop, W_1_2_drop)
    drop_val_acc = evaluate(X_val, y_val, W_0_1_drop, W_1_2_drop)
    ctrl_train_acc = evaluate(X_train, y_train, W_0_1_ctrl, W_1_2_ctrl)
    ctrl_val_acc = evaluate(X_val, y_val, W_0_1_ctrl, W_1_2_ctrl)

    assert (abs(drop_train_acc - drop_val_acc) < abs(ctrl_train_acc - ctrl_val_acc)), f"Part 3's dropout function does not reduce the difference between best weights' training accuracy and validation accuracy."
    

def test_part_3_correct_dropout():
    try:
        from part3_dropout import dropout_step
    except ImportError as e:
        raise ImportError(f"Error loading dropout_step from part 3. {e}")
    layer_0 = np.array([[1.0, 2.0, 3.0]])
    target = np.array([[5.0]])
    orig_W_0_1 = np.array([[4.0, 5.0, 6.0], [5.0, 7.0, 9.0], [2.0, 2.0, 2.0]])
    W_0_1 = np.array([[4.0, 5.0, 6.0], [5.0, 7.0, 9.0], [2.0, 2.0, 2.0]])
    W_1_2 = np.array([[4.0], [2.0], [1.0]])
    mask = np.array([0.0, 1.0, 0.0])
    alpha = 0.05
    dropout_step(layer_0, target, W_0_1, W_1_2, mask, alpha)
    modified_columns = np.any(W_0_1 != orig_W_0_1, axis=0)
    assert np.array_equal(modified_columns, mask.squeeze() != 0), "The set of modified weights is not the same as the mask."


if __name__ == "__main__":
    tests = [name for name in dir() if name.startswith('test_')]
    for test_name in sorted(tests):
        test_func = globals()[test_name]
        try:
            test_func()
            print(f' PASS: {test_name}')
        except AssertionError as e:
            print(f' FAIL: {test_name} -- {e}')

