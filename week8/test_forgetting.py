import numpy as np
import inspect

from data import load_mnist, one_hot

(X_train, y_train), (X_test, y_test) = load_mnist()
X_val, y_val = X_train[1000:2000], one_hot(y_train[1000:2000])
X_train, y_train = X_train[:1000], one_hot(y_train[:1000])

def evaluate(X, Y, W_0_1, W_1_2):
    correct = 0
    for i in range(len(X)):
        layer_1 = relu(X[i:i+1].dot(W_0_1))
        layer_2 = layer_1.dot(W_1_2)
        correct += int(np.argmax(layer_2) == np.argmax(Y[i]))
    
    return correct / len(X)

def test_part_1_overfitting():
    try:
        from part1_memorize import train
    except ImportError as e:
        raise ImportError(f"Error loading train from part 1. {e}")
    _, _, train_hist, val_hist = train(X_train, y_train, X_val, y_val,
iterations=50)

    assert (train_hist[-1] > 0.95), f"Part 1: final training accuracy should be greater than 0.95; final training accuracy was {train_hist[-1]}"
    assert (train_hist[-1] - val_hist[-1] > 0.1), f"Part 1: inal training accuracy should be at least 0.1 greater than final validation accuracy; final difference was {train_hist[-1]-val_hist[-1]}"


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
    # contains the word "test"....
    assert (not ("test" in train_1_sig)), "Part 1's signature contains the word \"test\""
    assert (not ("test" in train_1_source)), "Part 1's source contains the word \"test\""
    assert (not ("test" in train_2_sig)), "Part 2's signature contains the word \"test\""
    assert (not ("test" in train_2_source)), "Part 2's source contains the word \"test\""
    assert (not ("test" in train_3_sig)), "Part 3's signature contains the word \"test\""
    assert (not ("test" in train_3_source)), "Part 3's source contains the word \"test\""
    assert (not ("test" in train_4_sig)), "Part 4's signature contains the word \"test\""
    assert (not ("test" in train_4_source)), "Part 4's source contains the word \"test\""
    

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
    assert np.allclose(forward_pass, np.amax(val_hist), 1e-9), f"Part 2's returned weights do not produce the same value as val_hist's best value: {forward_pass} vs {np.amax(val_hist)}"

def test_part_3_dropout_improves():
    try:
        from part3_dropout import 
    except ImportError:
        raise ImportError("")

def test_part_3_correct_dropout():
    try:
        from part3_dropout import 
    except ImportError:
        raise ImportError("")

def test_correct_early_stopping_time():
    try:
        from part2_early_stopping import 
    except ImportError:
        raise ImportError("")
    

def test_correct_set_splitting():
    try:
        from part1_memorize import 
        from part2_early_stopping import 
        from part3_dropout import 
        from part4_batching import 
    except ImportError:
        raise ImportError("")


def test_unstopped_works():
