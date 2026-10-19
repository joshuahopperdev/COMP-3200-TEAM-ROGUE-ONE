import numpy as np
from data import load_mnist, one_hot
import forms as av  # activation function
from part2_nonlinearity import main
from part3_upgrade import train_step, evaluate
from part4_compare import train_forms


def test_softmax_distro():
    # Part 1: checksums
    softmax_sum_test()
    softmax_sum_test(shape=(6, 7))
    softmax_sum_test(seed=42)
    softmax_sum_test(42, (6, 7))

    # Part 2: finity and affinity
    softmax_finite_test(np.array([[1000.0, 1001.0]]), np.array([[0.269, 0.731]]))


def test_ranges():
    x = np.linspace(-15, 15, 1001)
    sigmoid_x = av.sigmoid(x)
    tanh_x = av.tanh(x)
    relu_x = av.relu(x)

    assert (
        np.min(sigmoid_x) > 0 and np.max(sigmoid_x) < 1
    ), f"Sigmoid must be in bounds (0, 1) -- got min={np.min(sigmoid_x)}, max={np.max(sigmoid_x)}"
    assert (
        np.min(tanh_x) > -1 and np.max(tanh_x) < 1
    ), f"Tanh must be in bounds (-1, 1) -- got min={np.min(tanh_x)}, max={np.max(tanh_x)}"
    assert np.all(relu_x >= 0), f"ReLU must be >= 0, got {relu_x}"

    x_idx = np.where(x > 0, 1, 0)
    relu_x_idx = np.where(x != 0, 1, 0)
    assert [
        a == b for a, b in zip(x, relu_x) if a > 0
    ], f"ReLU must have matching indices -- expected {x_idx}, got {relu_x_idx}"


def test_relu_and_slope():
    x = np.linspace(-15, 15, 1001)
    xa = av.relu(x)
    neg_idx = np.asarray([i for i in range(len(xa)) if xa[i] < 0])
    assert np.all(
        np.array([xa[i] == 0 for i in neg_idx])
    ), f"ReLU should be 0 for every negative x -- got  {xa.tolist()}"
    assert np.array_equal(
        av.relu_deriv(xa), x > 0
    ), f"ReLU's slope should match the input's sign -- got {xa.tolist()}"


def test_lin_layers_collapse():
    np.random.seed(1)
    X = np.random.randn(5, 4)
    W1 = np.random.randn(4, 3)
    W2 = np.random.randn(3, 2)

    assert np.allclose(
        (X @ W1) @ W2, X @ (W1 @ W2), rtol=1e-9, atol=1e-9
    ), f"Input with weights should be associative"
    assert not np.allclose(
        av.relu(X @ W1) @ W2, X @ (W1 @ W2)
    ), f"ReLU should break linearity"


def test_compare_tanh_sigmoid():
    (X_train, y_train), (X_test, y_test) = load_mnist()
    X_val, y_val = X_train[1000:2000], one_hot(y_train[1000:2000])
    X_train, y_train = X_train[:1000], one_hot(y_train[:1000])
    X_test, y_test = X_test, one_hot(y_test)

    np.random.seed(1)
    alpha, epochs, hidden_size, batch_size = (0.02, 30, 100, 100)

    W_0_1_TANH = 0.02 * np.random.random((784, hidden_size)) - 0.01
    W_1_2_TANH = 0.2 * np.random.random((hidden_size, 10)) - 0.1

    # Glorot for sigmoid. I remembered we mentioned this briefly in ML.
    W_0_1_SIG = np.random.randn(784, hidden_size) * np.sqrt(2 / (784 + hidden_size))
    W_1_2_SIG = np.random.randn(hidden_size, 10) * np.sqrt(2 / (hidden_size + 10))

    train_forms(
        X_train,
        y_train,
        X_val,
        y_val,
        W_0_1_TANH,
        W_1_2_TANH,
        alpha,
        epochs,
        batch_size,
        av.tanh,
        av.tanh_deriv,
    )

    tanh_final_test_score = evaluate(X_test, y_test, W_0_1_TANH, W_1_2_TANH, av.tanh)

    train_forms(
        X_train,
        y_train,
        X_val,
        y_val,
        W_0_1_SIG,
        W_1_2_SIG,
        alpha,
        epochs,
        batch_size,
        av.sigmoid,
        av.sigmoid_deriv,
    )

    sig_final_test_score = evaluate(X_test, y_test, W_0_1_SIG, W_1_2_SIG, av.sigmoid)

    assert (
        tanh_final_test_score > sig_final_test_score
    ), f"Tanh final validation accuracy should be greater than sigmoid -- got tanh_final_val_acc={tanh_final_test_score}, sigmoid_final_val_acc={sig_final_test_score}"


def softmax_sum_test(seed=1, shape=(8, 10)):
    np.random.seed(seed)
    X = np.random.randn(shape[0], shape[1])
    Xa = av.softmax(X)
    Xsums = np.sum(Xa, axis=1)

    # Test: each row sums to 1; thus use axis=1 to sum the rows
    assert np.allclose(
        Xsums, 1, atol=1e-9, rtol=1e-9
    ), f"Softmax needs a stable distribution -- got {Xsums.tolist()}"


def softmax_finite_test(X, expected):
    Xa = av.softmax(X)
    # Test 1: no nan
    assert not np.any(np.isnan(X)), f"X cannot have NaN -- got {np.isnan(X)}"
    # Test 2: close to expected
    assert np.allclose(
        Xa, expected, rtol=0.02, atol=0.02
    ), f"Must be close to the expected value -- expected {expected}, got {Xa}"


# Copied from other tests
if __name__ == "__main__":
    tests = [name for name in dir() if name.startswith("test_")]
    for test_name in sorted(tests):
        test_func = globals()[test_name]
        try:
            test_func()
            print(f" PASS: {test_name}")
        except AssertionError as e:
            print(f" FAIL (AssertionError): {test_name} -- {e}")
