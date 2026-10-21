import numpy as np
import forms as av

from data import load_mnist, one_hot
from conv import get_patches, forward
from part3_train import train_step, train


def test_patch_shape():
    test1 = get_patches(np.random.randn(100, 28, 28))
    test2 = get_patches(np.random.randn(1, 6, 6))
    test3 = get_patches(np.random.randn(50, 2048, 2048))

    assert test1.shape == (67600, 9), f"Input shape (100, 28, 28) -- expected shape (67600, 9), got {test1.shape}"
    assert test2.shape == (36, 9), f"Input shape (1, 6, 6) -- expected (36, 9), got {test2.shape}"
    assert test3.shape == (2048 * 2048 * 50, 9), f"Input shape (50, 2048, 2048) -- expected (209715200, 9), got {test3.shape}"

def test_patch_contents():
    truth = np.array([[[1, 2, 3, 4, 5, 6],
                       [7, 8, 9, 10, 11, 12],
                       [13, 14, 15, 16, 17, 18],
                       [19, 20, 21, 22, 23, 24],
                       [25, 26, 27, 28, 29, 30],
                       [31, 32, 33, 34, 35, 36]]]) # (1, 6, 6)
    truth_patch = truth[:, :4, :4].flatten()
    first_patch = get_patches(truth)[0]
    assert first_patch == truth_patch, f"First patch should be the top-left corner, got {first_patch}"

# TODO: finish after part 2 is finished
def test_forward_shape():
    # somehow get layer_1
    batch = 128
    layer_1 = forward()
    assert layer_1.shape == (batch, 10816), f"Layer 1's shape should be (batch, 10816), got {layer_1.shape}"
    assert np.all(np.array([-1 < o < 1 for o in layer_1])), "Tanh output should be in range (-1, 1)"

# NOTE: copied from assignment; may need tweaking later
def test_step():
    rng = np.random.RandomState(0)
    layer_0 = rng.rand(3, 28, 28)  # 3 tiny "images"
    kernels, W_1_2 = 0.5 * rng.randn(9, 16), 0.05 * rng.randn(10816, 10)
    Y = np.eye(10)[[3, 7, 1]]
    mask = rng.randint(2, size=(3, 10816))
    fp, es = get_patches(layer_0)
    ko = fp.dot(kernels)
    l1 = np.tanh(ko.reshape(layer_0.shape[0], 676 * 16))
    layer_2 = av.softmax((l1 * mask * 2).dot(W_1_2))
    delta_2 = (layer_2 - Y) / 3  # batch average: 3 images
    d1 = delta_2.dot(W_1_2.T) * (1 - l1 ** 2) * mask * 2
    expected = fp.T.dot(d1.reshape(ko.shape))  # (9, 16)
    got = train_step(layer_0, Y, kernels.copy(), W_1_2.copy(), mask, alpha=0.02)
    assert got.shape == (9, 16)
    # either backward-mask convention: mask * 2 (exact) or mask (Trask)
    assert np.allclose(got, expected) or np.allclose(got, expected / 2)

# NOTE: Anticipate signature and return value changes
def test_learn():
    (X_train, y_train), _ = load_mnist()
    X_val, y_val = X_train[1000:2000], one_hot(y_train[1000:2000])
    X_train, y_train = X_train[:1000], one_hot(y_train[:1000])
    val_acc = train(X_train, y_train, X_val, y_val,
                    alpha=0.2, epochs=30, batch_size=128)
    assert val_acc[-1] > 0.75, f"Final validation accuracy should be comfortably above chance, got {val_acc}"

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
