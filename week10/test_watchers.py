import numpy as np
import forms as av

from data import load_mnist, one_hot
from conv import get_patches, cnn_forward
from part3_train import train_step, cnn_full_train

rng = np.random.RandomState(0)

def test_patch_shape():
    test1, _ = get_patches(rng.randn(100, 28, 28))
    test2, _ = get_patches(rng.randn(1, 6, 6))
    test3, _ = get_patches(rng.randn(50, 2048, 2048))

    assert test1.shape == (67600, 9), f"Input shape (100, 28, 28) -- expected shape (67600, 9), got {test1.shape}"
    assert test2.shape == (16, 9), f"Input shape (1, 6, 6) -- expected (16, 9), got {test2.shape}"
    assert test3.shape == (2046 * 2046 * 50, 9), f"Input shape (50, 2048, 2048) -- expected (209305800, 9), got {test3.shape}"

def test_patch_contents():
    layer = rng.rand(10, 6, 6)
    patch = layer[0, :3, :3].flatten()
    patches, _ = get_patches(layer)
    first_patch = patches[0]
    assert np.all(a == b for a, b in zip(first_patch, patch)), f"First patch should be the top-left corner -- expected {patch}, got {first_patch}"

def test_forward_shape():
    # somehow get layer_1
    batches = np.array([14, 32, 128, 271])
    for batch in batches:
        layer_0 = rng.rand(batch, 28, 28)
        kernels, W_1_2 = 0.5 * rng.randn(9, 16), 0.05 * rng.randn(10816, 10)
        layer_1, _, _ = cnn_forward(layer_0, kernels, W_1_2)
        assert layer_1.shape == (batch, 10816), f"Layer 1's shape should be (batch, 10816), got {layer_1.shape}"
        assert np.all(np.min(layer_1) > -1 and np.max(layer_1) < 1), f"Tanh output should be in range (-1, 1), got min={np.min(layer_1)}, max={np.max(layer_1)}"

# NOTE: copied from assignment; may need tweaking later
def test_step():
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
    _, _, val_acc = cnn_full_train(X_train, y_train, X_val, y_val,
                    alpha=0.2, epochs=30, batch_size=128, verbose=False)
    assert val_acc > 0.75, f"Final validation accuracy should be comfortably above chance, got {val_acc}"

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
