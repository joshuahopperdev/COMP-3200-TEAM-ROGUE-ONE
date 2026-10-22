import numpy as np
from forms import tanh, tanh_deriv, softmax


def get_patches(layer):  # layer: (batch, 28, 28)
    batch_size = layer.shape[0]
    patches = []
    for r in range(layer.shape[1] - 3 + 1):  # 0..25
        for c in range(layer.shape[2] - 3 + 1):  # 0..25
            patch = layer[:, r : r + 3, c : c + 3]  # (batch, 3, 3)
            # the 1 is a slot for the patch-position axis
            patches.append(patch.reshape(batch_size, 1, 3, 3))

    # concatenate fills that slot: 676 positions side by side
    expanded = np.concatenate(patches, axis=1)  # (batch, 676, 3, 3)
    es = expanded.shape

    # flatten EACH patch to 9 numbers, one row per (image, position)
    return expanded.reshape(es[0] * es[1], 9), es  # (batch*676, 9)


# e.g. layer_0 of 50 x 28 x 28, 50 images of 28 rows of 28 pictures
# flat_patches will then be 50*676 x 9
# and 16 kernels will be a 9 x 16 matrix, so multiplying
# flat_patches by kernels gives a 50*676 x 16 output
def cnn_forward(layer_0, kernels, W_1_2):
    # build a set of patches from layer_0
    flat_patches = get_patches(layer_0)
    n = layer_0.shape[0]
    kernel_out = flat_patches.dot(kernels)

    layer_1 = tanh(kernel_out.reshape(n, 676 * 16))

    # layer_1's shape is confirmed in test_watchers.py

    slope = tanh_deriv(layer_1)

    mask = np.random.randint(2, size=layer_1.shape)
    layer_1 *= mask * 2

    layer_2 = softmax(layer_1.dot(W_1_2))

    return layer_1, layer_2, slope


if __name__ == "__main__":
    # Some dummy layer_0s, kernels, and W_1_2s
    rng = np.random.RandomState(0)
    batch = 14
    layer_0 = rng.rand(batch, 28, 28)
    kernels, W_1_2 = 0.5 * rng.randn(9, 16), 0.05 * rng.randn(10816, 10)
    layer_1, layer_2, slope = cnn_forward(layer_0, kernels, W_1_2)

    print(f"Layer 1's shape is {layer_1.shape}...it should be ({batch}, 10816)")
