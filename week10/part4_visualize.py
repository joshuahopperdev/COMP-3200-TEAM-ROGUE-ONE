import numpy as np
import matplotlib.pyplot as plt
from data import load_mnist, one_hot
from forms import tanh, tanh_deriv, softmax
from conv import get_patches, cnn_forward
from part3_train import cnn_full_train

if __name__ == "__main__":
    # Load MNIST dataset as 3 sets: 1000-long training set, 1000-long validation set,
    # and unknown-length test set.
    (X_train, y_train), (X_test, y_test) = load_mnist()
    X_val, y_val = X_train[1000:2000], one_hot(y_train[1000:2000])
    X_train, y_train = X_train[:1000], one_hot(y_train[:1000])
    X_test, y_test = X_test, one_hot(y_test)
    alpha, epochs, batch_size = (0.02, 300, 128)

    kernels, weights, _ = cnn_full_train(X_train, y_train, X_val, y_val, alpha, epochs, batch_size)

    # get the shape ready for 3x3
    filters = kernels.T.reshape(16, 3, 3)

    figure, axis = plt.subplots(4, 4, figsize=(9, 9))

    # Need 0-255
    normalized_pixels = (filters - filters.min()) / (filters.max() - filters.min())
    gray_pixels = (normalized_pixels * 255).astype(np.uint8)

    for i, j in enumerate(axis.flat):
        j.imshow(gray_pixels[i], cmap="gray", vmin=0, vmax=255)
        j.axis("off")

    plt.savefig("kernels.png")

    plt.show()

"""
The first image in the top left, is seemingly a light corner detection. Somewhat similar to the 8th(row 2/column 4)
but lighter. Row 1/column 4, row 2/column 1 and row 3/column 4 are also very similar in nature. Row 4/column 1/column 3 seem
to be line horizontal and line vertical. Row 2/column 3 is another edge like corner. Row 3/column 1 is some kind of lined/corner
pattern.

The rest are more difficult to spot any meaningful pattern. Row 2/column 2 have some gradient looking lines, but that doesn't 
immediately register as something distinct. Others like row 2/column 1 and row 4/column 4 outright just don't seem to have anything
at all recognizable.
"""


