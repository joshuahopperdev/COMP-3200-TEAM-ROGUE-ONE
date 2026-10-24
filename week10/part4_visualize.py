import numpy as np
from data import load_mnist, one_hot
from forms import tanh, tanh_deriv, softmax
from conv import get_patches, cnn_forward
from part3_train import cnn_full_train

# Load MNIST dataset as 3 sets: 1000-long training set, 1000-long validation set,
# and unknown-length test set.
(X_train, y_train), (X_test, y_test) = load_mnist()
X_val, y_val = X_train[1000:2000], one_hot(y_train[1000:2000])
X_train, y_train = X_train[:1000], one_hot(y_train[:1000])
X_test, y_test = X_test, one_hot(y_test)

alpha, epochs, batch_size = (0.02, 300, 128)

if __name__ == "__main__":
    kernels, weights = cnn_full_train(X_train, y_train, X_val, y_val, alpha, epochs, batch_size)
    print(f"Kernels:\n{kernels}")





