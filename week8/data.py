import os, urllib.request, numpy as np

URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"

def load_mnist(path="mnist.npz"):
    if not os.path.exists(path):
        urllib.request.urlretrieve(URL, path)
    d = np.load(path)
    X_train = d["x_train"].reshape(len(d["x_train"]), 784) / 255.0
    X_test = d["x_test"].reshape(len(d["x_test"]), 784) / 255.0
    return (X_train, d["y_train"]), (X_test, d["y_test"])

def one_hot(labels, n=10):
    out = np.zeros((len(labels), n))
    out[np.arange(len(labels)), labels] = 1
    return out