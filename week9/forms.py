import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(out):
    return out * (1 - out)  # takes the OUTPUT


def tanh(x):
    return np.tanh(x)


def tanh_deriv(out):
    return 1 - out**2  # takes the OUTPUT


def relu(x):
    return (x > 0) * x


def relu_deriv(out):
    return out > 0  # takes the OUTPUT


def softmax(z):  # z.shape == (batch, k)
    z = z - np.max(z, axis=1, keepdims=True)  # subtract row max
    exp_z = np.exp(z)  # exp(onentiate)
    return exp_z / exp_z.sum(axis=1, keepdims=True)  # divide by row sum row-sum


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = np.linspace(-6, 6, 400)
    plt.plot(x, sigmoid(x), color="gray", label="sigmoid (0, 1)")
    plt.plot(x, tanh(x), color="blue", label="tanh (-1, 1)")
    plt.plot(x, relu(x), color="orange", label="relu [0, inf)")
    plt.axhline(0, color="black", lw=0.5)
    plt.xlabel("input x")
    plt.ylabel("Form output")
    plt.legend()
    plt.title("Three Forms")
    plt.savefig("forms_plot.png")
    plt.show()
