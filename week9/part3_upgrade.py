import numpy as np
from data import load_mnist, one_hot
import matplotlib.pyplot as plt
from forms import tanh, tanh_deriv, softmax

# Load MNIST dataset as 3 sets: 1000-long training set, 1000-long validation set,
# and unknown-length test set.
(X_train, y_train), (X_test, y_test) = load_mnist()
X_val, y_val = X_train[1000:2000], one_hot(y_train[1000:2000])
X_train, y_train = X_train[:1000], one_hot(y_train[:1000])
X_test, y_test = X_test, one_hot(y_test)

# Record default seed and various variables
np.random.seed(1) # seed FIRST, then initialise
alpha, epochs, hidden_size, batch_size = (0.02, 300, 100, 100)
# Roll up our random weights: one a 784 x 100 matrix intaking
# all values from an image, as random values from -.01 to .01,
# one a 100 x 10 matrix intaking the hidden layer and outputting
# log-probs for each of the possible numbers, as random values
# from -.1 to .1
W_0_1 = 0.02 * np.random.random((784, hidden_size)) - 0.01
W_1_2 = 0.2 * np.random.random((hidden_size, 10)) - 0.1

# Run a single training step over a full batch, updating
# weight matrices in place, taking in a single mask for the
# whole batch from the external function, using any provided
# activation functions on the hidden layer.
def train_step(layer_0, y_batch, W_0_1, W_1_2, mask,
alpha, hidden_f=tanh, hidden_d=tanh_deriv):
    # calculate layer 1
    layer_1 = hidden_f(layer_0 @ W_0_1)
    # save the derivative of layer 1 before moving on; we'll use
    # it later.
    slope = hidden_d(layer_1)

    # intentionally uses the same mask for all in the batch
    layer_1 *= mask * 2

    # calculate layer 2, convert from log probs to full probs
    # note that we don't subtract the largest value from all values
    # first, so we're not super confident that we won't get any 
    # infinities
    layer_2 = softmax(layer_1 @ W_1_2)

    # count how many are in the batch so we can divide by that much
    batch_size = layer_0.shape[0]

    # calculate for the whole batch
    delta_2 = (layer_2 - y_batch) / batch_size

    # here's where we use slope from earlier
    delta_1 = delta_2 @ W_1_2.T * slope

    # scale in our mask
    delta_1 *= mask * 2

    # weight updates
    W_1_2 -= alpha * layer_1.T @ delta_2
    W_0_1 -= alpha * layer_0.T @ delta_1

    # a little odd that we return this, but it's in the assignment spec
    return delta_1


def evaluate(X, y, W_0_1, W_1_2, hidden_f=tanh):
    """Runs through a set of two weights W_0_1 and W_1_2, applying the
    listed activation function to layer 1, for each X in set X, checking
    final output against the corresponding y in set y. Does not use
    softmax on the output; this does not influence the result.
    Returns the fraction of input that was correctly predicted."""
    correct = 0
    for i in range(len(X)):
        layer_1 = hidden_f(X[i:i+1].dot(W_0_1))
        layer_2 = layer_1.dot(W_1_2)
        correct += int(np.argmax(layer_2) == np.argmax(y[i]))
    
    return correct / len(X)




def train_forms(X_train, y_train, X_val, y_val, X_test, y_test, alpha = 0.005, iterations=350, batch_size=100):
    np.random.seed(1)

    val_hist = []

    for it in range(iterations):

        # note that this discards incomplete batches, e.g. 
        # batch size 100, total size 1079 will ditch the
        # last 79 entries
        for i in range(len(X_train) // batch_size):
            start, end = i * batch_size, (i+1) * batch_size
            # ---- Forward ----
            layer_0 = X_train[start:end]
            mask = np.random.randint(2, size=hidden_size) # ~50% ones, ~50% zeros
            train_step(X_train[start:end], y_train[start:end], W_0_1, W_1_2, mask, alpha, tanh, tanh_deriv)




        val_hist.append(evaluate(X_val, y_val, W_0_1, W_1_2))

        # print(f"Iteration {it+1}/{iterations} - Train Acc: {evaluate(X_train, y_train, W_0_1, W_1_2):.4f}, Val Acc: {evaluate(X_val, y_val, W_0_1, W_1_2):.4f}")
    return W_0_1, W_1_2, val_hist

if __name__ == "__main__":
    W_0_1_batch, W_1_2_batch, val_hist_batch = train_forms(X_train, y_train, X_val, y_val, X_test, y_test, alpha, epochs, batch_size)
    final_test_score = evaluate(X_test, y_test, W_0_1_batch, W_1_2_batch)
    print(f"Final score after {epochs} epochs on the test set: {final_test_score}")


    # ---- Plot ----
    plt.plot(range(1, epochs + 1), val_hist_batch, label="Validation Accuracy with Batching")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy By Epoch")
    plt.legend()
    plt.grid(True)

    plt.savefig("upgrade_plot.png")
    plt.show()