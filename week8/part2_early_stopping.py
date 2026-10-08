import copy
import numpy as np
from data import load_mnist, one_hot

# I'm trying not to reinvent the wheel, but if
# evaluate takes the weights from its own file
# instead of the weights in this file, I may need
# to write at least my own evaluate function.
from part1_memorize import relu, relu_d, evaluate

# Set up globals
np.random.seed(1)
alpha = 0.005
hidden_size = 40

W_0_1 = 0.2 * np.random.random((784, hidden_size)) - 0.1
W_1_2 = 0.2 * np.random.random((hidden_size, 10)) - 0.1


def train_early_stopping(
    X_train, Y_train, X_val, Y_val, iterations=350, patience=10, verbose=False
):
    train_hist = []
    val_hist = []

    # New: variables for early stopping
    best_val = 0.0  # Used to compare with the current validation accuracy
    best_weights = None  # The fallback in case patience runs out
    since_improvement = 0  # How we know when to run out of patience

    for it in range(iterations):
        correct = 0

        for i in range(len(X_train)):
            # ---- Forward ----
            layer_0 = X_train[i : i + 1]
            layer_1 = relu(layer_0.dot(W_0_1))
            layer_2 = layer_1.dot(W_1_2)

            # ---- Score ----
            correct += int(np.argmax(layer_2) == np.argmax(Y_train[i]))

            # ---- Backward ----
            delta_2 = layer_2 - Y_train[i : i + 1]
            delta_1 = delta_2.dot(W_1_2.T) * relu_d(layer_1)

            W_1_2 -= alpha * layer_1.T.dot(delta_2)
            W_0_1 -= alpha * layer_0.T.dot(delta_1)

        train_acc = correct / len(X_train)
        val_acc = evaluate(X_val, Y_val, W_0_1, W_1_2)

        # Check if we're on the right track by comparing
        # the validation accuracy to the best val_acc we
        # have so far
        if val_acc > best_val:
            best_val = val_acc
            best_weights = (copy.deepcopy(W_0_1), copy.deepcopy(W_1_2))
            since_improvement = 0
        else:
            since_improvement += 1
            if since_improvement >= patience:
                if verbose:
                    print(f"Stopping at iteration {it}; best val_acc={best_val:.3f}")
                break

        train_hist.append(train_acc)
        val_hist.append(val_acc)

        if verbose:
            print(
                f"Iteration {it+1}/{iterations} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}"
            )

    W_0_1, W_1_2 = best_weights

    return W_0_1, W_1_2, train_hist, val_hist


if __name__ == "__main__":
    # Load MNIST
    (X_train, y_train), (X_test, y_test) = load_mnist()
    X_val, Y_val = X_train[1000:2000], one_hot(y_train[1000:2000])
    X_train, Y_train = X_train[:1000], one_hot(y_train[:1000])

    final_w_0_1, final_w_1_2, train_history, val_history = train_early_stopping(
        X_train, Y_train, X_val, Y_val, iterations=350, patience=10, verbose=True
    )

    print(f"Final weights: W_0_1={final_w_0_1}; W_1_2={final_w_1_2}")
    print(f"Training history: {train_history[::50]}")
    print(f"Validation accuracy history: {val_history[::50]}")
