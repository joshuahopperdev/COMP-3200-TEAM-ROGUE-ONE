import numpy as np
import matplotlib.pyplot as plt
from data import load_mnist, one_hot
import time
from part1_memorize import relu, relu_d, evaluate
from part3_dropout import train_dropout


"""
Most of this is copied directly from part 3. This part 4
is adding in batching and seeing how it compares with the
previous dropout only.
"""

(X_train, y_train), (X_test, y_test) = load_mnist()
X_val, Y_val = X_train[1000:2000], one_hot(y_train[1000:2000])
X_train, Y_train = X_train[:1000], one_hot(y_train[:1000])

np.random.seed(1) # seed FIRST, then initialise
alpha, hidden_size, iterations, batch_size = 0.1, 40, 350, 100
W_0_1 = 0.2 * np.random.random((784, hidden_size)) - 0.1 # [-0.1, 0.1)
W_1_2 = 0.2 * np.random.random((hidden_size, 10)) - 0.1

def train_batched(X_train, Y_train, X_val, Y_val, alpha = 0.005, iterations=350, batch_size=100, dropout = True):
    start_time = time.perf_counter()
    np.random.seed(1)
    alpha = alpha
    hidden_size = 40

    W_0_1 = 0.2 * np.random.random((784, hidden_size)) - 0.1
    W_1_2 = 0.2 * np.random.random((hidden_size, 10)) - 0.1

    train_hist = []
    val_hist = []

    for it in range(iterations):
        correct = 0
        
        for i in range(len(X_train) // batch_size):
            start, end = i * batch_size, (i+1) * batch_size
            # ---- Forward ----
            layer_0 = X_train[start:end]
            layer_1 = relu(layer_0.dot(W_0_1))

            # default to no mask...
            mask = np.ones(layer_1.shape)
            # but if we are using a mask, generate and apply it here
            if dropout == True:
                mask = np.random.randint(2, size=layer_1.shape) # ~50% ones, ~50% zeros
                layer_1 *= mask * 2 # zero (on average) half, double the rest


            layer_2 = layer_1.dot(W_1_2)

            # ---- Score ----
            correct += np.sum(np.argmax(layer_2, axis=1) == np.argmax(Y_train[start:end], axis=1))

            # ---- Backward ----
            delta_2 = (layer_2 - Y_train[start:end]) / batch_size
            # make sure not to modify any weights by masked values!
            # If they didn't contribute to the final run, don't modify them
            # as if they did!
            delta_1 = delta_2.dot(W_1_2.T) * relu_d(layer_1) * mask # mask is all 1s if dropout is false, else it's the dropout list

            W_1_2 -= alpha * layer_1.T.dot(delta_2)
            W_0_1 -= alpha * layer_0.T.dot(delta_1)

        train_acc = correct / len(X_train)
        val_acc = evaluate(X_val, Y_val, W_0_1, W_1_2)

        train_hist.append(train_acc)
        val_hist.append(val_acc)

        # print(f"Iteration {it+1}/{iterations} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    end_time = time.perf_counter()
    return W_0_1, W_1_2, train_hist, val_hist, end_time - start_time

if __name__ == "__main__":
    W_0_1_batch, W_1_2_batch, train_hist_batch, val_hist_batch, time_taken_batch = train_batched(X_train, Y_train, X_val, Y_val, alpha, iterations, batch_size, True)

    best_pass_b = np.argmax(val_hist_batch) + 1
    best_val_acc_b = val_hist_batch[best_pass_b-1]

    print("With Batching:")
    print("________________")
    print("Final training round's accuracy on the training set was:")
    print(train_hist_batch[-1])
    print("Final weights' accuracy on the training set was:")
    print(evaluate(X_train, Y_train, W_0_1_batch, W_1_2_batch))
    print(f"Best validation accuracy, in pass {best_pass_b}, was:")
    print(best_val_acc_b)
    print("Final validation accuracy was:")
    print(val_hist_batch[-1])

    start_time = time.perf_counter()
    W_0_1_drop, W_1_2_drop, train_hist_dropout, val_hist_dropout = train_dropout(X_train, Y_train, X_val, Y_val, iterations, True)
    end_time = time.perf_counter()
    time_taken_drop = end_time - start_time

    best_pass_d = np.argmax(val_hist_dropout) + 1
    best_val_acc_d = val_hist_dropout[best_pass_d-1]
    print("Without Batching:")
    print("________________")
    print("Final training round's accuracy on the training set was:")
    print(train_hist_dropout[-1])
    print("Final weights' accuracy on the training set was:")
    print(evaluate(X_train, Y_train, W_0_1_drop, W_1_2_drop))
    print(f"Best validation accuracy, in pass {best_pass_d}, was:")
    print(best_val_acc_d)
    print("Final validation accuracy was:")
    print(val_hist_dropout[-1])

    print("\nTime Taken Difference\n")
    print(f"Batching: {time_taken_batch}")
    print(f"Without Batching: {time_taken_drop}")

    # Batching
    # Final Validation: 0.842
    # Not batching
    # Final Validation: 0.853

    # ---- Plot ----
    plt.plot(range(1, iterations + 1), train_hist_batch, label="Training Accuracy with Batching")
    plt.plot(range(1, iterations + 1), val_hist_batch, label="Validation Accuracy with Batching")

    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)

    plt.savefig(f"curves_part4_a-{alpha}_i-{iterations}_b-{batch_size}.png")
    plt.show()