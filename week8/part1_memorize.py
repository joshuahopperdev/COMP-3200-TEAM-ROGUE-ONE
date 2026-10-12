import numpy as np
import matplotlib.pyplot as plt
from data import load_mnist, one_hot

(X_train, y_train), (X_test, y_test) = load_mnist()
X_val, Y_val = X_train[1000:2000], one_hot(y_train[1000:2000])
X_train, Y_train = X_train[:1000], one_hot(y_train[:1000])

np.random.seed(1) # seed FIRST, then initialise
alpha, hidden_size, iterations = 0.005, 40, 350
W_0_1 = 0.2 * np.random.random((784, hidden_size)) - 0.1 # [-0.1, 0.1)
W_1_2 = 0.2 * np.random.random((hidden_size, 10)) - 0.1

def relu(x): return (x > 0) * x
def relu_d(x): return (x > 0) # > 0, not >= 0: an off unit gets no blame

def evaluate(X, Y, W_0_1, W_1_2):
    correct = 0
    for i in range(len(X)):
        layer_1 = relu(X[i:i+1].dot(W_0_1))
        layer_2 = layer_1.dot(W_1_2)
        correct += int(np.argmax(layer_2) == np.argmax(Y[i]))
    
    return correct / len(X)

def train(X_train, Y_train, X_val, Y_val, iterations=350):
    np.random.seed(1)
    alpha = 0.005
    hidden_size = 40

    W_0_1 = 0.2 * np.random.random((784, hidden_size)) - 0.1
    W_1_2 = 0.2 * np.random.random((hidden_size, 10)) - 0.1

    train_hist = []
    val_hist = []

    for it in range(iterations):
        correct = 0
        
        for i in range(len(X_train)):
            # ---- Forward ----
            layer_0 = X_train[i:i+1]
            layer_1 = relu(layer_0.dot(W_0_1))
            layer_2 = layer_1.dot(W_1_2)

            # ---- Score ----
            correct += int(np.argmax(layer_2) == np.argmax(Y_train[i]))

            # ---- Backward ----
            delta_2 = layer_2 - Y_train[i:i+1]
            delta_1 = delta_2.dot(W_1_2.T) * relu_d(layer_1)

            W_1_2 -= alpha * layer_1.T.dot(delta_2)
            W_0_1 -= alpha * layer_0.T.dot(delta_1)

        train_acc = correct / len(X_train)
        val_acc = evaluate(X_val, Y_val, W_0_1, W_1_2)

        train_hist.append(train_acc)
        val_hist.append(val_acc)

        print(f"Iteration {it+1}/{iterations} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    return W_0_1, W_1_2, train_hist, val_hist

if __name__ == "__main__":
    W_0_1, W_1_2, train_hist, val_hist = train(X_train, Y_train, X_val, Y_val, iterations)

    # Find the best validation accuracy
    best_iteration = np.argmax(val_hist) + 1
    best_val_acc = val_hist[best_iteration - 1]
    print(f"Validation accuracy peaked at iteration {best_iteration} with accuracy {best_val_acc:.4f}")

    # ---- Plot ----
    plt.plot(range(1, iterations + 1), train_hist, label="Training Accuracy")
    plt.plot(range(1, iterations + 1), val_hist, label="Validation Accuracy")

    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)

    plt.savefig("curves_part1.png")
    plt.show()