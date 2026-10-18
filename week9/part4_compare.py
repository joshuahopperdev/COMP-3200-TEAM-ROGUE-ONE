import numpy as np
from data import load_mnist, one_hot
import matplotlib.pyplot as plt
from part3_upgrade import train_step, evaluate

from forms import sigmoid, sigmoid_deriv, tanh, tanh_deriv, relu, relu_deriv

(X_train, y_train), (X_test, y_test) = load_mnist()
X_val, y_val = X_train[1000:2000], one_hot(y_train[1000:2000])
X_train, y_train = X_train[:1000], one_hot(y_train[:1000])
X_test, y_test = X_test, one_hot(y_test)

np.random.seed(1) 
alpha, epochs, hidden_size, batch_size = (0.02, 300, 100, 100)

W_0_1_TANH = 0.02 * np.random.random((784, hidden_size)) - 0.01
W_1_2_TANH = 0.2 * np.random.random((hidden_size, 10)) - 0.1

W_0_1_RELU = np.random.random((784, hidden_size)) * np.sqrt(2 / 784)
W_1_2_RELU = np.random.random((hidden_size, 10)) * np.sqrt(2 / hidden_size)

# Glorot for sigmoid. I remembered we mentioned this briefly in ML.
W_0_1_SIG = np.random.randn(784, hidden_size) * np.sqrt(2 / (784 + hidden_size))
W_1_2_SIG = np.random.randn(hidden_size, 10) * np.sqrt(2 / (hidden_size + 10))



# This is taken from Part 3 and modified to be more modular for the types of activation functions.
def train_forms(X_train, y_train, X_val, y_val, w_0_1, w_1_2, alpha = 0.005, epochs=350, batch_size=100, func=tanh, func_deriv=tanh_deriv):
    val_hist = []
    # For each epoch...
    for _ in range(epochs):
        for i in range(len(X_train) // batch_size):
            start, end = i * batch_size, (i+1) * batch_size
            mask = np.random.randint(2, size=hidden_size)
            train_step(X_train[start:end], y_train[start:end], w_0_1, w_1_2, mask, alpha, func, func_deriv)

        val_hist.append(evaluate(X_val, y_val, w_0_1, w_1_2, func))

        # print(f"Epoch {it+1}/{epoch} - Train Acc: {evaluate(X_train, y_train, W_0_1, W_1_2):.4f}, Val Acc: {evaluate(X_val, y_val, W_0_1, W_1_2):.4f}")
    return val_hist



if __name__ == "__main__":
    tanh_hist_batch = train_forms(X_train, y_train, X_val, y_val, W_0_1_TANH, W_1_2_TANH, alpha, epochs, batch_size, tanh, tanh_deriv)

    tanh_final_test_score = evaluate(X_test, y_test, W_0_1_TANH, W_1_2_TANH, tanh)

    print(f"Final score after {epochs} epochs on the tanh test set: {tanh_final_test_score}")


    relu_hist_batch = train_forms(X_train, y_train, X_val, y_val, W_0_1_RELU, W_1_2_RELU, alpha, epochs, batch_size, relu, relu_deriv)
    
    relu_final_test_score = evaluate(X_test, y_test, W_0_1_RELU, W_1_2_RELU, relu)

    print(f"Final score after {epochs} epochs on the relu test set: {relu_final_test_score}")


    sig_hist_batch = train_forms(X_train, y_train, X_val, y_val, W_0_1_SIG, W_1_2_SIG, alpha, epochs, batch_size, sigmoid, sigmoid_deriv)
    
    sig_final_test_score = evaluate(X_test, y_test, W_0_1_SIG, W_1_2_SIG, sigmoid)

    print(f"Final score after {epochs} epochs on the sig test set: {sig_final_test_score}")






    # ---- Plot ----
    plt.plot(range(1, epochs + 1), tanh_hist_batch, label="TANH Accuracy")
    plt.plot(range(1, epochs + 1), relu_hist_batch, label="RELU Accuracy")
    plt.plot(range(1, epochs + 1), sig_hist_batch, label="SIGMOID Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy By Activation Function")
    plt.legend()
    plt.grid(True)

    plt.savefig("compare_plot.png")
    plt.show()



