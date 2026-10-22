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

# presumes average dropout rate of 50%
def dropout_step(layer_0, target, W_0_1, W_1_2, mask, alpha):
    # redundant, but clarifies
    layer_0 = layer_0
    layer_1 = relu(layer_0.dot(W_0_1)) * mask * 2
    layer_2 = layer_1.dot(W_1_2)

    delta_2 = layer_2 - target
    delta_1 = delta_2.dot(W_1_2.T) * relu_d(layer_1) * mask
    W_1_2 -= alpha * layer_1.T.dot(delta_2)
    W_0_1 -= alpha * layer_0.T.dot(delta_1)



def train_dropout(X_train, Y_train, X_val, Y_val, iterations=350, dropout = True):
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

            # default to no mask...
            mask = np.ones(layer_1.shape)
            # but if we are using a mask, generate and apply it here
            if dropout == True:
                mask = np.random.randint(2, size=layer_1.shape) # ~50% ones, ~50% zeros
                layer_1 *= mask * 2 # zero (on average) half, double the rest


            layer_2 = layer_1.dot(W_1_2)

            # ---- Score ----
            correct += int(np.argmax(layer_2) == np.argmax(Y_train[i]))

            # ---- Backward ----
            delta_2 = layer_2 - Y_train[i:i+1]
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

#        print(f"Iteration {it+1}/{iterations} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    return W_0_1, W_1_2, train_hist, val_hist

if __name__ == "__main__":
    W_0_1_contr, W_1_2_contr, train_hist_control, val_hist_control = train_dropout(X_train, Y_train, X_val, Y_val, iterations, False)
    W_0_1_drop, W_1_2_drop, train_hist_dropout, val_hist_dropout = train_dropout(X_train, Y_train, X_val, Y_val, iterations, True)
    best_pass_c = np.argmax(val_hist_control) + 1
    best_val_acc_c = val_hist_control[best_pass_c-1]
    best_pass_d = np.argmax(val_hist_dropout) + 1
    best_val_acc_d = val_hist_dropout[best_pass_d-1]

    print("Without Dropout:")
    print("________________")
    print("Final training round's accuracy on the training set was:")
    print(train_hist_control[-1])
    print("Final weights' accuracy on the training set was:")
    print(evaluate(X_train, Y_train, W_0_1_contr, W_1_2_contr))
    print(f"Best validation accuracy, in pass {best_pass_c}, was:")
    print(best_val_acc_c)
    print("Final validation accuracy was:")
    print(val_hist_control[-1])

    print("With Dropout:")
    print("_____________")
    print("Final training round's accuracy on the training set was:")
    print(train_hist_dropout[-1])
    print("Final weights' accuracy on the training set was:")
    print(evaluate(X_train, Y_train, W_0_1_drop, W_1_2_drop))
    print(f"Best validation accuracy, in pass {best_pass_d}, was:")
    print(best_val_acc_d)
    print("Final validation accuracy was:")
    print(val_hist_dropout[-1])



    # ---- Plot ----
    plt.plot(range(1, iterations + 1), train_hist_control, label="Training Accuracy without Dropout")
    plt.plot(range(1, iterations + 1), val_hist_control, label="Validation Accuracy without Dropout")
    plt.plot(range(1, iterations + 1), train_hist_dropout, label="Training Accuracy with Dropout")
    plt.plot(range(1, iterations + 1), val_hist_dropout, label="Validation Accuracy with Dropout")

    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)

    plt.savefig("week8/curves_part3.png")
    plt.show()