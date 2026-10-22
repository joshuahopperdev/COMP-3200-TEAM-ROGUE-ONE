import numpy as np
from data import load_mnist, one_hot
from forms import tanh, tanh_deriv, softmax
from conv import cnn_forward

# Load MNIST dataset as 3 sets: 1000-long training set, 1000-long validation set,
# and unknown-length test set.
(X_train, y_train), (X_test, y_test) = load_mnist()
X_val, y_val = X_train[1000:2000], one_hot(y_train[1000:2000])
X_train, y_train = X_train[:1000], one_hot(y_train[:1000])
X_test, y_test = X_test, one_hot(y_test)

alpha, epochs, batch_size = (0.02, 300, 128)



def train_step(layer_0, Y_batch, kernels, W_1_2, mask, alpha):
    layer_1, layer_2, slope = cnn_forward(layer_0, kernels, W_1_2, use_mask = True, generate_mask = False, mask = mask)

    
def evaluate(layer_0, Y, kernels, W_1_2):
    _, layer_2, _ = cnn_forward(layer_0, kernels, W_1_2, use_mask=False, generate_mask = False)
    return np.mean(layer_2 = Y)


def cnn_full_train(X_train, y_train, X_val, y_val, alpha, epochs, batch_size):
    np.random.seed(1)
    kernels = 0.02 * np.random.random((9, 16)) - 0.01
    W_1_2 = 0.2 * np.random.random((10816, 10)) - 0.1

    for ep in range(epochs):
        for i in range(len(X_train) // batch_size):
            start, end = i * batch_size, (i+1) * batch_size
            layer_0 = X_train[start:end]
            y_batch = y_train[start:end]
            mask = np.random.randint(2, size=(len(layer_0), 10816))

            train_step(layer_0, y_batch, kernels, W_1_2, mask, alpha)

        if ep % 10 == 0:
            val_acc = evaluate(X_val, y_val, kernels, W_1_2)
            print(f"Epoch {ep}: Validation accuracy: {val_acc:.4f}")

    return kernels, W_1_2
        



    

#write training function:
#>loop over batches
#>>build train_step:
#>>>write mask
#>>>use part 2's forward pass, but with mask input
#>>>update kernels
#>>>update w_1_2
#>>>return k_update
#>>print no-dropout val acc every 10
#>score once on final weights

