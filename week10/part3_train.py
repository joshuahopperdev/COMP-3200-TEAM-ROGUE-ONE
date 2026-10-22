import numpy as np
from data import load_mnist, one_hot
from forms import tanh, tanh_deriv, softmax
from conv import get_patches, cnn_forward

# Load MNIST dataset as 3 sets: 1000-long training set, 1000-long validation set,
# and unknown-length test set.
(X_train, y_train), (X_test, y_test) = load_mnist()
X_val, y_val = X_train[1000:2000], one_hot(y_train[1000:2000])
X_train, y_train = X_train[:1000], one_hot(y_train[:1000])
X_test, y_test = X_test, one_hot(y_test)

alpha, epochs, batch_size = (0.02, 300, 128)



def train_step(layer_0, Y_batch, kernels, W_1_2, mask, alpha):

    flat_patches, _ = get_patches(layer_0.reshape(layer_0.shape[0], 28, 28))

    n = layer_0.shape[0]
    kernel_out = flat_patches.dot(kernels)

    layer_1 = tanh(kernel_out.reshape(n, 676 * 16))

    slope = tanh_deriv(layer_1)

    layer_1 *= mask * 2

    layer_2 = softmax(layer_1.dot(W_1_2))

    delta_2 = (layer_2 - Y_batch) / layer_0.shape[0] # batch average
    delta_1 = delta_2.dot(W_1_2.T) * slope # slope saved before the mask
    delta_1 *= mask * 2 # forward was * mask * 2
    W_1_2 -= alpha * layer_1.T.dot(delta_2)

    delta_1_reshaped = delta_1.reshape(kernel_out.shape) # (batch*676, 16)
    k_update = flat_patches.T.dot(delta_1_reshaped) # (9, 16)
    kernels -= alpha * k_update

    
def evaluate(layer_0, Y, kernels, W_1_2):
    _, layer_2, _ = cnn_forward(layer_0.reshape(layer_0.shape[0], 28, 28), kernels, W_1_2, use_mask=False, generate_mask = False)
    preds = np.argmax(layer_2, axis = 1)
    trues = np.argmax(Y, axis = 1)
    return np.mean(preds == trues)


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
        

cnn_full_train(X_train, y_train, X_val, y_val, alpha, epochs, batch_size)

    

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

