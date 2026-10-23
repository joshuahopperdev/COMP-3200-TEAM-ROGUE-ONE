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
    """Runs a single step of batched gradient descent on a CNN. 
    
    parameters:

    layer_0: a batch_size x 784 numpy array of batch_size 28x28 
    images; reshaped in the first line into batch_size x 28 x 28
    
    Y_batch: a batch_size x 10 numpy array of batch_size one-hot
    encoded answers to each image; used to compute deltas
    
    kernels: a 9 x 16 numpy array of 16 3x3 kernels; used to get
    layer 1, updated at the end.
    
    W_1_2: a 10816 x 10 weight matrix, outputting 10 values that
    are then softmaxed in the final forward prop step.
    
    mask: a mask with the same shape as layer_1, zeroing random
    pixels of the kernels' output.
    
    alpha: the learning rate for both W_1_2 and kernels"""
    # reshape into flat patches
    flat_patches, _ = get_patches(layer_0.reshape(layer_0.shape[0], 28, 28))

    # record the batch size
    n = layer_0.shape[0]
    # apply the kernels
    kernel_out = flat_patches.dot(kernels)
    # apply the activation function after reshaping to have one image's 
    # full kernel output per row
    layer_1 = tanh(kernel_out.reshape(n, 676 * 16))

    # save the slope now
    slope = tanh_deriv(layer_1)

    # apply our provided mask
    layer_1 *= mask * 2

    # compute layer 2 with W_1_2 and apply softmax
    layer_2 = softmax(layer_1.dot(W_1_2))

    # compute deltas
    delta_2 = (layer_2 - Y_batch) / layer_0.shape[0] 
    delta_1 = delta_2.dot(W_1_2.T) * slope 
    # apply mask to delta_1
    delta_1 *= mask * 2

    # update W_1_2
    W_1_2 -= alpha * layer_1.T.dot(delta_2)

    # adjust delta_1's shape to match our kernel output
    delta_1_reshaped = delta_1.reshape(kernel_out.shape) # (batch*676, 16)
    # dot delta_1 with the input to get the kernel update
    k_update = flat_patches.T.dot(delta_1_reshaped) # (9, 16)
    # update the kernels
    kernels -= alpha * k_update

    # return that kernel update (before alpha)
    return k_update

    
def evaluate(layer_0, Y, kernels, W_1_2):
    # find the final predictions of a forward pass of layer_0 with these kernels and W_1_2
    _, layer_2, _ = cnn_forward(layer_0.reshape(layer_0.shape[0], 28, 28), kernels, W_1_2, use_mask=False, generate_mask = False)
    # adjust those predictions to have the index of the largest value rather than being a probability spread
    preds = np.argmax(layer_2, axis = 1)
    # adjust Y to have the index of the largest value rather than being one-hot encoded
    trues = np.argmax(Y, axis = 1)
    # return the fraction of the time that they match
    return np.mean(preds == trues)


def cnn_full_train(X_train, y_train, X_val, y_val, alpha, epochs, batch_size):
    # initialize random kernels and weights
    np.random.seed(1)
    kernels = 0.02 * np.random.random((9, 16)) - 0.01
    W_1_2 = 0.2 * np.random.random((10816, 10)) - 0.1

    # for each epoch...
    for ep in range(epochs):

        # for each batch in the epoch...
        for i in range(len(X_train) // batch_size):
            # find the edges of that batch...
            start, end = i * batch_size, (i+1) * batch_size
            layer_0 = X_train[start:end]
            y_batch = y_train[start:end]
            # define a mask...
            mask = np.random.randint(2, size=(len(layer_0), 10816))
            # and run the training
            train_step(layer_0, y_batch, kernels, W_1_2, mask, alpha)
        # Every tenth epoch, print our accuracy.
        if ep % 10 == 0:
            val_acc = evaluate(X_val, y_val, kernels, W_1_2)
            print(f"Epoch {ep}: Validation accuracy: {val_acc:.4f}")

    return kernels, W_1_2
        


if __name__ == "__main__":
    cnn_full_train(X_train, y_train, X_val, y_val, alpha, epochs, batch_size)

    

