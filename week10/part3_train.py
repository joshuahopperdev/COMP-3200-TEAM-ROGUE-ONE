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
    



def cnn_full_train():
    pass
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

