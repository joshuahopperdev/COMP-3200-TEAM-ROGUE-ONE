# Week 5: The Cybernetic Heresy - The Trial of Reflection

This week builds on the previous gradient descent work by introducing a hidden layer and backpropagation. The goal is to train a small neural network to recognize a pattern that a single layer cannot learn.

## Part 1: Single Layer Fails

# Caleb

This part shows why a single layer is not enough for the sensing pattern. The model's error decreases during training, but it does not reach zero because the pattern cannot be represented with one weighted sum.

## Part 2: Forward Propagation

# Nathanael

A hidden layer with four units is added to the network. The first weights connect the three inputs to the hidden layer, and the second weights connect the hidden layer to the final output.

The hidden layer uses ReLU before passing its values to the output layer.

```py
def relu(x):
    return x * (x > 0)

def forward(layer_0, weights_0_1, weights_1_2):
    layer_1 = relu(layer_0 @ weights_0_1)
    layer_2 = layer_1 @ weights_1_2
    return layer_1, layer_2
```

## Part 3: One Backpropagation Step

# Nathanael

This part introduces backpropagation by calculating the output error and passing it backward through the network. The weights are then adjusted based on that error.

The ReLU derivative is used so inactive hidden units do not contribute to the update.

```py
def relu2deriv(y):
    return y > 0
```

## Part 4: Full Training Loop

# Oliver, Caleb on a few final tweaks

The forward pass and backpropagation step are combined into a full training loop. The network trains across all four sensing inputs for multiple epochs and learns weights that produce the correct strike or hold prediction.

The learned hidden units respond to different combinations of the tells. Some units detect specific patterns in the inputs while another helps suppress the bluff case.

Different hidden layer sizes and random seeds were also tested. A hidden size of 1 failed for every seed, while larger hidden layers were more successful. Hidden sizes 8 and 16 succeeded for all three tested seeds, while smaller sizes were more dependent on the random starting weights.

## Part 5: Unit Tests

# Josiah

Unit tests were used to check the main parts of the neural network, including ReLU, the single-layer failure, forward propagation shapes, backpropagation, full training, and deterministic results.

```text
PASS: test_backprop_step
PASS: test_determinism
PASS: test_forward_shapes
PASS: test_full_train_converge
PASS: test_relu
PASS: test_single_layer_failure
```
