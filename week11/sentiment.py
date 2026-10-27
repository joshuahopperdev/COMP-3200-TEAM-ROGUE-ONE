import numpy as np

np.random.seed(1)
alpha, iterations = 0.04, 2 # 4 x Trask's 0.01: the slope below is at most 0.25
hidden_size = 100
weights_0_1 = 0.2 * np.random.random((len(vocab), hidden_size)) - 0.1
weights_1_2 = 0.2 * np.random.random((hidden_size, 1)) - 0.1

layer_1 = sigmoid(np.sum(weights_0_1[x], axis=0)) # row-sum lookup
layer_2 = sigmoid(np.dot(layer_1, weights_1_2))
layer_2_delta = layer_2 - y
layer_1_delta = layer_2_delta.dot(weights_1_2.T)
layer_1_delta *= layer_1 * (1 - layer_1) # sigmoid's slope
weights_0_1[x] -= layer_1_delta * alpha # only rows for x
weights_1_2 -= np.outer(layer_1, layer_2_delta) * alpha