import numpy as np

np.random.seed(1)
np.random.seed(1)
np.random.shuffle(input_dataset) # Trask shuffles the reviews once
alpha, iterations = 0.05, 2
hidden_size, window, neg = 50, 2, 5
weights_0_1 = (np.random.rand(len(vocab), hidden_size) - 0.5) * 0.2
weights_1_2 = np.random.rand(len(vocab), hidden_size) * 0 # zeros
layer_2_target = np.zeros(neg + 1); layer_2_target[0] = 1

target_samples = [review[t]] + list(concatenated[(np.random.rand(neg) * len(concatenated)).astype('int')])
left = review[max(0, t-window): t]
right = review[t+1: min(len(review), t+window)]
layer_1 = np.mean(weights_0_1[left + right], axis=0) # (hidden,)
layer_2 = sigmoid(layer_1.dot(weights_1_2[target_samples].T)) # (6,)
layer_2_delta = layer_2 - layer_2_target
layer_1_delta = layer_2_delta.dot(weights_1_2[target_samples])
weights_0_1[left + right] -= layer_1_delta * alpha
weights_1_2[target_samples] -= np.outer(layer_2_delta, layer_1) * alpha