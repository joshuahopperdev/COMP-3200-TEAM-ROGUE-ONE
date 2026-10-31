import numpy as np
from forms import tanh, tanh_deriv, sigmoid, softmax
from preprocess import tokenize_inputs

def word_to_index(vocab):
    word2index = {w: i for i, w in enumerate(vocab)}
    return word2index

def sentiment(vocab, new_labels, input_dataset, hidden_size, alpha, iterations, seed = 1, eval_test_size = 1000, verbose = True):
    np.random.seed(seed)

    weights_0_1 = 0.2 * np.random.random((len(vocab), hidden_size)) - 0.1
    weights_1_2 = 0.2 * np.random.random((hidden_size, 1)) - 0.1

    for it in range(0, iterations):
        correct = 0

        for i in range(len(input_dataset) - eval_test_size):
            x, y = input_dataset[i], new_labels[i]

            # Forward Pass
            layer_1 = sigmoid(np.sum(weights_0_1[x], axis=0)) # row-sum lookup
            layer_2 = sigmoid(np.dot(layer_1, weights_1_2))

            # Error Tracking
            prediction = layer_2.item()
            correct += int((prediction >= 0.5) == y)

            # Delta
            layer_2_delta = layer_2 - y
            layer_1_delta = layer_2_delta.dot(weights_1_2.T)
            layer_1_delta *= layer_1 * (1 - layer_1) # sigmoid's slope

            # Weight Updates
            weights_0_1[x] -= layer_1_delta * alpha # only rows for x
            weights_1_2 -= np.outer(layer_1, layer_2_delta) * alpha

            if verbose and (i + 1) % 1000 == 0:
                print(f"Iter {it + 1}/{iterations}\t\tAccuracy: {correct / (i + 1) * 100:.4f}%")

    word2index = word_to_index(vocab)

    return weights_0_1, word2index, vocab


if __name__ == "__main__":
    raw_reviews, raw_labels, new_labels, vocab, input_dataset = tokenize_inputs()

    weights_0_1, word2index, vocab = sentiment(vocab, new_labels, input_dataset, 100, 0.04, 2)

