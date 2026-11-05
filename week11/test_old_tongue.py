import numpy as np

from preprocess import tokenize_inputs, open_files, regularize_labels
from sentiment import word_to_index, sentiment
from forms import sigmoid
from similarity import similar

def test_preprocessing():
    vocab = tokenize_inputs()[3]
    vocab_size = len(vocab)
    assert 70000 <= vocab_size <= 80000, f"Vocabulary size should be in the range [70,000, 80,000], got {vocab_size}"
    word2index = word_to_index(vocab)
    assert vocab[word2index['the']], f"word2index['the'] should return a valid integer index, got {word2index['the']}"

def test_lookup():
    vocab, hidden_size = tokenize_inputs()[3], 100

    # copy weight initialization from sentiment.py
    weights_0_1 = 0.2 * np.random.random((len(vocab), hidden_size)) - 0.1
    x = [2, 5, 9]
    assert np.all(weights_0_1[x].sum(axis=0) == (weights_0_1[2] + weights_0_1[5] + weights_0_1[9])), (
        f"Bundled indices should produce the same sum as split indices, got {weights_0_1[x].sum(axis=0)},"
        f" expected {weights_0_1[2] + weights_0_1[5] + weights_0_1[9]}.")

def test_sentiment():
    labels, vocab, input_dataset = tokenize_inputs()[2:]
    weights_0_1, weights_1_2, _, _ = sentiment(vocab, labels, input_dataset, 100, 0.04, 4, eval_test_size=24700, verbose=False)
    test_correct = 0
    test_start = len(input_dataset) - 100

    for i in range(test_start, len(input_dataset)):
        x, y = input_dataset[i], labels[i]

        layer_1 = sigmoid(np.sum(weights_0_1[x], axis=0))
        layer_2 = sigmoid(np.dot(layer_1, weights_1_2))

        prediction = layer_2.item()
        test_correct += int((prediction >= 0.5) == y)

    test_accuracy = test_correct / 100
    assert test_accuracy > 0.6, f"Test accuracy should be comfortably above chance, expected > 0.6, got {test_accuracy}"

def test_similarity():
    _, _, labels, vocab, input_dataset = tokenize_inputs()
    weights_0_1, _, word2index, _ = sentiment(vocab, labels, input_dataset, 100, 0.04, 2, eval_test_size=24700, verbose=False)

    words = ["beautiful", "terrible", "actor"]
    for w in words:
        common_scores = similar(weights_0_1, word2index, target=w)
        first_word, first_dist = common_scores[0]
        assert first_word == w and first_dist == 0, f"A word should be closest to itself, expected ({w}, 0), got ({common_scores[0]})"

def test_neg_sampling_shape():
    _, _, _, vocab, input_dataset = tokenize_inputs()
    concatenated = np.concatenate(input_dataset)

    np.random.seed(1)
    np.random.shuffle(input_dataset)  # Trask shuffles the reviews once
    hidden_size, window, neg = 50, 2, 5
    weights_0_1 = (np.random.rand(len(vocab), hidden_size) - 0.5) * 0.2
    weights_1_2 = np.random.rand(len(vocab), hidden_size) * 0  # zeros
    layer_2_target = np.zeros(neg + 1); layer_2_target[0] = 1
    review = input_dataset[42]

    for t in range(len(review)):
        target_samples = [review[t]] + list(
            concatenated[(np.random.rand(neg) * len(concatenated)).astype('int')])
        left = review[max(0, t - window): t]
        right = review[t + 1: min(len(review), t + window)]
        layer_1 = np.mean(weights_0_1[left + right], axis=0)  # (hidden,)
        layer_2 = sigmoid(layer_1.dot(weights_1_2[target_samples].T))  # (6,)
        assert layer_2.shape == (neg+1,), f"Layer 2 has shape (neg+1,), expected {(neg+1,)}, got {layer_2.shape}"
        assert np.all(layer_2_target == np.array([1, 0, 0, 0, 0, 0])), f"Layer 2 target must be [1, 0, 0, 0, 0, 0], got {layer_2_target}"

# Copied from other tests
if __name__ == "__main__":
    tests = [name for name in dir() if name.startswith("test_")]
    for test_name in sorted(tests):
        test_func = globals()[test_name]
        try:
            test_func()
            print(f" PASS: {test_name}")
        except AssertionError as e:
            print(f" FAIL (AssertionError): {test_name} -- {e}")
