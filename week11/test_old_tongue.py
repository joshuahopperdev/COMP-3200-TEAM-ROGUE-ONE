import numpy as np

from preprocess import tokenize_inputs, open_files, regularize_labels
from sentiment import word_to_index, sentiment
from forms import sigmoid
from similarity import similar
#import fillblank as fb
#import analogy

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

def tokenize_some(limit = 200):
    raw_reviews, raw_labels = open_files()
    some_reviews, some_labels = raw_reviews[:limit], raw_labels[:limit]

    tokens = list(map(lambda s: set(s.split(" ")), some_reviews))
    # sorted: a set's order changes every run
    vocab = sorted({ word for sent in tokens for word in sent if len(word) > 0 })

    word2index = {w: i for i, w in enumerate(vocab)}
    input_dataset = [list({word2index[w] for w in sent if w in word2index}) for sent in tokens]
    new_labels = regularize_labels(some_labels)

    return new_labels, vocab, input_dataset

def test_sentiment():
    labels, vocab, input_dataset = tokenize_inputs()[2:] #tokenize_some(300) # use only some of the corpus for a short training run
    weights_0_1, weights_1_2, word2index, vocab = sentiment(vocab, labels, input_dataset, 100, 0.04, 4, verbose=False)
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
    weights_0_1, _, word2index, _ = sentiment(vocab, labels, input_dataset, 100, 0.04, 2, verbose=False)

    words = ["beautiful", "terrible", "actor"]
    for w in words:
        common_scores = similar(weights_0_1, word2index, target=w)
        first_word, first_dist = common_scores[0]
        assert first_word == w and first_dist == 0, f"A word should be closest to itself, expected ({w}, 0), got ({common_scores[0]})"

def test_neg_sampling_shape():
    pass

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
