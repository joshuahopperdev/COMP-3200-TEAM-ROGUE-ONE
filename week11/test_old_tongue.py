import numpy as np

from preprocess import tokenize_inputs
import sentiment
import similarity as sim
import fillblank as fb
import analogy

def test_preprocessing():
    _, _, _, vocab, word2index = tokenize_inputs()
    vocab_size = len(vocab)
    assert 70000 <= vocab_size <= 80000, f"Vocabulary size should be in the range [70,000, 80,000], got {vocab_size}"
    assert vocab[word2index['the']], f"word2index['the'] should return a valid integer index, got {word2index['the']}"

def test_lookup():
    pass

def test_sentiment():
    pass

def test_similarity():
    pass

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
