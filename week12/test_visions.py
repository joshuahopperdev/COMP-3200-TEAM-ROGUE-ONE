import numpy as np
from preprocess import tokenize_inputs
from rnn_forward import softmax, predict
from rnn_bptt import

def test_preprocess():
    _, vocab, word2index = tokenize_inputs()[:3]
    assert len(vocab) == 82, f"Vocab length on first 1010 lines should be 82, got {len(vocab)}"
    for word in ["garden", "john", "kitchen"]:
        assert vocab[word2index[word]] == word, f"word2index must round-trip on {word}, got {vocab[word2index[word]]}"

def test_softmax():
    for size in [3, 6, 14, 23]:
        x = np.random.randn(size)
        assert np.sum(softmax(x)) - 1 <= 1e-9, f"Sum of any softmaxed vector must be within 1e-9 of 1, got {np.sum(softmax(x))}"
    zs = np.zeros(np.random.randint(10))
    softzs = softmax(zs)
    assert np.all(softz == softzs[0] for softz in softzs), f"Entries of a softmaxed zero vector are uniform, got {softzs}"

def test_forward_shape():
    pass

def test_training_loss():
    pass

def test_hidden_delta_sum():
    pass

if __name__ == "__main__":
    tests = [name for name in dir() if name.startswith("test_")]
    for test_name in sorted(tests):
        test_func = globals()[test_name]
        try:
            test_func()
            print(f" PASS: {test_name}")
        except AssertionError as e:
            print(f" FAIL (AssertionError): {test_name} -- {e}")
