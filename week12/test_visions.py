import numpy as np
from preprocess import tokenize_inputs
from rnn_forward import softmax, predict, perplexity
from rnn_bptt import RNNW, bptt, train_step

def test_preprocess():
    _, vocab, word2index = tokenize_inputs()[:3]
    assert len(vocab) == 82, f"Vocab length on first 1010 lines should be 82, got {len(vocab)}"
    for word in ["garden.", "john", "kitchen."]:
        assert vocab[word2index[word]] == word, f"word2index must round-trip on {word}, got {vocab[word2index[word]]}"

def test_softmax():
    for size in [3, 6, 14, 23]:
        x = np.random.randn(size)
        assert np.sum(softmax(x)) - 1 <= 1e-9, f"Sum of any softmaxed vector must be within 1e-9 of 1, got {np.sum(softmax(x))}"
    zs = np.zeros(np.random.randint(10))
    softzs = softmax(zs)
    assert np.all(softz == softzs[0] for softz in softzs), f"Entries of a softmaxed zero vector are uniform, got {softzs}"

def test_forward_shape():
    np.random.seed(1)
    vocab, _, _, tokens = tokenize_inputs()[1:]
    weights = RNNW(10, len(vocab))
    sent = tokens[0][1:]
    layers, loss = predict(sent,
                           weights.start,
                           weights.decoder,
                           weights.recurrent,
                           weights.embed)
    assert len(layers) == len(sent) + 1, f"Returned layers must equal len(sent) + 1 ({len(sent) + 1}), got {len(layers)}"
    assert 80 <= perplexity(loss, len(sent)) <= 84, f"Perplexity must be in the range [80, 84], got {perplexity(loss, len(sent))}"

# when we get a training loop from part 4 i think we'll replace mine with that one
def test_training_loss():
    np.random.seed(1)
    vocab, _, _, data = tokenize_inputs()[1:]
    sent = data[0][1:]
    S, V = len(sent), len(vocab)
    alpha, one_hot = 0.005, np.eye(V)
    weights = RNNW(3, V)
    old_perplexity, _ = train_step(
        sent, weights.start, weights.decoder, weights.recurrent, weights.embed, alpha, one_hot
    )
    for _ in range(100):
        new_perplexity, _ = train_step(
            sent, weights.start, weights.decoder, weights.recurrent, weights.embed, alpha, one_hot
        )
        assert new_perplexity <= old_perplexity, f"Perplexity must decrease over time, got new_perplexity={new_perplexity}, old_perplexity={old_perplexity}"
        old_perplexity = new_perplexity

def test_hidden_delta_sum():
    np.random.seed(1)
    vocab, _, _, data = tokenize_inputs()[1:]
    sent = [sent for sent in data if 3 <= len(sent) <= 7][0]
    one_hot = np.eye(len(vocab))
    weights = RNNW(3, len(vocab))
    layers, _ = predict(sent, weights.start, weights.decoder, weights.recurrent, weights.embed)
    layers = bptt(sent, layers, weights, one_hot)
    i = np.random.randint(1, len(layers) - 2)
    layer = layers[i]
    d_h, d_o = layer['hidden_delta'], layer['output_delta']
    h_r = layers[i+1]['hidden_delta'].dot(weights.recurrent.T)
    h_d = d_o.dot(weights.decoder.T)
    assert np.all(d_h == h_r + h_d), "Hidden delta must differ from output_delta.dot(decoder.T) alone"

if __name__ == "__main__":
    tests = [name for name in dir() if name.startswith("test_")]
    for test_name in sorted(tests):
        test_func = globals()[test_name]
        try:
            test_func()
            print(f" PASS: {test_name}")
        except AssertionError as e:
            print(f" FAIL (AssertionError): {test_name} -- {e}")
