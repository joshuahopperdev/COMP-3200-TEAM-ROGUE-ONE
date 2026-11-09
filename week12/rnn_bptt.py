import numpy as np
from preprocess import tokenize_inputs, words2indices
from rnn_forward import predict, perplexity

def bptt(sent, layers, weights, one_hot):
    """Backward Propagation Through Time
    Parameters:
        sent (in): sentence as a word2index mapping used for getting a target index
        layers (in-out): list of layers, which are a dictionary with pred (optional) and a hidden state, soon to have an output_delta and a hidden_delta
        weights (in): RNNW object for accessing decoder and recurrent weight matrices
        one_hot (in): used for calculating output deltas
    """
    # Layer 0 doesn't have a pred, so leave it out of the loop
    # The -1:0:-1 slice goes in reverse iterating through the range
    # [last..0) -- the last layer down to but not including the first layer
    for i, layer in enumerate(layers[-1:0:-1]):
        # i starts from 0, so we need to calculate the distance from the end
        layer_idx = len(layers) - i - 1
        target = sent[layer_idx - 1]
        layer['output_delta'] = layer['pred'] - one_hot[target]
        new_hidden_delta = layer['output_delta'].dot(weights.decoder.T)

        if layer_idx == len(layers) - 1:
            layer['hidden_delta'] = new_hidden_delta
        else:
            layer['hidden_delta'] = new_hidden_delta + layers[layer_idx + 1]['hidden_delta'].dot(weights.recurrent.T)

    # Get the hidden_delta for layer 0 so the walk is complete; this must necessarily
    # be after the loop so layer 1's hidden_delta exists
    layers[0]['hidden_delta'] = layers[1]['hidden_delta'].dot(weights.recurrent.T)

    return layers # just in case in-out parameters don't work...

# Preconditions:
# - My weights class won't be used by others
# - The random seed is already set
def train_step(sent, start, decoder, embed, recurrent, alpha, one_hot):
    # Predict
    layers, loss = predict(sent, start, decoder, embed, recurrent)

    weights = pack(embed, recurrent, start, decoder) # pack the weights into my class

    # Compare
    layers = bptt(sent, layers, weights, one_hot)

    # Learn
    weights.update(sent, layers, alpha)

    return perplexity(loss, len(sent)), weights.unmarshal()

def pack(embed, recurrent, start, decoder):
    """Pack weights defined externally
    (beyond this file) into the class"""
    return RNNW().marshal(embed, recurrent, start, decoder)

# probably overengineered, but here's a class for weights.
# it's essentially a struct for each of the four weight
# tensors with (un)marshal methods and the update method.
class RNNW:
    """RNN Weights, namely start, embed, recurrent, and decoder"""
    def __init__(self, embed_size = None, vocab_size = None):
        self.embed, self.recurrent, self.start, self.decoder = None, None, None, None

        if embed_size is not None and vocab_size is not None:
            self.embed = (np.random.rand(vocab_size, embed_size) - 0.5) * 0.1
            self.recurrent = np.eye(embed_size)
            self.start = np.zeros(embed_size)
            self.decoder = (np.random.rand(embed_size, vocab_size) - 0.5) * 0.1

    def marshal(self, embed, recurrent, start, decoder):
        self.embed = embed
        self.recurrent = recurrent
        self.start = start
        self.decoder = decoder
        return self

    def unmarshal(self):
        return self.embed, self.recurrent, self.start, self.decoder

    def update(self, sent, layers, alpha):
        """Update the weights given the deltas
        baked into each layer; we also have the
        sentence for getting the length and alpha"""
        scalar = alpha / len(sent)
        self.start -= layers[1]['hidden_delta'] * scalar
        for layer_idx, layer in enumerate(layers[1:]):
            self.decoder -= np.outer(layers[layer_idx]['hidden'],
                                     layer['output_delta']) * scalar
            if layer_idx + 2 < len(layers): # last state makes no guess
                grad_h = layers[layer_idx + 2]['hidden_delta']
                self.embed[sent[layer_idx]] -= grad_h * scalar
                self.recurrent -= np.outer(layers[layer_idx]['hidden'],
                                           grad_h) * scalar

if __name__ == "__main__":
    np.random.seed(1)
    vocab, word2index, _, data = tokenize_inputs()[1:]
    weights = RNNW(embed_size = 10, vocab_size = len(vocab))
    sent = data[0][1:]
    one_hot = np.eye(len(vocab))
    layers = predict(sent, weights.start, weights.decoder, weights.recurrent, weights.embed)[0]
    layers = bptt(sent, layers, weights, one_hot)

    # read the deltas
    # a random k
    k = np.random.randint(2, len(layers))
    print(f"-- Layer {k} --")
    print(f"Pred: {layers[k]['pred']}")
    print(f"Hidden (stored in previous state): {layers[k-1]['hidden']}")
    print(f"Hidden delta (stored in next state): {layers[k+1]['hidden_delta']}")
    print()

    # exploring start
    print(f"-- Start (Layer 0) --")
    print(f"Pred (should be `start`): {layers[0]['hidden']}")
    print(f"Hidden delta (in layer 1): {layers[1]['hidden_delta']}")

    # sent[i]'s effect on layers after it
    # a random i
    i = np.random.randint(len(sent))
    print(f"-- sent[{i}]'s effects on later layers --")
    print(f"sent[{i}]: {sent[i]}")
    print(f"Layer {i+1}'s state: {layers[i+1]['hidden']}")
    print(f"sent[{i}]'s gradient (found in layer {i+2}): {layers[i+2]['hidden_delta']}")
    print()
