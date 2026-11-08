import numpy as np
import preprocess as prep

def bptt(sent, layers, weights, one_hot):
    """Backward Propagation Through Time
    Parameters:
        sent (in): sentence as a word2index mapping used for getting a target index
        layers (in-out): list of layers, which are a dictionary with pred (optional) and a hidden state, soon to have an output_delta and a hidden_delta
        weights (in):
    """
    # Layer 0 doesn't have a pred, so leave it out of the loop
    for layer_idx, layer in enumerate(layers[1:]):
        target = sent[layer_idx - 1]
        layer['output_delta'] = layer['pred'] - one_hot[target]
        new_hidden_delta = layer['output_delta'].dot(weights.decoder.T)

        if layer_idx == len(layers) - 1:
            layer['hidden_delta'] = new_hidden_delta
        else:
            layer['hidden_delta'] = new_hidden_delta + layers[layer_idx + 1]['hidden_delta'].dot(weights.reccurent.T)

    # Get the hidden_delta for layer 0 so the walk is complete; this must necessarily
    # be after the loop so layer 1's hidden_delta exists
    layers[0]['hidden_delta'] = layers[1]['hidden_delta'].dot(weights.recurrent.T)

    return layers # just in case in-out parameters don't work...

def update_weights(layers, weights):
    """"""
    pass

# probably overengineered, but here's a class for weights
class RNNW:
    """RNN Weights, namely start, embed, recurrent, and decoder"""
    def __init__(self, seed, embed_size, vocab_size):
        np.random.seed(seed)
        self.embed = (np.random.rand(vocab_size, embed_size) - 0.5) * 0.1
        self.recurrent = np.eye(embed_size)
        self.start = np.zeros(embed_size)
        self.decoder = (np.random.rand(embed_size, vocab_size) - 0.5) * 0.1



if __name__ == "__main__":

