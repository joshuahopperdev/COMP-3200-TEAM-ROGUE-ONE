import numpy as np
from preprocess import tokenize_inputs

def softmax(x):
    x -= np.max(x)
    x = np.exp(x)
    x /= np.sum(x)
    return x

if __name__ == "__main__":
    np.random.seed(1)
    embed_size = 10
    raw_sentences, vocab, word2index, index2word, input_dataset, input_dataset_list = tokenize_inputs()
    V = len(vocab)
    embed = (np.random.rand(V, embed_size) - 0.5) * 0.1
    recurrent = np.eye(embed_size)
    start = np.zeros(embed_size)
    decoder = (np.random.rand(embed_size, V) - 0.5) * 0.1
    one_hot = np.eye(V)