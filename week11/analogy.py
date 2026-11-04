import numpy as np
import math
from collections import Counter
from preprocess import tokenize_inputs
from sentiment import word_to_index
from fillblank import train_fill_blanks

def analogy(positive, negative, weights_0_1, word2index, k=5):
    q = np.zeros(weights_0_1.shape[1])
    for w in positive: q += weights_0_1[word2index[w]]
    for w in negative: q -= weights_0_1[word2index[w]]
    scores = Counter()
    for word, idx in word2index.items():
        d = weights_0_1[idx] - q
        scores[word] = -math.sqrt(np.sum(d * d))
    # exclude the input words from the result
    return [w for w, _ in scores.most_common(20)
            if w not in positive + negative][:k]

if __name__ == '__main__':
    _, _, new_labels, vocab, input_dataset = tokenize_inputs()[2:]
    concatenated = np.concatenate(input_dataset)
    word2index = word_to_index(vocab)

    np.random.seed(1)
    np.random.shuffle(input_dataset)  # Trask shuffles the reviews once
    alpha, iterations, hidden_size, window, neg = 0.05, 2, 50, 2, 5
    weights_0_1 = (np.random.rand(len(vocab), hidden_size) - 0.5) * 0.2
    weights_1_2 = np.random.rand(len(vocab), hidden_size) * 0  # zeros
    layer_2_target = np.zeros(neg + 1); layer_2_target[0] = 1
    weights_0_1, _ = train_fill_blanks(weights_0_1, weights_1_2, input_dataset, alpha, iterations, window, neg)

    analogies = analogy(
        ['terrible', 'good'],
        ['bad'],
        weights_0_1,
        word2index,
    )
    print(f"terrible + good - bad = {analogies}")
    """The results of terrible + good - bad are:
    
    """

    our_analogies = analogy(
        ['light', 'cheery'],
        ['dark'],
        weights_0_1,
        word2index,
    )
    print(f"light + cheery - dark = {our_analogies}")
    """The results of light + cheery - dark are:
    
    """
