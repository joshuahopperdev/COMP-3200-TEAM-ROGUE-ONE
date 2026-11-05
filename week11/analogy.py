import numpy as np
import random
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
    raw_reviews, _, labels, vocab = tokenize_inputs()[:4]
    word2index = word_to_index(vocab)

    # Make a list instead of a set to preserve order and repeat words
    tokens = list(map(lambda s: list(s.split(" ")), raw_reviews))
    dataset = [[word2index[w] for w in sent if w in word2index] for sent in tokens]

    np.random.seed(1), random.seed(1)
    random.shuffle(dataset)  # Trask shuffles the reviews once
    alpha, iterations, hidden_size, window, neg = 0.05, 2, 50, 2, 5
    weights_0_1 = (np.random.rand(len(vocab), hidden_size) - 0.5) * 0.2
    weights_1_2 = np.random.rand(len(vocab), hidden_size) * 0  # zeros
    W_0_1, _ = train_fill_blanks(weights_0_1, weights_1_2, dataset, alpha, iterations, window, neg)

    analogies = analogy(
        ['terrible', 'good'],
        ['bad'],
        W_0_1,
        word2index
    )
    print(f"terrible + good - bad = {analogies}")
    """The results of terrible + good - bad are:
    1. mediocre
    2. stellar
    3. lousy
    4. pleasant
    5. fine
    """

    our_analogies = analogy(
        ['light', 'cheery'],
        ['dark'],
        W_0_1,
        word2index
    )
    print(f"light + cheery - dark = {our_analogies}")
    """The results of light + cheery - dark are:
    1. snap
    2. queue
    3. stead
    4. hinges
    5. strap
    """

    other_analogy = analogy(
        ['actor', 'woman'],
        ['man'],
        W_0_1,
        word2index
    )
    print(f"actor - man + woman = {other_analogy}")
    """The results of actor + woman - man are:
    1. actress
    2. assassin
    3. comedian
    4. englishman
    5. ordeal
    """
