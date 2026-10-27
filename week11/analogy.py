import numpy as np

def analogy(positive, negative, k=5):
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