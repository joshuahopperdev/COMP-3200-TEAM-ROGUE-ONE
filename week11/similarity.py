import numpy as np
from collections import Counter
from forms import tanh, tanh_deriv, sigmoid, softmax
from preprocess import tokenize_inputs
from sentiment import sentiment, test_accuracy, word_to_index

def similar(weights_0_1, word2index, target='beautiful'):
    target_vec = weights_0_1[word2index[target]]
    scores = Counter()
    for word, index in word2index.items():
        diff = weights_0_1[index] - target_vec
        scores[word] = -np.sqrt(np.sum(diff * diff)) # smaller dist -> higher score
    return scores.most_common(10)

if __name__ == "__main__":
    raw_reviews, raw_labels, new_labels, vocab, input_dataset, input_dataset_list = tokenize_inputs()

    weights_0_1, weights_1_2, word2index, vocab = sentiment(vocab, new_labels, input_dataset, 100, 0.04, 2)

    common_scores = similar(weights_0_1, word2index)

    print("Similar words - beautiful")
    for score in common_scores:
        print(f"{score}")

    common_scores = similar(weights_0_1, word2index, target="terrible")
    
    print("Similar words - terrible")
    for score in common_scores:
        print(f"{score}")

    common_scores = similar(weights_0_1, word2index, target="actor")
        
    print("Similar words - actor")
    for score in common_scores:
        print(f"{score}")

    # test_accuracy(weights_0_1, weights_1_2, input_dataset, new_labels, 1000)

"""
Similar words - beautiful
('beautiful', np.float64(-0.0))
('necessary', np.float64(-0.7056959950000653))
('underrated', np.float64(-0.706256571515744))
('superbly', np.float64(-0.7098548196631785))
('paramount', np.float64(-0.7138199919200069))
('appreciated', np.float64(-0.7167918909094897))
('mine', np.float64(-0.7204286610706009))
('ralph', np.float64(-0.7210625106080841))
('quiet', np.float64(-0.7255143221837442))
('finest', np.float64(-0.7258827990691072))
Similar words - terrible
('terrible', np.float64(-0.0))
('disappointment', np.float64(-0.8301462054398466))
('boring', np.float64(-0.8451990096262897))
('annoying', np.float64(-0.8543064991206111))
('dull', np.float64(-0.8648644307489477))
('avoid', np.float64(-0.8679342268380068))
('poor', np.float64(-0.8909239014364243))
('save', np.float64(-0.902951104933264))
('poorly', np.float64(-0.9084764592093274))
('disappointing', np.float64(-0.9131441062499509))
Similar words - actor
('actor', np.float64(-0.0))
('sucker', np.float64(-0.6398902483881945))
('squashed', np.float64(-0.6431003989170067))
('cato', np.float64(-0.6473903102285469))
('bogdansker', np.float64(-0.6477239154827741))
('unwisely', np.float64(-0.6484111248557112))
('paranormal', np.float64(-0.6505696606404567))
('coster', np.float64(-0.6548103007331858))
('prairie', np.float64(-0.6584850505965923))
('shadow', np.float64(-0.6594120179464833))
"""