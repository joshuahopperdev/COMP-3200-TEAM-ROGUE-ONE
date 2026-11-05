import numpy as np
import random
from forms import tanh, tanh_deriv, sigmoid, softmax
from preprocess import tokenize_inputs
from similarity import similar
from sentiment import word_to_index



raw_reviews, raw_labels, new_labels, vocab, input_dataset, input_dataset_list = tokenize_inputs()


np.random.seed(1); random.seed(1)
random.shuffle(input_dataset_list) # Trask shuffles the reviews once
alpha, iterations = 0.05, 2
hidden_size, window, neg = 50, 2, 5
weights_0_1 = (np.random.rand(len(vocab), hidden_size) - 0.5) * 0.2
weights_1_2 = np.random.rand(len(vocab), hidden_size) * 0 # zeros
layer_2_target = np.zeros(neg + 1); layer_2_target[0] = 1

def train_fill_blanks(weights_0_1, weights_1_2, dataset, alpha, iterations, window, neg, reassuring = False):
    word2index = word_to_index(vocab)
    concatenated = np.concatenate(dataset)
    for it in range(iterations):
        for i, review in enumerate(dataset[:1000]):
            if reassuring:
                if i % 1000 == 0:
                    print("Iteration: ", it, "Review: ", i)
                if i % 10000 == 0:
                        print(similar(weights_0_1, word2index, "terrible"))
                        print(similar(weights_0_1, word2index, "beautiful"))
                        print(similar(weights_0_1, word2index, "actor"))
            for t in range(len(review)):
                target_samples = [review[t]] + list(concatenated[(np.random.rand(neg) * len(concatenated)).astype('int')])
                left = review[max(0, t-window): t]
                right = review[t+1: min(len(review), t+window)]
                layer_1 = np.mean(weights_0_1[left + right], axis=0) # (hidden,)
                layer_2 = sigmoid(layer_1.dot(weights_1_2[target_samples].T)) # (6,)
                layer_2_delta = layer_2 - layer_2_target
                layer_1_delta = layer_2_delta.dot(weights_1_2[target_samples])
                weights_0_1[left + right] -= layer_1_delta * alpha
                weights_1_2[target_samples] -= np.outer(layer_2_delta, layer_1) * alpha

    return weights_0_1, weights_1_2

if __name__ == "__main__":
    word2index = word_to_index(vocab)
    train_fill_blanks(weights_0_1, weights_1_2, input_dataset_list, alpha, iterations, window, neg, reassuring = True)
    print(similar(weights_0_1, word2index, "terrible"))
    print(similar(weights_0_1, word2index, "beautiful"))
    print(similar(weights_0_1, word2index, "actor"))


"""
Similar words - terrible
'terrible', np.float64(-0.0)
'horrible', np.float64(-3.012859830752236)
'lousy', np.float64(-3.0523954575916505)
'bad', np.float64(-3.497417844056076)
'pathetic', np.float64(-3.6542910918700953)
'dreadful', np.float64(-3.7396463432557874)
'brilliant', np.float64(-3.9439201029351234) 
'lame', np.float64(-3.983734633112278)
'pitiful', np.float64(-4.043469444632994) 
'great', np.float64(-4.150066657219581)

Similar words - beautiful
'beautiful', np.float64(-0.0)
'lovely', np.float64(-3.2138389035105335)
'cute', np.float64(-3.2211642468843147)
'delightful', np.float64(-3.44899033337854)
'vibrant', np.float64(-3.5090312050745918)
'repulsive', np.float64(-3.5700124507226287)
'lovable', np.float64(-3.699571624533544)
'gorgeous', np.float64(-3.7046779741705507)
'broad', np.float64(-3.7189430529233536)
'heartbreaking', np.float64(-3.7339128891802997)

Similar words - actor
'actor', np.float64(-0.0)
'actress', np.float64(-4.189155440762897)
'comedian', np.float64(-5.437025405177901)
'artwork', np.float64(-5.47018142364434)
'comedians', np.float64(-5.602007506490803)
'englishman', np.float64(-5.612872082258708)
'performer', np.float64(-5.690213916285616)
'egg', np.float64(-5.736313975973628)
'organ', np.float64(-5.746607904451557)
'dancer', np.float64(-5.769578589682124)
"""