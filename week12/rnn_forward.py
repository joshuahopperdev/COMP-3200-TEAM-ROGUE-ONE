import numpy as np
from preprocess import tokenize_inputs, words2indices

def softmax(x):
    x -= np.max(x)
    x = np.exp(x)
    x /= np.sum(x)
    return x

def predict(sent, start, decoder, recurrent, embed):
    hiddens = [start]
    preds = [[]]
    loss = 0.0

    for t in range(len(sent)):
        h_prev = hiddens[-1]
        pred = softmax(h_prev.dot(decoder))
        loss += -np.log(pred[sent[t]])
        h_new = h_prev.dot(recurrent) + embed[sent[t]]
        hiddens.append(h_new)
        preds.append(pred)
    return hiddens, preds, loss


def perplexity(loss, length):
    return np.exp(loss/length)

if __name__ == "__main__":
    np.random.seed(1)
    embed_size = 10
    raw_sentences, vocab, word2index, index2word, input_dataset = tokenize_inputs(output_as_list = True)
    V = len(vocab)
    embed = (np.random.rand(V, embed_size) - 0.5) * 0.1
    recurrent = np.eye(embed_size)
    start = np.zeros(embed_size)
    decoder = (np.random.rand(embed_size, V) - 0.5) * 0.1
    one_hot = np.eye(V)
    hiddens, preds, loss = predict(input_dataset[0][1:], start, decoder, recurrent, embed)
    print(perplexity(loss, len(input_dataset[0][1:])))