import numpy as np
from preprocess import tokenize_inputs, words2indices


# From an array of relative log-probs, calculates the actual probabilities
def softmax(x):
    # first, without losing relative magnitudes, subtract the largest.
    # This means the largest x is 0, so we can't get any values larger than 1,
    # so we can't have anything larger than a float can be
    x -= np.max(x)

    # exponentiate
    x = np.exp(x)
    # divide by the total, bringing our final total to 1
    x /= np.sum(x)
    # return that array of probabilities
    return x

# Given the list of all words (except the first, see below for external 
# implementation), our initial bias term, our decoder, our adjustment 
# to past hidden states and our embedding, predict each token in a sequence.

# notably, this intentionally departs from assignment spec by having
# multiple parameters rather than just using global variables, and by
# returning two separate index-matched arrays for hiddens and preds.

# Note: preds[k] is the guess for sent[k-1], made after seeing
# sent[:k-1], everything up until that point. That is, preds[3]
# is the guess for what word will be sent[2], after feeding it
# start*recurrence*recurrence, the embedding*recurrence of sent[0] 
# and the embedding of sent[1].
#
# But the sentences we're feeding our predict() have the first
# word lopped off for no reason (see below, in main() and later in train.py),
# so preds[k] will be the guess for original_sentence[k], but without the 
# information of the first word. E.g. preds[3] is the guess for
# original_sentence[3], with only start*recurrence*recurrence*recurrence, 
# the embedding*recurrence of original_sentence[1] and the embedding of 
# original_sentence[2] and not the embedding*recurrence*recurrence of 
# original_sentence[0] (because we lopped it off).
def predict(sent, start, decoder, recurrent, embed):
    hiddens = [start]
    preds = [[]]
    loss = 0.0

    # for each word, sequentially:
    for t in range(len(sent)):
        # our current hidden layer is the last
        # entry in hiddens
        h_prev = hiddens[-1]
        # our prediction is the softmax of our
        # hidden layer times the decoder
        pred = softmax(h_prev.dot(decoder))
        # our loss is the log-difference between
        # our actual chance (ln 1 = 0) and our
        # predicted chance (e.g. ln 0.37 = -1,
        # total loss = 0 - -1 = 1). No need to
        # absolute value this, because softmax
        # prevents ever predicting more than 1
        loss += -np.log(pred[sent[t]])

        # our hidden layer for next time will be the
        # current hidden layer * recurrence plus the
        # embedding of the real word that we just
        # tried to predict
        h_new = h_prev.dot(recurrent) + embed[sent[t]]

        # stick 'em on! See comments above predict()
        # for where what goes
        hiddens.append(h_new)
        preds.append(pred)
    return hiddens, preds, loss

# perplexity = e^(average loss)
def perplexity(loss, length):
    return np.exp(loss/length)


if __name__ == "__main__":
    # various variables, tokenization
    np.random.seed(1)
    embed_size = 10
    raw_sentences, vocab, word2index, index2word, input_dataset = tokenize_inputs(output_as_list = True)
    V = len(vocab)
    embed = (np.random.rand(V, embed_size) - 0.5) * 0.1
    recurrent = np.eye(embed_size)
    start = np.zeros(embed_size)
    decoder = (np.random.rand(embed_size, V) - 0.5) * 0.1
    one_hot = np.eye(V)
    # why, oh why are we starting from the second word? Claude and I concluded that this
    # was somewhat unhinged and didn't have any coherent reason, and might be a duplication
    # of the first-number removal in the preprocessing, even after checking Trask's code
    # on GitHub. 
    # 
    # The assignment spec is clear, we are to use this, it gets the same
    # perplexity for the same seed as Dr. Faith does. It calls it "Trask's convention", 
    # but we can find no reason for this convention.
    hiddens, preds, loss = predict(input_dataset[0][1:], start, decoder, recurrent, embed)
    print(perplexity(loss, len(input_dataset[0][1:])))