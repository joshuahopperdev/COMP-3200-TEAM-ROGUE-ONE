import numpy as np
import csv
import time

from preprocess import tokenize_inputs
from rnn_bptt import RNNW, train_step
from rnn_forward import predict

def train(alpha, iter, TRAIN_N, seed, sentances, vocab, word2indecies, indecies2words, tokens, verbose = False):
    # Set random
    np.random.seed(seed)
    one_hot = np.eye(len(vocab))

    # build RNN weights class
    weights = RNNW(embed_size = 10, vocab_size = len(vocab))

    # for writing out to CSV
    perplexity_log = []
    
    for it in range(iter):
        sent = tokens[it % TRAIN_N][1:]
        layers, loss = predict(sent, weights.start, weights.decoder, weights.recurrent, weights.embed)

        # ---- back propagate (BPTT in Code, 3 of 3) ---- fills in the deltas
        # ---- update weights (The Weight Updates) ---- subtracts them

        perplexity, _ = train_step(sent, weights.start, weights.decoder, weights.embed, weights.recurrent, alpha, one_hot)

        if it % 1000 == 0:
            perplexity_log.append((it, perplexity))
            if verbose:
                print(f"Iteration {it}:\tPerplexity: {perplexity}")

    # Save csv
    with open("perplexity_log.csv", "w", newline = "") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "perplexity"])
        writer.writerow(perplexity_log)

    # Save weights
    np.savez("trained_weights.npz", start = weights.start, decoder = weights.decoder, recurrent = weights.recurrent, embed = weights.embed)

    return perplexity_log


if __name__ == "__main__":
    sentances, vocab, word2index, index2word, tokens = tokenize_inputs(output_as_list = True)

    train(0.001, 30000, 1000, 1, sentances, vocab, word2index, index2word, tokens, verbose = True)

