
def open_files():
    with open("data/qa1_train.txt", "r", encoding="utf-8") as f:
        raw_sentences = f.read().splitlines()

    # Inefficient: reads all 3000, but only returns 1010.
    # Claude suggests islice, for iterator slicing, to avoid this.
    return raw_sentences[:1010]

def tokenize_inputs(output_as_list = False):
    # capture the sentences (filters out newlines)
    raw_sentences = open_files()

    # Tokenize each sentence into a set of all words (lowercase) that appear in it,
    # excluding the first two characters.
    tokens = list(map(lambda s: set(s[2:].lower().split(" ")), raw_sentences))

    # If we want our final output as a list, make this a list instead so we can
    # reconstitute it later. Still generate the set: we use it to make vocab()
    # (though in this case we could just use tokens_list without much harm)
    if output_as_list==True:
        tokens_list = list(map(lambda s: list(s[1:].lower().split(" ")), raw_sentences))

    # list all words that appear (once each, because this is a set)

    # note: some words appear in multiple forms, e.g. "bathroom." and "\tbathroom\t1"
    # and various such. I asked, we're not supposed to funnel all those variations into
    # the same spot, we count them as distinct.
    vocab = sorted({word for sent in tokens for word in sent if len(word) > 0})


    # create dictionaries in both directions so we can convert word lists to indices
    # and indices to word lists. Only word2index is required, but index2word is really
    # useful for testing.
    word2index = {w: i for i, w in enumerate(vocab)}
    index2word = {i: w for i, w in enumerate(vocab)}

    # the list of all sentences' sets of words, each set in a random order
    # (note: because a lot of our numbers here are small, often it will be in
    # alphabetical order because hash(n)=n for small n and we sort the vocab)
    if output_as_list==False:
        input_dataset = [list({word2index[w] for w in sent if w in word2index}) for sent in tokens]
    # the list of all sentences' lists of words, each list in its original order
    if output_as_list==True:
        input_dataset = [[word2index[w] for w in sent if w in word2index] for sent in tokens_list]


    return raw_sentences, vocab, word2index, index2word, input_dataset


# given an array of strings, convert it into an array of the corresponding
# indices in vocab
def words2indices(sentence, word2index):
    return list(word2index[word] for word in sentence)

# given an array of indices, convert it into an array of the corresponding
# strings in vocab
def indices2words(sentence, index2word):
    return list(index2word[index] for index in sentence)




if __name__ == "__main__":
    # tokenize everything
    raw_sentences, vocab, word2index, index2word, input_dataset = tokenize_inputs(output_as_list=False)
    _, _, _, _, input_dataset_list = tokenize_inputs(output_as_list=True)

    # print the first four sentences and various forms of them for verification
    for i, term in zip(range(7), ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh"]):
        print(f"{term} raw sentence is:\n{raw_sentences[i]}\nAfter processing:\n")
        print(f"Corresponding indices as a set are {input_dataset[i]}, which correspond to the words {indices2words(input_dataset[i], index2word)}")    
        print(f"Corresponding indices as a list are {input_dataset_list[i]}, which correspond to the words {indices2words(input_dataset_list[i], index2word)}\n\n")    
