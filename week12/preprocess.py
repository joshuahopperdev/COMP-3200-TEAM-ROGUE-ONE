import re

def open_files():
    with open("data/qa1_train.txt", "r", encoding="utf-8") as f:
        raw_sentences = f.read().splitlines()

    # Inefficient: reads all 3000, but only returns 1010.
    # Claude suggests islice, for iterator slicing, to avoid this.
    return raw_sentences[:1010]

def tokenize_inputs():
    raw_sentences = open_files()


    tokens = list(map(lambda s: set(re.sub(r'[\d\t?.]+', '', e) for e in s[1:].lower().split(" ")), raw_sentences))
    tokens_list = list(map(lambda s: list(re.sub(r'[\d\t?.]+', '', e) for e in s[1:].lower().split(" ")), raw_sentences))


    vocab = set()
    for sent in tokens:
        for word in sent:
            if len(word) > 0: 
                vocab.add(word)
    vocab = sorted(vocab) # sorted: a set's order changes every run

    word2index = {w: i for i, w in enumerate(vocab)}

    input_dataset = [list({word2index[w] for w in sent if w in word2index}) for sent in tokens]
    input_dataset_list = [list([word2index[w] for w in sent if w in word2index]) for sent in tokens_list]


    return raw_sentences, vocab, word2index, input_dataset, input_dataset_list


def words2indices(sentence, word2index):
    return list(word2index[word] for word in sentence)


if __name__ == "__main__":
    raw_sentences, vocab, word2index, input_dataset, input_dataset_list = tokenize_inputs()