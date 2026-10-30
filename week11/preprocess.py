def open_files():
    with open("data/reviews.txt", "r", encoding="utf-8") as f:
        raw_reviews = f.read().splitlines()

    with open("data/labels.txt", "r", encoding="utf-8") as f:
        raw_labels = f.read().splitlines()

    return raw_reviews, raw_reviews

def tokenize_inputs():
    raw_reviews, raw_labels = open_files()

    tokens = list(map(lambda s: set(s.split(" ")), raw_reviews))

    vocab = set()
    for sent in tokens:
        for word in sent:
            if len(word) > 0: 
                vocab.add(word)
    vocab = sorted(vocab) # sorted: a set's order changes every run

    word2index = {w: i for i, w in enumerate(vocab)}
    input_dataset = [list({word2index[w] for w in sent if w in word2index}) for sent in tokens]

    new_labels = regularize_labels(raw_labels)

    return raw_reviews, raw_labels, new_labels, vocab, input_dataset

def regularize_labels(raw_labels):
    new_labels = []
    for label in raw_labels:
        if label.strip().lower() == "positive":
            new_labels.append(1)
        else:
            new_labels.append(0)
    return new_labels

if __name__ == "__main__":
    raw_reviews, raw_labels, new_labels, vocab, input_dataset = tokenize_inputs()

    print(f"Reviews loaded: {len(raw_reviews)}")
    print(f"Labels loaded: {len(new_labels)}")
    print(f"Unique words: {len(vocab)}")

    print(f"\n")

    print("First review:", raw_reviews[0])
    print("First label:", new_labels[0])