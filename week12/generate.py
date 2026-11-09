"""Show next-word predictions on three statements past the training cut."""

from pathlib import Path

import numpy as np

from preprocess import indices2words, tokenize_inputs
from rnn_forward import predict


def next_word_tables():
    """Return Markdown tables using the vocabulary and weights from training."""
    raw_lines, _, _, index2word, tokens = tokenize_inputs(output_as_list=True)
    statement_indices = [
        index for index, line in enumerate(raw_lines[1000:1010], start=1000)
        if line.endswith(".") and "?" not in line
    ][:3]
    if len(statement_indices) != 3:
        raise ValueError("Expected three statements at corpus indices 1000 through 1009.")

    tables = []
    weights_path = Path(__file__).resolve().with_name("trained_weights.npz")
    with np.load(weights_path) as saved:
        weights = {name: saved[name] for name in ("start", "decoder", "recurrent", "embed")}

    for index in statement_indices:
        sent = tokens[index][1:]
        words = indices2words(sent, index2word)
        layers, _ = predict(sent, **weights)
        original = raw_lines[index].split(maxsplit=1)[1]
        rows = [
            f"**Corpus index {index}** ({original})",
            "",
            "| Context | True next word | Network prediction |",
            "| --- | --- | --- |",
        ]
        # Layer 1 predicts from the start state, before seeing any words.
        for position in range(1, len(words)):
            context = " ".join(words[:position])
            prediction = index2word[layers[position + 1]["pred"].argmax()]
            rows.append(f"| {context} | {words[position]} | {prediction} |")
        tables.append("\n".join(rows))

    return "\n\n".join(tables)


if __name__ == "__main__":
    print(next_word_tables())
