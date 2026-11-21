# Week 14 — What to Remember

This week builds an LSTM from our NumPy autograd engine. The forget, input, and output gates control what the cell keeps, writes, and exposes while it learns to predict the next character.

## Setup

Use Python 3 with NumPy and Jupyter. Download Trask's `shakespear.txt` as `week14/data/qa_oldtongue.txt`, following [DATASETS.md](../DATASETS.md#everything-else-comes-from-one-public-repo). Keep the corpus in the gitignored `data/` folder.

## Run

From the repository root:

```text
cd week14
python exploding_gradients.py
python train_lstm.py
python test_week14.py
```

The training script runs a 30-batch demo. Open `what_to_remember.ipynb` with `week14` as the working directory and use **Restart Kernel and Run All Cells**. The notebook also trains a fresh seed-0 model for 60 batches, includes the two plots and five short explanations, and ends with the holocron reading. Its source cells preserve the existing Python files verbatim and add no external imports beyond NumPy.

## Results

The current local corpus contains bAbI sentences, with 42 characters, rather than the required Shakespeare text with 62. On this file, running perplexity fell from **42.00** to **34.64** at batch 30 and **29.18** at batch 60. All four original tests and the additional notebook comparison passed during sequential execution in a fresh Python process. Interactive Jupyter Restart & Run All was not verified.

To match the assignment's 62-character reference, use the Shakespeare download linked above. The dataset was left unchanged.
