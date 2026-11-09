# Week 12 — Visions

This week trains a NumPy recurrent neural network to predict the next word in bAbI sentences. The model trains on the first 1,000 lines. The demo uses three statements from the next ten lines to show where the model learns a sentence pattern and where its guesses miss.

## Setup

Use Python with NumPy and Matplotlib. Use Jupyter to open and run the notebook.

Download the [bAbI single-supporting-fact training file](https://raw.githubusercontent.com/iamtrask/Grokking-Deep-Learning/master/tasksv11/en/qa1_single-supporting-fact_train.txt) and save it as `week12/data/qa1_train.txt`. The `data/` folder is ignored by Git.

## Run

Run these commands from the repository root.

```text
cd week12
python test_visions.py
python train.py
python generate.py
```

`train.py` regenerates `trained_weights.npz` and `perplexity_log.csv`. If you already have the saved weights, you can run the demo directly.

Open `visions.ipynb` with `week12` as the working directory. Use **Restart & Run All** to reproduce the training plot, prediction tables, and test output.
