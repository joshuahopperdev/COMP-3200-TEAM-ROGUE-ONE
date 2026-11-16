# Week 13 — The Forge

This week builds a small NumPy autograd engine and uses it to train three models on four inputs. Each model learns to predict the second input bit.

## Setup

Use Python 3 with NumPy, Matplotlib, and Jupyter. The training data is defined in the scripts.

## Run

From the repository root:

```text
cd week13
python train_mlp.py
python train_deep.py
```

`train_mlp.py` runs the Sequential and bare MLPs. `train_deep.py` runs the network with tanh and sigmoid activations.

Open `forge.ipynb` in Jupyter with `week13` as the working directory. Use **Restart Kernel and Run All Cells** to reproduce the three loss curves.
