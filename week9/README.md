# Week 9: Forms

This week builds on the MNIST network from Week 8. We implement activation functions, demonstrate why hidden layers need a nonlinearity, and compare hidden Forms. `forms.ipynb` preserves all six original Week 9 Python files verbatim, with explanations between parts and separate cells for notebook execution.

## Part 1: Build the Forms - Josiah

`forms.py` implements sigmoid, tanh, ReLU, their hidden-layer derivatives, and stable row-wise softmax. The derivatives reuse the activation output, avoiding another forward computation. Softmax subtracts the row maximum before exponentiating, then normalizes each row. The original `forms_plot.png` shows the three hidden activation shapes and their ranges.

## Part 2: Prove You Need a Form - Oliver

`part2_nonlinearity.py` shows that `(X @ W1) @ W2` equals `X @ (W1 @ W2)` up to floating-point error. ReLU breaks this equality. With seed 1, the best linear fit still has a residual norm of approximately 0.08642, so one matrix cannot reproduce even all five example outputs.

## Part 3: The Right Form per Layer - Nathanael

`part3_upgrade.py` uses tanh hidden units and softmax output with a 784 -> 100 -> 10 network. Settings are seed 1, learning rate 0.02, 300 epochs, and batches of 100. The slope is computed before dropout, both deltas precede weight updates, and the backward mask uses `mask * 2`.

The existing recorded result is **87.80% final validation accuracy** and **86.61% final test accuracy**. This is 3.6 percentage points above Week 8's 84.2% final batched validation result, but the hidden width also increased from 40 to 100. The gain cannot be credited to the activation alone. The original validation curve is `upgrade_plot.png`; current rerun metrics appear in notebook output.

## Part 4: Compare the Forms - Caleb

The notebook trains with the original routines and reports final validation accuracy as a text table. It uses no additional plot for the table.

| Hidden Form | Original recorded final validation accuracy |
|---|---:|
| tanh | 87.6% |
| ReLU | 88.0% |
| sigmoid | 84.3% |

The existing `compare_plot.png` is the bar chart. Sigmoid trails tanh by 3.3 percentage points in these recorded runs. Its smaller derivative and positive-only output can influence optimization at the same learning rate. However, the original Part 4 code also uses different initializations and different dropout draws, so these results do not isolate activation alone. Rerun results are printed in the notebook separately from the original recorded table.

## Part 5: Unit Testing

The original `test_forms.py` has five discovered tests. It is preserved in the notebook, but its test-set accuracy comparison is not called there. Instead, the notebook runs the four original mathematical checks and three supplemental checks:

1. Stable softmax distribution.
2. Activation ranges.
3. ReLU and its slope.
4. Linear-layer collapse and nonlinearity.
5. Finite-difference derivatives for tanh and sigmoid.
6. A one-step backpropagation check, including both updated weight matrices.
7. A 30-epoch tanh-versus-sigmoid validation comparison with matched seed and starting weights.

The notebook supplements the checks without modifying the standalone test file. Running `python test_forms.py` separately invokes its original five tests, including extra test-set evaluation.

## Part 6: Holocron Reading

The notebook ends with a 150-250 word response connecting the measured results, their limitations, and Korrith's warning against treating one Form as universal.

## Running the notebook

Open `week9/forms.ipynb` with Python 3, NumPy, Matplotlib, and Jupyter installed. Keep the original `.py` files and three image files beside it. The setup finds `week9/` from either the repository root or the notebook directory and uses the repository's `mnist.npz` when available. Otherwise, the original loader downloads it. Pixels are normalized once.

Use Restart Kernel and Run All Cells. Setup assigns a non-main module name, so the unchanged script-only blocks remain visible but do not execute. Separate cells call the training routines, score the full test set once in Part 3, and use validation only in Part 4 and the supplemental comparison check. Repeating individual training cells without restarting can reuse mutated weights.

Only these three existing images are referenced: `forms_plot.png`, `upgrade_plot.png`, and `compare_plot.png`. The notebook does not create, save, overwrite, or embed new image data. The existing `compare_curves.png` is unused and is left untouched. Original images describe the earlier recorded runs and are not regenerated when the notebook runs.

## Preserved-source assignment gaps

To preserve the requested original code, assembly leaves these implementation gaps visible:

- Parts 3 and 4 broadcast one hidden-unit mask across each batch instead of drawing a separate mask per example.
- Their loops drop incomplete batches, although all 1,000 rows are used with the current batch size of 100.
- Part 4 uses different initializations and does not reset the seed for each Form.
- Original script-only Part 4 and comparison-test code evaluate the test set again. Notebook execution skips those paths.
- The standalone test file still needs the derivative and one-step backpropagation tests and a validation-only comparison. Some original assertions also check less than their stated requirements.

These gaps need team changes to the original Python files before claiming full assignment compliance. Notebook checks do not establish PR review or merge status.

## Assembly verification

All notebook code cells were executed sequentially in a fresh Python process. The rerun reproduced every recorded metric above, and all seven notebook checks passed. All six original-source cells were checked against their files, and the notebook references exactly three existing images. Interactive Jupyter Restart & Run All has not been verified in this environment.
