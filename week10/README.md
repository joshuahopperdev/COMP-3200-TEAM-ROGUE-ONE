# Week 10: The Watchers

This week builds on the MNIST network from Week 9. We extract local image patches, apply shared convolution kernels, and train the Watchers to recognize digits. `watchers.ipynb` preserves all six original Week 10 Python files verbatim, with explanations between parts and separate cells for notebook execution.

## Part 1: Slice the Image into Patches

`conv.py` implements `get_patches` for 3 x 3 neighborhoods with stride 1 and no padding. A 28 x 28 image gives 26 x 26 = 676 patches. The intermediate shape is `(batch, 676, 3, 3)`, and the returned matrix is `(batch * 676, 9)`. Rows run image by image, with each image's positions ordered left to right and then top to bottom. The final pixel row and column are included.

## Part 2: The Convolution Forward Pass

The same `(9, 16)` kernel matrix multiplies every patch. The responses reshape into `(batch, 10816)` before tanh. Its slope is saved before dropout, which uses a separate mask per example and `mask * 2`. The dense `(10816, 10)` classifier feeds stable softmax. Evaluation disables dropout and mask generation.

## Part 3: Backprop & Train

`part3_train.py` uses seed 1, learning rate 0.02, 300 epochs, and batches of 128. Both deltas precede weight updates, and the backward mask uses `mask * 2`. Reshaping the hidden delta back to `(batch * 676, 16)` lets `flat_patches.T.dot(delta_1_reshaped)` accumulate the shared kernel gradient. The batch average is applied once through the output delta.

The measured notebook result is **85.50% final validation accuracy** and **86.37% final test accuracy**. The original loop drops the final 104 training rows, so these numbers describe 896 training images per epoch. The notebook scores all 10,000 test images once on the final weights, using evaluation batches to limit memory use. Training updates are unchanged.

## Part 4: Read the Stolen Eye

`part4_visualize.py` renders the 16 learned kernels as a 4 x 4 grayscale grid. The existing `kernels.png` is linked inline. Row 3, column 1 resembles a horizontal edge; row 2, column 3 suggests a vertical edge. Several tiles show corner-like or diagonal contrasts, while others have no clear pattern. The notebook describes the saved image without assigning a meaning to every tile.

The notebook reports first-layer weights, total weights, and final test accuracy as a text table. It uses no additional plot for the table.

| Network | First-layer weights | Total weights | Final test accuracy |
|---|---:|---:|---:|
| Week 9 dense, 784 -> 100 -> 10 | 78,400 | 79,400 | 86.61% |
| Week 10 convolution, 16 shared 3 x 3 kernels | 144 | 108,304 | 86.37% |

Neither network uses biases. The convolution's 144 first-layer weights are shared across positions, but the classifier adds 108,160 weights. Its total exceeds the dense network's 79,400. The measured test result is 0.24 percentage points below Week 9's recorded result. Architecture, batching, dropout masks, and training rows also differ, so this comparison does not isolate weight sharing or establish better generalization.

## Part 5: Unit Testing

The original `test_watchers.py` has five discovered tests. It is preserved in the notebook, and the notebook calls all five original tests plus two supplemental checks:

1. Patch shapes, including `(100, 28, 28)` to `(67600, 9)`.
2. First patch contents.
3. Forward hidden shape and tanh range.
4. One-step kernel-gradient shape and values.
5. A 30-epoch validation-only learning check at the original test's alpha 0.2.
6. Deterministic patch contents and ordering across image boundaries.
7. Exact updates of both weight matrices, a finite-difference kernel check with a fixed mask, and evaluation-batch agreement.

The notebook supplements the checks without modifying the standalone test file. Running `python test_watchers.py` separately invokes its original five tests. Its learning recipe differs from the full Part 3 run and leaves the test set alone. The recorded notebook execution passed **7/7 checks**.

## Part 6: Holocron Reading

The notebook ends with a 228-word response connecting the measured accuracy, kernel observations, weight counts, and limitations to Korrith's warning about the purpose of recognition. `holocron_reading.md` contains the same final reading as a separate note.

## Running the notebook

Open `week10/watchers.ipynb` with Python 3, NumPy, Matplotlib, and Jupyter installed. Keep the original `.py` files and `kernels.png` beside it. The setup finds `week10/` from either the repository root or the notebook directory and uses the repository's `mnist.npz` when available. Otherwise, the original loader downloads it. Pixels are normalized once.

Use Restart Kernel and Run All Cells. Setup assigns a non-main module name, so the unchanged script-only blocks remain visible but do not execute. Separate cells call the training routine, score the full test set once in Part 3, and run the tests with validation only. Evaluation is batched at 128 examples, including remainders, to avoid materializing the full test set's patches at once. The notebook applies this evaluation wrapper to its own namespace and the imported training module. Repeating individual cells without restarting can reuse notebook state; the evaluation-wrapper cell in particular should be run only once per fresh session.

Only the existing `kernels.png` is referenced. The notebook does not create, save, overwrite, or embed new image data. The original image is not regenerated when the notebook runs, and its training provenance is not stored with the PNG. The visual reading therefore describes the saved artifact separately from the measured rerun.

## Preserved-source assignment gaps

To preserve the original code, assembly leaves these implementation gaps visible:

- Part 3 drops the last 104 images each epoch instead of training on the required eighth, shorter batch. It uses 896 of the selected 1,000 training rows.
- The standalone Part 3 main block returns final validation accuracy without printing it and does not score the test set. The notebook supplies the single final test evaluation.
- The standalone visualization script retrains independently, and the saved PNG has no checkpoint metadata connecting it to this notebook's measured run.
- The original patch-content test uses random data rather than the requested hand-built image. The notebook adds the deterministic check without modifying that file.
- The original one-step test checks the returned kernel gradient but does not directly verify both updated matrices. The notebook adds that verification.
- The original patch-shape test includes a large `(50, 256, 256)` case, so the standalone suite can use substantial memory despite the assignment's request for fast tests.

These gaps need team review before claiming full assignment compliance. Notebook checks do not establish PR review or merge status.

## Assembly verification

All notebook code cells were executed sequentially in a fresh Python process. The full 300-epoch run produced the metrics above, and 7 of seven notebook checks passed. All six original-source cells were checked against their files, and the notebook references exactly one existing image. The final holocron reading contains 228 words and matches the separate note. Interactive Jupyter Restart & Run All could not be verified because kernel startup did not complete in this environment.
