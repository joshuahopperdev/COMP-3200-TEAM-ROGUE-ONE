import numpy as np

def get_patches(layer):             # layer: (batch, 28, 28)
    batch_size = layer.shape[0]
    patches = []
    for r in range(layer.shape[1] - 3 + 1):         # 0..25
        for c in range(layer.shape[2] - 3 + 1):     # 0..25
            patch = layer[:, r:r+3, c:c+3]          # (batch, 3, 3)
            # the 1 is a slot for the patch-position axis
            patches.append(patch.reshape(batch_size, 1, 3, 3))

    # concatenate fills that slot: 676 positions side by side
    expanded = np.concatenate(patches, axis=1)      # (batch, 676, 3, 3)
    es = expanded.shape

    # flatten EACH patch to 9 numbers, one row per (image, position)
    return expanded.reshape(es[0]*es[1], 9), es     # (batch*676, 9)