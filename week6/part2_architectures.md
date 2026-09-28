# Wider
layer_0 --[ weights_0_1 (3, 32), relu ]--> layer_1 --[ weights_1_2 (32, 1) ]--> layer_2
(1, 3) (1, 32) (1, 1)
Weights: 3x32 + 32x1 = 128

# Deeper
layer_0 --[ weights_0_1 (3, 4), relu ]--> layer_1 --[ weights_1_2 (4, 4), relu ] --> layer_2 --[ weights_2_3 (4, 4), relu ] -->layer_3 --[ weights_3_4 (4, 1) ] --> layer_4
(1, 3) (1, 4) (1, 4) (1, 4) (1, 1)
Weights: 3x4 + 4x4 + 4x4 + 4x4 + 4x1 = 64

# Multi-output
layer_0 --[ weights_0_1 (3, 8), relu ]--> layer_1 --[ weights_1_2 (8, 4) ]--> layer_2
(1, 3) (1, 8) (1, 4)
Weights: 3x8 + 8x4 = 56

# Custom
### 16 measurements from a brain activity sensor are used to classify whether a person is resting, focused, or moving
layer_0 --[ weights_0_1 (16, 32), relu ]--> layer_1 --[ weights_1_2 (32, 16) ]--> layer_2 --[ weights_2_3 (16, 3)]--> layer_3
(1, 16) (1, 32) (1, 16) (1, 3)
Weights: 16x32 + 32x16 + 16x3 = 1072