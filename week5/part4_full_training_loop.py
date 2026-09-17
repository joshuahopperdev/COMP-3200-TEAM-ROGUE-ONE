import numpy as np
from part2_forward_hidden import forward
from part3_one_backprop_step import one_step

# 4 sensings, 3 binary tells each
tells = np.array([[1, 0, 1], # foot shift, no guard drop, exhale
[0, 1, 1], # no shift, guard drop, exhale
[0, 0, 1], # only exhale
[1, 1, 1]]) # all three (the bluff)

# Ground truth: 1 = strike imminent, 0 = they will hold
strike = np.array([[1, 1, 0, 0]]).T # column vector, shape (4, 1)



def train(tells, strike, alpha, epochs, hidden_size, seed):
    np.random.seed(seed)

    weights_0_1 = 2 * np.random.random((3, hidden_size)) - 1
    weights_1_2 = 2 * np.random.random((hidden_size, 1)) - 1

    error_history = []

    for epoch in range(epochs):
        total_epoch_error = 0.0

        for i in range(len(tells)):
            layer_0 = tells[i]
            target = strike[i]

            weights_0_1, weights_1_2, layer_2, layer_2_error = one_step(
                layer_0, target, weights_0_1, weights_1_2, alpha
            )

            total_epoch_error += float(layer_2_error[0])

        error_history.append(total_epoch_error)

        # Print total squared error every 10 epochs
        # +1 so it doesn't print right away
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:2d} | Total Error: {total_epoch_error:.6f}")

    return weights_0_1, weights_1_2, error_history

def main():
    alpha = 0.2
    epochs = 60
    hidden_size = 4
    seed = 1

    print("=== Training Stochastic Gradient Descent ===")
    weights_0_1, weights_1_2, error_history = train(
        tells, strike, alpha, epochs, hidden_size, seed
    )

    print("\n=== Final Predictions vs Goals ===")
    for i in range(len(tells)):
        _, pred = forward(tells[i], weights_0_1, weights_1_2)
        goal = strike[i][0]
        pred_val = float(pred[0])
        print(f"Sensing {i}: Pred = {pred_val:7.4f} | Goal = {goal} | Correct Side: {(pred_val > 0.5) == (goal == 1)}")

    print("\n=== Hidden Layer Weights ===")
    print("weights_0_1 (3x4):\n", weights_0_1.round(2))
    print("\nweights_1_2 (4x1):\n", weights_1_2.round(2))

    print("\n=== Hidden Unit Analysis ===")
    print("Unit 0: Detects foot shift without guard drop (positive driver for strikes).")
    print("Unit 1: Dead unit (stays inactivated by negative weights across inputs).")
    print("Unit 2: Detects guard drop without foot shift (positive driver for strikes).")
    print("Unit 3: Detects the bluff condition (fires on all tells to suppress false positives).")

    print("\n=== Hidden-Size Sweep ===")
    sizes = [1, 2, 4, 8, 16]
    seeds = [1, 2, 3]
    
    for h_size in sizes:
        for s in seeds:
            _, _, errs = train(tells, strike, alpha, epochs, h_size, s)
            print(f"Hidden Size: {h_size:2d} | Seed: {s} | Final Total Error: {errs[-1]:.6f}")

if __name__ == "__main__":
    main()


"""
Analysis:

Hidden size 1 fails every time, sizes 4, 8, and 16 succeed every time, and size 2 is a coin flip.

The smallest size that works on seed 1 is size 2, but the smallest that works reliably across all 
seeds is size 4. A network may solve the trial under one seed and fail under another because random 
weight initialization can either position units inside active relu gradient regions or freeze them 
permanently in zero gradient dead zones.
"""