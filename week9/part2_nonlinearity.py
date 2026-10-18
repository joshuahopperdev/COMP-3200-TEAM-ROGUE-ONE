import numpy as np
from forms import relu

def main():
    np.random.seed(1)
    X = np.random.randn(5, 4)
    W1 = np.random.randn(4, 3)
    W2 = np.random.randn(3, 2)

    Y_two = (X @ W1) @ W2

    W_prime = W1 @ W2
    Y_one = X @ W_prime

    print("Do the two linear layers match the collapsed layer?")
    print(np.allclose(Y_two, Y_one))
    print("Maximum absolute difference:", np.max(np.abs(Y_two - Y_one)))

    Y_nl = relu(X @ W1) @ W2

    print("\nDoes inserting ReLU still match the collapsed linear layer?")
    print(np.allclose(Y_nl, Y_one))
    print("Maximum absolute difference:", np.max(np.abs(Y_nl - Y_one)))

    
    W_best, residuals, rank, singular_values = np.linalg.lstsq(
        X, Y_nl, rcond=None
    )
    best_linear_output = X @ W_best
    residual_norm = np.linalg.norm(Y_nl - best_linear_output)

    print("\nResidual from the best-fit single linear transformation:")
    print(residual_norm)
    print("Is the residual nonzero?", residual_norm > 1e-9)

if __name__ == "__main__":
    main()

# Without ReLU, the two weight matrices can be combined into one because
# matrix multiplication is linear. ReLU changes this by setting negative
# values to zero while keeping positive values, so the second layer receives
# different values than it would from a purely linear transformation.
# Therefore, the two layers can no longer be collapsed into one equivalent
# linear transformation for all possible inputs.