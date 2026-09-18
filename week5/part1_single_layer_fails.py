import numpy as np

def single_layer_train(tells, strike, alpha, epochs, seed, debug = False):
    """
    This function trains with a single layer, no hidden layers.
    """
    np.random.seed(seed)
    weights = 2 * np.random.random(3) - 1
    mse_hist = []
    total_error_hist = []
    for epoch in range(epochs):
        total_error = 0
        for i in range(len(tells)):
            pred = tells[i].dot(weights)
            delta = pred - strike[i]
            weights -= alpha * delta * tells[i]
            total_error += (pred - strike[i]) ** 2
            mse = total_error / len(tells)
            mse_hist.append(mse)
            total_error_hist.append(total_error)
        if epoch % 10 == 0:
            print(f"epoch {epoch + 1:>2}: MSE = {mse:.4f}")
            if debug:
                print(f"epoch {epoch + 1:>2}: total_error = {total_error:.4f}")
    if debug:
        print(f"Final Weight: {weights}")

    return weights, mse_hist, total_error_hist


def main():
    tells = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1]])
    strike = np.array([1, 1, 0, 0])
    alpha = 0.1

    single_layer_train(tells, strike, alpha, 60, 1, True)

if __name__ == "__main__":
    main()

"""
This was gone over in class where given the tells, there is no pattern that be derived
linearly. To the human eye, we can tell that the pattern is an XoR with a potential AND
if we had more data for the third variable having 0 components. But we don't.
[1, 0, 1] 1
[0, 1, 1] 1
[0, 0, 1] 0
[1, 1, 1] 0
This can't converge because it has no way for the weights to represent that it has to be
one or the other.
"""