
from helpers import vect_mat_mul
import copy
import numpy as np

# Takes two vectors, a and b, and outputs a matrix.
# The matrix has len(a) rows and len(b) columns.
# matrix[i][j] = a[i] * b[j]
def outer_product(a, b):
    big_box = [[0]*len(b) for _ in range(len(a))]
    for i in range(len(a)):
        for j in range(len(b)):
            big_box[i][j] = a[i] * b[j]

    return big_box


def gradient_descent_full(input, weights, trues, alpha, iterations):

    
    cur_weights = weights

    weights_history = [[[0]*len(weights[0]) for _ in range(len(weights))] for _ in range(iterations+1)]
    weights_history[0] = copy.deepcopy(weights)

    error_history = [0]*iterations
    for iter in range(iterations):
        preds = vect_mat_mul(input, cur_weights)


        assert len(preds) == len(trues)
        deltas = [0]*len(preds)
        for i in range(len(preds)):
            deltas[i]=preds[i]-trues[i]

        error = 0
        for i in range(len(deltas)):
            error += deltas[i]**2
        error /= len(deltas)
        error_history[iter] = error

        weight_deltas = outer_product(deltas, input)

        for i in range(len(cur_weights)):
            for j in range(len(cur_weights[0])):
                cur_weights[i][j] -= alpha*weight_deltas[i][j]

        weights_history[iter+1] = copy.deepcopy(cur_weights)

    return cur_weights, error_history, weights_history

def main():
    final_weights, errors, weights_history = gradient_descent_full(input = [8.5, 0.65, 1.2], weights = [[0.1, 0.1, -0.3], [0.1, 0.2, 0.0], [0.0, 1.3, 0.1]], 
        trues = [0.0, 1.0, 0.1], alpha = 0.01, iterations = 15)
    print(final_weights)
    print(errors)
    #print(weights_history)


if __name__ == "__main__":
    main()