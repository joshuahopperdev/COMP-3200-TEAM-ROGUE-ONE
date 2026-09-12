
from helpers import vect_mat_mul


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
    weights_history[0] = weights

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
        weights_history[iter+1] = cur_weights

    return cur_weights, error_history, weights_history


final_weights, errors, weights_history = gradient_descent_full([1, 2, 3], [[1, 2, 2], [3, 1, 4], [2, 2, 2]], [1, 2, 3], 0.1, 5)
print(final_weights)
print(errors)
print(weights_history)