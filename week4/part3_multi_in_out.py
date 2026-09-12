
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

# takes two vectors of equal length and returns a vector of their differences
def get_deltas(preds, trues):
        assert len(preds) == len(trues)
        deltas = [0]*len(preds)
        for i in range(len(preds)):
            deltas[i]=preds[i]-trues[i]
        return deltas

# takes a vector, squares all its values and sums them, and divides by its length
def calc_mse(deltas):
    sq_sum = 0
    for i in range(len(deltas)):
        sq_sum += deltas[i]**2
    sq_sum /= len(deltas)
    return sq_sum
 


def gradient_descent_full(input, weights, trues, alpha, iterations):

    # kind of irrelevant
    cur_weights = weights

    # make a big box to hold all our weights for each iteration; 1 more slot than there are iterations,
    # to store starting state as well
    weights_history = [[[0]*len(weights[0]) for _ in range(len(weights))] for _ in range(iterations+1)]
    # make a copy of weights and stick it in slot 0
    weights_history[0] = copy.deepcopy(weights)

    # same deal for error history
    error_history = [0]*(iterations+1)

    # as many times as we have iterations...
    for iter in range(iterations):
        # calculate predictions...
        preds = vect_mat_mul(input, cur_weights)

        # find the difference between the predictions and reality...
        deltas = get_deltas(preds, trues)

        # calculate the error and store it...
        error = calc_mse(deltas)
        # appends error before the modification of the weights,
        # so it catches the initial error and misses the last error.
        # We append the last error after the loop.
        error_history[iter] = error

        # create the outer product of our deltas and our inputs...
        weight_deltas = outer_product(deltas, input)

        # and modify our weights by that amount.
        for i in range(len(cur_weights)):
            for j in range(len(cur_weights[0])):
                cur_weights[i][j] -= alpha*weight_deltas[i][j]

        #(And also save those weights after modification.)
        weights_history[iter+1] = copy.deepcopy(cur_weights)

    # run one more pass and save its error too!
    preds = vect_mat_mul(input, cur_weights)
    error_history[-1]=calc_mse(get_deltas(vect_mat_mul(input, cur_weights), trues))


    return cur_weights, error_history, weights_history


def grad_desc_numpy_full(input, weights, trues, alpha, iterations):
    cur_weights = np.array(weights)
    np_trues = np.array(trues)
    np_input = np.array(input)
    weights_history = np.empty((iterations+1, *cur_weights.shape))
    error_history = np.empty(iterations+1)
    for iter in range(iterations):
        # save weights before modification; so we'll need to do it one more time at the end
        weights_history[iter] = cur_weights

        preds = cur_weights@np_input
        deltas = preds - np_trues
        mse = np.mean(deltas**2)
        # same deal, save error before modification
        error_history[iter] = mse

        weight_deltas = np.outer(deltas, np_input)

        cur_weights -= alpha * weight_deltas


    # save final weights
    weights_history[-1] = cur_weights
    # one last pass to find final error, then save it
    error_history[-1]=np.mean(((cur_weights@np_input)-np_trues)**2)

    return cur_weights, error_history, weights_history
    

def main():
    final_weights, errors, weights_history = gradient_descent_full(input = [8.5, 0.65, 1.2], weights = [[0.1, 0.1, -0.3], [0.1, 0.2, 0.0], [0.0, 1.3, 0.1]], 
        trues = [0.0, 1.0, 0.1], alpha = 0.01, iterations = 15)

    # –- NumPy version –-   
    np_weights, np_errors, np_weight_history = grad_desc_numpy_full(input = [8.5, 0.65, 1.2], weights = [[0.1, 0.1, -0.3], [0.1, 0.2, 0.0], [0.0, 1.3, 0.1]], 
        trues = [0.0, 1.0, 0.1], alpha = 0.01, iterations = 15)

    if (np.allclose(np.array(final_weights), np_weights)):
        print("Numpy version and from scratch version agree on final weights!")
    if (np.allclose(np.array(errors), np_errors)):
        print("Numpy version and from scratch version agree on error history!")
    if (np.allclose(np.array(weights_history), np_weight_history)):
        print("Numpy version and from scratch version agree on weight history!")
    

     



if __name__ == "__main__":
    main()



# Final weights are roughly:
# [[0.0363, 0.0951, -0.3090], 
#  [0.1023, 0.2002, 0.0003], 
#  [-0.0992, 1.2924, 0.0860]]
# From a starting point of:
# [[0.1, 0.1, -0.3], 
#  [0.1, 0.2, 0.0], 
#  [0.0, 1.3, 0.1]]
# With input of:
# [8.5, 0.65, 1.2]
# And trues of:
# [0.0, 1.0, 0.1]
# Due to its large input value, around 10 times the other two inputs,
# column 1 shifted substantially, while the others stayed the same.
# But strikes_high started out predicting 0.98, almost exactly
# at the true value, so it barely updated at all - and that's a good
# thing! If you're already there, and nobody else is anywhere close, 
# it's good for you to stay put!