from part1_multi_input import ele_mul
from helpers import w_sum
import copy


# for a single delta, calculates its squared error,
# or mean squared error if you like, seeing as there's
# only one.
def calc_error(delta):
    
    return delta**2



def gradient_descent_frozen(input, weights, true, alpha, iterations, frozen):
    # pred = input w_sum weights, delta = pred - true, error = delta**2,
    # weights_deltas = inputs * delta for non-frozen weights
    # weights = weights - weights_deltas*alpha

    # kinda irrelevant
    cur_weights = weights

    # make the boxes that will hold our past errors and weights.
    # they have size iterations+1 to store the initial state too,
    # e.g. 4 iterations = 5 errors to store including the starting error
    error_history = [0]*(iterations+1)
    # this one is a multi-dimensional box since it'll be storing weight vectors
    weights_history = [[0]*len(weights) for _ in range(iterations+1)]

    # as many times as we're supposed to...
    for iter in range(iterations):
        #store our weights before this iteration...
        weights_history[iter] = copy.deepcopy(cur_weights)

        # run a forward pass...
        pred = w_sum(input, cur_weights)

        # find the difference between that prediction and the truth...
        delta = pred - true

        # calculate the error from that and save it...
        error_history[iter] = calc_error(delta)

        # calculate how much we want to change each weight...
        weight_deltas = ele_mul(delta, input)

        # obliterate all such changes for frozen weights...
        for i in range(len(frozen)):
            # for all values in frozen, set the corresponding spot 
            # in weight_deltas to 0, so we won't update in the next loop
            weight_deltas[frozen[i]] = 0

        # and subtract our (post-freezing) weight modifications!
        for i in range(len(cur_weights)):
            # (scaled by alpha, of course) 
            cur_weights[i] -= alpha*weight_deltas[i]

    # don't forget to save our final weights, since we normally do so at 
    # the start of the loop and it won't loop back one final time!
    weights_history[-1] = copy.deepcopy(cur_weights)
    # run a forward pass here to check the final error too
    error_history[-1] = calc_error(w_sum(input, cur_weights) - true)
    return cur_weights, error_history, weights_history

def main():
    weights_one, err_hist_one, weight_hist_one = gradient_descent_frozen([8.5, 0.65, 1.2], [0.1, 0.2, -0.1], 1.0, 0.01, 5, [])
    print(f"First one, final weights are:\n{weights_one}\nError history is:\n{err_hist_one}\nAnd full weight history is:\n{weight_hist_one}")
    weights_two, err_hist_two, weight_hist_two = gradient_descent_frozen([8.5, 0.65, 1.2], [0.1, 0.2, -0.1], 1.0, 0.3, 5, [0, 2])
    print(f"\n\nSecond one, final weights are:\n{weights_two}\nError history is:\n{err_hist_two}\nAnd full weight history is:\n{weight_hist_two}")
    weights_three, err_hist_three, weight_hist_three = gradient_descent_frozen([8.5, 0.65, 1.2], [0.1, 0.2, -0.1], 1.0, 0.3, 5, [0, 1])
    print(f"\n\nThird one, final weights are:\n{weights_three}\nError history is:\n{err_hist_three}\nAnd full weight history is:\n{weight_hist_three}")




if __name__ == "__main__":
    main()