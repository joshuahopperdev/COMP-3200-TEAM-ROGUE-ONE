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

    cur_weights = weights

    error_history = [0]*(iterations+1)
    weights_history = [[0]*len(weights) for _ in range(iterations+1)]
    
    for iter in range(iterations):
        weights_history[iter] = copy.deepcopy(cur_weights)

        pred = w_sum(input, cur_weights)

        delta = pred - true

        error_history[iter] = calc_error(delta)

        weight_deltas = ele_mul(delta, input)

        for i in range(len(frozen)):
            weight_deltas[frozen[i]] = 0

        for i in range(len(cur_weights)):
            cur_weights[i] -= alpha*weight_deltas[i]


    weights_history[-1] = copy.deepcopy(cur_weights)
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