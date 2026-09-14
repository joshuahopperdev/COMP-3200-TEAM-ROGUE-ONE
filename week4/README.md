# Week 4: Gradient Descent - Many Strikes, Many Adjustments

In contrast to the previous weeks work of gradient descent, this is multiple "knobs".
We have multiple weights per input, and those have multiple adjustments that need to be made
in each iteration.

## Part 1: Many In, One out & The Loud Channel(1b)
In this area, ele_mul() and normalize() are defined, with their respective uses to multiple
weights and a singular output. Gradient descent was then performed with these two functions
to show the effects with and without normalizing the input.

```py
def ele_mul(scalar, vector, debug = False):
    output = [0] * len(vector)
    if debug:
        print(f"Length of vector: {len(vector)}")
    for i in range(len(vector)):
        output[i] = scalar *  vector[i]
    return output

def normalize(channel):
    max_value = max(channel)
    result_array = []
    for i in channel:
        result_array.append(i/max_value)
    return result_array
```

## Part 2: One In, Many Out
This section shows multiple ouputs instead of just a single output, like in part 1. This time,
1 input, 3 weights, gives 3 outputs. Before it was 1 input, 3 weights, 1 output.

```py
def gradient_descent_outputs(input, weights, trues, alpha, iterations, debug = False):
    error_result = []
    weight_hist_result = []
    for iter in range(0, iterations):
        # Predict
        preds = [input * w for w in weights]

        # Compare and get error
        deltas = [preds[i] - trues[i] for i in range(len(weights))]
        errors = [d ** 2 for d in deltas]
        mse = sum(errors) / len(errors)
        error_result.append(mse)

        # Learn
        weighted_deltas = [d * input for d in deltas]
        for i in range(len(weights)):
            weights[i] -= alpha * weighted_deltas[i]

        weight_hist_result.append(weights.copy())

        if debug:
            print(f"Iter: {iter+1}\tPred: {preds}\n\tError: {errors}\n\tWeights: {weights}\n\tMSE: {mse}")

    return weights, error_result, weight_hist_result
```

## Part 3: Many In, Many Out
In part 3 its scaled up to the full intention, multiple weights with multiple outputs.
As before, we can see how the weights change and adjust over the iterations as well
as how it effects the outputs.

```py
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
```

## Part 4: Freezing the Inner Gaze
This section brings in the concept of locking/freezing weights where they won't change 
during the training process or a portion of it. This forces the other weights to do their
adjustments without the influence of the frozen ones being adjusted further.

```py
def gradient_descent_frozen(input, weights, true, alpha, iterations, frozen):
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
```

# Part 5: Watching the Weights

# Part 6: Unit Tests

