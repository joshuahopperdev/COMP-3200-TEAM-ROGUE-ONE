import numpy as np

balance = [0.65, 0.80, 0.80, 0.90]
clean = [1, 1, 0, 1]
# Train on sensing 0: input = 0.65, goal = 1
ex_weight = 0.5 # starting weight

def gradient_descent(input, goal, weight, iterations, err_stop_early = 0.001, stop_early = False, debug = False):
    """
    --- From Scratch ---
    This function is a basic gradient descent function, 
    with the additional functionality of a stop early variable.
    The function will perform x iterations, 
    predicting, 
    comparing the prediction to the goal, calculating error,
    applying the delta to the weight to 'learn'.
    The additional stop early functionality helps cut out early 
    in the event we get a small enough error and save computation.
    """
    result = []
    if debug:
        print("---From Scratch Gradient Descent---")

    for i in range(0, iterations):
        # Predict
        pred = input * weight

        # Compare
        error = (pred - goal) ** 2
        delta = pred - goal

        # Learn
        weight_delta = delta * input
        weight = weight - weight_delta

        result.append(error)
        
        if debug:
            print(f"Iter: {i+1}\tWeight: {weight}\tError: {error}")

        if stop_early:
            if error <= err_stop_early:
                return result

    return result

def np_gradient_descent(input, goal, weight, iterations, err_stop_early = 0.001, stop_early = False, debug = False):
    """
    --- Numpy Version ---
    This function is a basic gradient descent function, 
    with the additional functionality of a stop early variable.
    The function will perform x iterations, 
    predicting, 
    comparing the prediction to the goal, calculating error,
    applying the delta to the weight to 'learn'.
    The additional stop early functionality helps cut out early 
    in the event we get a small enough error and save computation.
    """
    result = []
    weight = np.float64(weight)
    goal = np.float64(goal)
    input = np.float64(input)

    if debug:
        print("---Numpy Gradient Descent---")

    for i in range(0, iterations):
        # Predict
        pred = input * weight

        # Compare
        error = (pred - goal) ** 2
        delta = pred - goal

        # Learn
        weight_delta = delta * input
        weight = weight - weight_delta

        result.append(error)

        if debug:
            print(f"Iter: {i+1}\tWeight: {weight}\tError: {error}")

        if stop_early:
            if error <= err_stop_early:
                return result

    return result

def compare_np_scratch(scratch_return, np_return):
    if len(scratch_return) == len(np_return):
        for i in range(0,len(scratch_return)):
            if scratch_return[i] != np_return[i]:
                print("From Scratch and Numpy Example are not equal.")
                print(f"Iter: {i+1}\t Scratch: {scratch_return[i]}\tNumpy: {np_return[i]}")
                return
        print("From Scratch and Numpy Example are equal.")
    else:
        print("From Scratch and Numpy Example are not equal. Wrong array lengths.")


"""
The stop early helps out a lot because we can clearly see the 2nd and 3rd example perform better
with the starting weight, than the 1st example. The first example takes 7 iterations to get to 
a stopping point given our stop_early of 0.001. The second and third example take only 4 iterations 
to get below 0.001. 
"""
def main():
    scratch_return = gradient_descent(balance[0], clean[0], ex_weight, 30, stop_early=True, debug=True)
    np_return = np_gradient_descent(balance[0], clean[0], ex_weight, 30, stop_early=True, debug=True)
    compare_np_scratch(scratch_return, np_return)
    
    scratch_return = gradient_descent(balance[1], clean[1], ex_weight, 30, stop_early=True, debug=True)
    np_return = np_gradient_descent(balance[1], clean[1], ex_weight, 30, stop_early=True, debug=True)
    compare_np_scratch(scratch_return, np_return)

    scratch_return = gradient_descent(balance[2], clean[2], ex_weight, 30, stop_early=True, debug=True)
    np_return = np_gradient_descent(balance[2], clean[2], ex_weight, 30, stop_early=True, debug=True)
    compare_np_scratch(scratch_return, np_return)


if __name__ == "__main__":
    main()