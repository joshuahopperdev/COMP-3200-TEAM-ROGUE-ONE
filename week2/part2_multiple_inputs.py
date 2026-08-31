weights = [0.1, 0.2, 0.0] # angle, balance, breath

# Copied from the slides
blade_angle = [8.5, 9.5, 9.9, 9.0] 
balance = [0.65, 0.80, 0.80, 0.90]
breath = [1.2, 1.3, 0.5, 1.0]

# Build the input vector for sensing 0
input = [blade_angle[0], balance[0], breath[0]]

def w_sum(a, b, debug = False):
    """
    This function takes two inputs, makes sure they are same length, and returns their weighted sum.
    """
    output = 0
    assert len(a) == len(b)
    if len(a) == len(b):
        for i in range(len(a)):
            output += a[i] * b[i]
        if debug:
            print(f"Output: {output}\tLength: {len(a)}")
    else:
        if debug:
            print(f"Length of inputs do not match. Length A:{len(a)}\tLength B:{len(b)}")
    return output

def neural_network(input, weights, debug = False):
    """
    This function takes in an array of inputs and weights and performs w_sum on them.
    """
    pred = w_sum(input, weights, debug)
    return pred

def main():
    neural_network(input, weights, debug=True)

    # Next iteration would be to just loop through the whole set of inputs, not just the first.

    # --- Numpy version ---
    # Acquire the magic of numpy
    import numpy as np
    # Convert our two arrays into numpy arrays.
    a = np.array(input)
    b = np.array(weights)
    # Perform the dot product
    pred = a.dot(b)
    # Show off the prediction for comparison
    print(pred)

    # First output
    # Output: 0.9800000000000001      Length: 3
    # Numpy output
    # 0.9800000000000001

    print("Complete Comparison")
    b = np.array(weights)
    for i in range(len(blade_angle)):
        input = [blade_angle[i], balance[i], breath[i]]
        print("----From Scratch----")
        neural_network(input, weights, True)
        a = np.array(input)
        pred = a.dot(b)
        print("----By Numpy----")
        print(f"Output: {pred}\tLength: {len(a)}")


if __name__ == "__main__":
    main()


