from part2_multiple_inputs import w_sum

weights = [
    # angle balance breath
    [0.1, 0.1, -0.3],  # -> opens_left?
    [0.1, 0.2, 0.0],  # -> strikes_high?
    [0.0, 1.3, 0.1],  # -> feints?
]

blade_angle = [8.5, 9.5, 9.9, 9.0]  # degrees off-vertical
balance = [0.65, 0.80, 0.80, 0.90]  # balance reading
breath = [1.2, 1.3, 0.5, 1.0]  # exhalations per second
input = [blade_angle[0], balance[0], breath[0]]
# Expected for sensing 0 (rounded): [0.555, 0.98, 0.965]

# Bundle each sensing
# [blade_angle[a], balance[a], breath[a]]
# together in an array of sensings
sensings = list(zip(blade_angle, balance, breath))


# One-line list-comprehension scratch version
def vect_mat_mul(vect, matrix):
    """Takes a vector of inputs, and uses the weighted
    sum of that vector and each row of the matrix for
    a list of outputs."""
    return [w_sum(vect, row, debug=False) for row in matrix]


# There should be four sparring sensings
def scratch_nn(debug=False):
    """Runs our scratch implementation of multi-in,
    multi-out neural network on all four sensings"""
    print("-- From scratch --")
    if debug:
        print(f"There are {len(sensings)} sensings: {sensings}")
    outputs = [0] * len(sensings)
    for i in range(len(sensings)):
        outputs[i] = vect_mat_mul(sensings[i], weights)
        print(outputs[i])
    return outputs


# Preconditions: input and weights are NumPy arrays
# and the numpy module is imported
def understand_np_shapes(input, weights):
    """Prints and explains the shapes of a sensing
    (row of input) and the weight matrix"""
    print()
    print("-- NumPy version --")
    print("Understanding Shapes...")
    print(f"Shape of any row of inputs: {input.shape}")
    print(f"Shape of weight matrix: {weights.shape}")
    print(
        "A shape of (3,) means that the array is one-dimensional and has three elements."
    )
    print(
        "A shape of (3,3) means that we have a square matrix with three rows and three columns, or elements in each row."
    )


# -- NumPy version --


# Precondition: numpy is imported
def np_nn(inputs, weights):
    print()
    print("Vector Matrix Multiplication with NumPy...")
    outputs = [0] * len(inputs)
    for i in range(len(inputs)):
        outputs[i] = inputs[i].dot(weights.T)
        print(outputs[i])
    return outputs


def main():
    # Get scratch results
    scratch_outputs = scratch_nn(debug=True)

    # -- NumPy version --
    import numpy as np

    # Convert inputs and weights to NumPy arrays
    np_weights = np.asarray(weights)
    np_sensings = np.asarray(sensings)

    # Print and explain shapes
    understand_np_shapes(np_sensings[0], np_weights)

    # Get NumPy results
    np_outputs = np_nn(np_sensings, np_weights)

    # Compare outputs for closeness
    for i in range(len(scratch_outputs)):
        print(
            f"Sensing {i}: {'close enough' if np.allclose(scratch_outputs[i], np_outputs[i], rtol=1e-9, atol=1e-9) else 'not close enough'}"
        )


if __name__ == "__main__":
    main()
