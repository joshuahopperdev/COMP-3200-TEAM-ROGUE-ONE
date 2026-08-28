# from part2_multiple_inputs import w_sum

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


def w_sum(input, weight):
    assert len(input) == len(weight)
    output = 0
    for i in range(len(input)):
        output += input[i] * weight[i]
    return output


def vect_mat_mul(vect, matrix):
    """Takes a vector of inputs, and uses the weighted
    sum of that vector and each row of the matrix for
    a list of outputs."""
    output = [0] * len(matrix)
    for i in range(len(matrix)):
        output[i] = w_sum(vect, matrix[i])
    return output
    # return [w_sum(vect, row) for row in matrix]


# Run on all four sparring sensings
print("-- From scratch --")
bruh = [8.5, 0.65, 1.2]
print(vect_mat_mul(bruh, weights))
print(vect_mat_mul(input, weights))
for i in [1, 2, 3]:
    input = [blade_angle[i], balance[i], breath[i]]
    print(vect_mat_mul(input, weights))

# -- NumPy version --
import numpy as np

# NumPy understanding verification
np_weights = np.asarray(weights)
np_ba = np.asarray(blade_angle)
np_bal = np.asarray(balance)
np_bre = np.asarray(breath)
np_in = np.array([np_ba[0], np_bal[0], np_bre[0]])

print()
print("-- NumPy version --")
print(f"Shape of any row of inputs: {np_in.shape}")
print(f"Shape of weight matrix: {np_weights.shape}")
print(f"Shape of weight matrix transposed: {np_weights.T.shape}")


def np_vmm(input, weights):
    return input.dot(weights.T)


print()
print(np_vmm(np_in, np_weights))
for i in [1, 2, 3]:
    np_in = np.array([np_ba[i], np_bal[i], np_bre[i]])
    print(np_vmm(np_in, np_weights))

# Comparing scratch vs numpy
print()
print("-- Comparing scratch vector matrix multiplication vs NumPy's version --")
for i in range(len(blade_angle)):
    input = [blade_angle[i], balance[i], breath[i]]
    np_input = np.array([blade_angle[i], balance[i], breath[i]])
    scratch_output = np.asarray(vect_mat_mul(input, weights))
    np_output = np_vmm(np_input, np_weights)
    print(f"Input: {input}")
    print(f"Scratch output: {scratch_output}")
    print(f"NumPy output: {np_output}")
    print(f"Close enough? {'yes' if np.allclose(
              scratch_output, 
              np_output, 
              rtol=1e-9, 
              atol=1e-9
              ) else 'no'}")
