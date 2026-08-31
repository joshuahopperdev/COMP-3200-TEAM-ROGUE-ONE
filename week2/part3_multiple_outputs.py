blade_angle = [8.5, 9.5, 9.9, 9.0] # degrees off-vertical
balance = [0.65, 0.80, 0.80, 0.90] # balance reading
breath = [1.2, 1.3, 0.5, 1.0] # exhalations per second

weights = [0.3, 0.2, 0.9] # opens_left, strikes_high, feints
input = balance[0] # one reading of balance

def ele_mul(scalar, vector):
    output = [0] * len(vector)
    for i in range(len(vector)):
        output[i] = scalar * vector[i]
    return output

def neural_network(input, weights):
    pred = ele_mul(input, weights)
    return pred

print("FROM-SCRATCH VERSION") 
print("--------------------") 
for input in balance:
    pred = neural_network(input, weights)
    print(f"Balance: {input} -> Predictions: {pred}")

# -- NumPy version --
import numpy as np

def neural_network_numpy(input, weights):
    input = np.asarray(input)
    weights = np.asarray(weights)

    # np auto multiplies scalars by every element in array
    pred = input * weights
    return pred

print("\nNUMPY VERSION") 
print("------------") 
for input in balance:
    pred = neural_network_numpy(input, weights)
    print(f"Balance: {input} -> Predictions: {pred}")