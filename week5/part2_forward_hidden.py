import numpy as np

# 4 sensings, 3 binary tells each
tells = np.array([[1, 0, 1], # foot shift, no guard drop, exhale
[0, 1, 1], # no shift, guard drop, exhale
[0, 0, 1], # only exhale
[1, 1, 1]]) # all three (the bluff)
# Ground truth: 1 = strike imminent, 0 = they will hold
strike = np.array([[1, 1, 0, 0]]).T # column vector, shape (4, 1)

np.random.seed(1)

hidden_size = 4
weights_0_1 = 2 * np.random.random((3, hidden_size)) - 1 # 3 x 4
weights_1_2 = 2 * np.random.random((hidden_size, 1)) - 1 # 4 x 1


# for use on vectors only!
def relu(x):
    zeroed = x * (x > 0)
    print(f"zeroed is{zeroed}")
    return zeroed

def forward(layer_0, weights_0_1, weights_1_2):
    layer_1 = relu(weights_0_1.T@layer_0)
    layer_2 = weights_1_2.T@layer_1
    return layer_1, layer_2


def main():

    # at first I zipped tells and strike together,
    # but there's literally no point, we don't use strike
    # in this step
    
    for i in range(len(tells)):
        layer_1, layer_2 = forward(tells[i], weights_0_1, weights_1_2)
        print(f"For sensing {i}, layer 1's result is {layer_1}, and layer 2's result is {layer_2}\n")
    

if __name__ == "__main__":
    main()