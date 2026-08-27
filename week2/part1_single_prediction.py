# week2/part1_single_prediction.py
blade_angle = [8.5, 9.5, 9.9, 9.0]
weight = 0.5

# Setting the weight to 0.2 over 0.5 would lower each of the values within blade_angle,
# since it is the only modifier here with the initial values. If it were a negative value,
# it would flip the sign of the input as well, sending it in the opposite direction.

def neural_network(input, weight, debug = False):
    """
    Inputs: input, weight
    Loop through input array, and multiply the weight passed in.
    """
    prediction = []
    for i in input:
        prediction.append(i * weight)

    if debug:
        print(prediction)
    return prediction

neural_network(blade_angle, weight, debug = True)
