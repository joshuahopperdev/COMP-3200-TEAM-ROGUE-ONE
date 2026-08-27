# week2/part1_single_prediction.py
blade_angle = [8.5, 9.5, 9.9, 9.0]
weight = 0.5

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
