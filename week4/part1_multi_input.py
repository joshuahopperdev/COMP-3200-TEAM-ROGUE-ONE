from helpers import w_sum, vect_mat_mul

def ele_mul(scalar, vector, debug = False):
    """
    Element wise multiplication.
    """
    output = [0] * len(vector)
    if debug:
        print(f"Length of vector: {len(vector)}")
    for i in range(len(vector)):
        output[i] = scalar *  vector[i]

    return output

def main():
    """
    Main function for testing functions.
    """

    # Example from class
    delta = -0.14
    input = [8.5, 0.65, 1.2]
    weight_deltas = ele_mul(delta, input)
    print(weight_deltas)

if __name__ == "__main__":
    main()