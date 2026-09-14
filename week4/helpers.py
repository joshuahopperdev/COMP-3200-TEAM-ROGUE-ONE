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

def vect_mat_mul(vect, matrix):
    """Takes a vector of inputs, and uses the weighted
    sum of that vector and each row of the matrix for
    a list of outputs."""
    return [w_sum(vect, row, debug=False) for row in matrix]