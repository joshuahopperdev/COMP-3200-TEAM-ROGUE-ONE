


# Takes two vectors, a and b, and outputs a matrix.
# The matrix has len(a) rows and len(b) columns.
# matrix[i][j] = a[i] * b[j]
def outer_product(a, b):
    big_box = [[0]*len(b) for _ in range(len(a))]
    for i in range(len(a)):
        for j in range(len(b)):
            big_box[i][j] = a[i] * b[j]

    return big_box
