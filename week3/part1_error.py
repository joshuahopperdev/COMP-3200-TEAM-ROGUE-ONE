# ======================= #
# Part 1: Measuring Error #
# ======================= #

# -- Scratch --

# Written as a lambda for the sake of one-liners
# To be clear, a lambda is written as
# `lambda *arguments*: *return expr*`
squared_error = lambda prediction, goal: (prediction - goal) ** 2


# Asserts equal lengths of predictions and goals,
# then sums the squared errors and divides by the
# length (of predictions, not that it matters much)
def mean_squared_error(predictions, goals):
    assert len(predictions) == len(
        goals
    ), "Length of predictions and goals should be equal"
    total = 0
    for i in range(len(predictions)):
        total += squared_error(predictions[i], goals[i])
    return total / len(predictions)


# -- Main --
if __name__ == "__main__":
    balance = [0.65, 0.80, 0.80, 0.90]
    clean = [1, 1, 0, 1]
    weight = 0.5

    # -- NumPy version --
    import numpy as np

    # Convert to NumPy arrays, to be used in mse
    np_balance = np.asarray(balance)
    np_clean = np.asarray(clean)

    # I'm not too sure how just one squared error can/should
    # be translated to NumPy...I'll try using np.square()
    np_squared_error = lambda prediction, goal: np.square(prediction - goal)

    # np.mean() takes a NumPy array and returns the average
    # np.square() squares each element of a NumPy array
    # I didn't reuse the np_squared_error lambda above like
    # I did in the scratch implementation because NumPy works
    # differently
    np_mean_squared_error = lambda predictions, goals: np.mean(
        np.square(predictions - goals)
    )

    # Get the predictions
    predictions = [sensing * weight for sensing in balance]
    # Broadcast multiplying the weight across the NumPy array;
    # to be used in mse
    np_preds = np_balance * weight

    # Compare each sensing of balance
    print("=== SCRATCH VS NUMPY ===")
    print("-- Comparing squared error --")
    for i in range(len(predictions)):
        scratch_error = squared_error(predictions[i], clean[i])
        np_error = np_squared_error(np_preds[i], np_clean[i])
        closeness = abs(scratch_error - np_error)
        agreed = closeness < 1e-10
        print(f"""Scratch:{scratch_error:.4f}, \
                NumPy:{np_error:.4f}, \
                Agree? {'YES' if agreed else 'no'} ({closeness:.4f})""")

    # Compare mean squared error
    print()
    print("-- Comparing mean squared error --")
    scratch_mse = mean_squared_error(predictions, clean)
    np_mse = np_mean_squared_error(np_preds, np_clean)
    mean_closeness = abs(scratch_mse - np_mse)
    mean_agreed = mean_closeness < 1e-10
    print(f"""Scratch:{scratch_mse:.4f}, \
        NumPy:{np_mse:.4f}, \
        Agree? {'YES' if mean_agreed else 'no'} ({mean_closeness:.4f})""")
