# test_learning.py -- Unit tests for the Week 3 learning loop
import numpy as np

# This is a set of unit tests for the different parts of week 3's work.
# The errors they're testing for are self documenting in nature.

def test_squared_error_basic():
    try:
        from part1_error import squared_error
    except:
        print("Part 1 - squared error does not exist")
        return "Part 1 - squared error does not exist"
    
    """Squared error of known values."""
    assert squared_error(1.0, 1.0) == 0.0, "Perfect prediction should give 0 error"
    assert squared_error(0.5, 1.0) == 0.25, "0.5 vs 1.0 should give 0.25"

def test_mean_squared_error():
    try:
        from part1_error import mean_squared_error
    except:
        print("Part 1 - mean squared error does not exist")
        return "Part 1 - mean squared error does not exist"

    predictions = [2.0, 1.5, 1.0, 0.1]
    goals = [2.0, 1.5, 1.0, 0.1]
    assert mean_squared_error(predictions, goals) == 0.0, "Perfect prediction should give 0 error"

    predictions = [1.0, 1.5, 2.0, 0.1]
    goals = [2.0, 1.49, 1.0, 1.0]
    assert mean_squared_error(predictions, goals) == 0.702525, "Error should be 0.702525"

    predictions = [-1.0, -0.5]
    goals = [2.0, 1.5]
    assert mean_squared_error(predictions, goals) == 6.5, "Error should be positive 6.5"

def test_gradient_descent():
    try:
        from part2_gradient_descent import gradient_descent, np_gradient_descent
    except:
        print("Part 2 - gradient descent does not exist")
        return "Part 2 - gradient descent does not exist"

    assert gradient_descent(1.0, 1.0, 1.0, 1)[-1] == 0.0, "Goal matches input, this should be 0"
    assert gradient_descent(1.0, 2.0, 0.5, 2)[-1] == 0.0, "This should produce a 0.0 error after an itertation"

    errors = gradient_descent(0.3, 2.15, 1.0, 45)
    prev_error = 4.0
    for error in errors:
        if error < prev_error:
            error = prev_error
        else:
            print(f"Gradient descent is not decreasing when it should be")
            return "Gradient descent is not decreasing when it should be"
    assert errors[-1] <= 0.001, "This value should be less than 0.001 error threshold by 45 iterations"

    np_errors = np_gradient_descent(0.3, 2.15, 1.0, 45)
    assert np.allclose(errors, np_errors, 1e-9), "Scratch vs Numpy should match"

def test_gradient_descent_alpha():
    try:
        from part3_alpha import gradient_descent_alpha
    except:
        print("Part 3 - gradient descent alpha does not exist")
        return "Part 3 - gradient descent alpha does not exist"

    assert gradient_descent_alpha(1.0, 1.0, 1.0, 0.1, 1)[-1] == 0.0, "Goal matches input, this should be 0"
    assert gradient_descent_alpha(10.0, 2.2, 0.5, 0.015, 8)[-1] <= 0.001, "Error should descend below 0.001 smoothly"
    assert gradient_descent_alpha(10.0, 2.2, 0.5, 1.0, 8)[-1] >= 10000, "Error should explode with the same input, weight, goal, and 1.0 alpha"

def test_divergence_detection():
    try:
        from part4_alpha_experiment import detect_divergence
    except:
        print("Part 4 - detect divergence does not exist")
        return "Part 4 - detect divergence does not exist"

    # These values were generated from variables Gradient Descent Alpha Tests
    Test1 = [7.839999999999999, 1.9599999999999997, 
             0.49000000000000027, 0.12250000000000007, 
             0.030624999999999937, 0.007656249999999907, 
             0.0019140624999999377, 0.00047851562499998443]
    Test2 = [7.839999999999999, 31.359999999999996, 
             125.43999999999998, 501.75999999999993, 
             2007.039999999999, 8028.1599999999935, 
             32112.639999999974, 128450.55999999987]
    Test3 = [7.839999999999999, 7.840000000000004, 
             7.840000000000004, 7.840000000000004, 
             7.840000000000004, 7.840000000000004, 
             7.840000000000004, 7.840000000000004]

    assert not detect_divergence(Test1), "This test is for downward trend. No divergence should be detected"
    assert detect_divergence(Test2), "This test is for upward trend. Divergence should be detected"
    assert detect_divergence(Test3), "This test is for a stalled out trend. Divergence should be detected"
    

if __name__ == '__main__':
    tests = [name for name in dir() if name.startswith('test_')]
    for test_name in sorted(tests):
        test_func = globals()[test_name]
        try:
            test_func()
            print(f' PASS: {test_name}')
        except AssertionError as e:
            print(f' FAIL: {test_name} -- {e}')