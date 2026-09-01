# test_learning.py -- Unit tests for the Week 3 learning loop
from part1_error import squared_error, mean_squared_error
from part2_gradient_descent import gradient_descent
from part3_alpha import gradient_descent_alpha
from part4_alpha_experiment import detect_divergence
import numpy as np
def test_squared_error_basic():
    """Squared error of known values."""
    assert squared_error(1.0, 1.0) == 0.0, "Perfect prediction should give 0 error"
    assert squared_error(0.5, 1.0) == 0.25, "0.5 vs 1.0 should give 0.25"
    # ... more tests ...

if __name__ == '__main__':
    tests = [name for name in dir() if name.startswith('test_')]
    for test_name in sorted(tests):
        test_func = globals()[test_name]
        try:
            test_func()
            print(f' PASS: {test_name}')
        except AssertionError as e:
            print(f' FAIL: {test_name} -- {e}')