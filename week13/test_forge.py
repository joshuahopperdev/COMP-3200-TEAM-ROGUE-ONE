import numpy as np
from tensor import Tensor
from layers import Sequential, Linear
from optim import SGD

def test_accumulation():
    pass

def test_add_grad():
    pass

def test_numeric_grad():
    pass

def test_step_dec_los():
    pass

def test_shape_param_count():
    pass

def test_wait():
    pass

def test_mm():
    pass

def test_whole_grad():
    pass

if __name__ == "__main__":
    tests = [name for name in dir() if name.startswith("test_")]
    for test_name in sorted(tests):
        test_func = globals()[test_name]
        try:
            test_func()
            print(f" PASS: {test_name}")
        except AssertionError as e:
            print(f" FAIL (AssertionError): {test_name} -- {e}")
