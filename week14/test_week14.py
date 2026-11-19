import numpy as np
from tensor import Tensor
from layers import LSTMCell, Embedding

def test_index_select():
    tensor = Tensor(np.eye(5))
    indices = [[1,2,3],[2,3,4]]
    rows = tensor.index_select(indices)
    assert np.all(rows == [0,1,2,2,1]), f"Rows selected incorrectly, expected [0,1,2,2,1], got {rows}"




if __name__ == "__main__":
    tests = [name for name in dir() if name.startswith("test_")]
    for test_name in sorted(tests):
        test_func = globals()[test_name]
        try:
            test_func()
            print(f" PASS: {test_name}")
        except AssertionError as e:
            print(f" FAIL (AssertionError): {test_name} -- {e}")