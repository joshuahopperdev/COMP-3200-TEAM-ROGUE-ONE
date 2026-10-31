import numpy as np

import preprocess as prep
import analogy
import fillblank as fb
import sentiment
import similarity as sim

def test_preprocessing():
    pass

def test_lookup():
    pass

def test_sentiment():
    pass

def test_similarity():
    pass

def test_neg_sampling_shape():
    pass

# Copied from other tests
if __name__ == "__main__":
    tests = [name for name in dir() if name.startswith("test_")]
    for test_name in sorted(tests):
        test_func = globals()[test_name]
        try:
            test_func()
            print(f" PASS: {test_name}")
        except AssertionError as e:
            print(f" FAIL (AssertionError): {test_name} -- {e}")
