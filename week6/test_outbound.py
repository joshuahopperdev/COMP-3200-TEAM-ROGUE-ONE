import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEEK5_DIR = PROJECT_ROOT / "week5"

sys.path.insert(0,str(PROJECT_ROOT))
sys.path.insert(0, str(WEEK5_DIR))


tells = np.array([[1, 0, 1],  # foot shift, no guard drop, exhale
                  [0, 1, 1],  # no shift, guard drop, exhale
                  [0, 0, 1],  # only exhale
                  [1, 1, 1]]) # all three (the bluff)

strike = np.array([[1, 1, 0, 0]]).T

def test_one_refactor_correctness():
    try:
        from week5 import part4_full_training_loop
    except ImportError as e:
        raise ImportError(f"Error retrieving training example from week 5. {e}")
    _, _, error_history = part4_full_training_loop.train(tells, strike, 0.2, 60, 4, 1)

    #print(f"{error_history}")
    final_error_regression = error_history[-1]

    try:
        from part1_refactored_loop import train
    except ImportError as e:
        raise ImportError(f"Error loading train from part 1. {e}")

    final_error = train(tells, strike, 0.2, 60, 4, 1)
    assert np.allclose(final_error, final_error_regression, rtol=1e-9), f"Final error should be near 0.000015. Final error: {final_error}"

def test_shape_sanity():
    pass

def test_deeper_net_runs():
    try:
        from part3_build_from_diagram import train_from_diagram
    except ImportError as e:
        raise ImportError(f"Error loading train_from_diagram from part 3. {e}")
    epochs = 60
    error_history = train_from_diagram(tells, strike, 0.2, epochs, 1)
    assert len(error_history) == epochs, f"Error history length should equal epochs. Error History Length: {len(error_history)}\tEpochs: {epochs}"

def test_deeper_net_convergence():
    try:
        from part3_build_from_diagram import train_and_predict
    except ImportError as e:
        raise ImportError(f"Error loading train_and_predict from part 3. {e}")

    error_history, predictions = train_and_predict(tells, strike, 0.1, 150, 4)

    final_error = error_history[-1]
    assert final_error <  0.001, f"Final Error should be less than 10^-3. Final Error: {final_error}"
    assert predictions[0] > 0.5, f"1st Prediction should be above 0.5. Prediction: {predictions[0]}"
    assert predictions[1] > 0.5, f"2nd Prediction should be above 0.5. Prediction: {predictions[1]}"
    assert predictions[2] < 0.5, f"3rd Prediction should be below 0.5. Prediction: {predictions[2]}"
    assert predictions[3] < 0.5, f"4th Prediction should be below 0.5. Prediction: {predictions[4]}"

def test_determinism():
    try:
        from part3_build_from_diagram import train_from_diagram
    except ImportError as e:
        raise ImportError(f"Error loading train_from_diagram from part 3. {e}")

    test_one_error_history = train_from_diagram(tells, strike, 0.2, 60, 42)
    test_two_error_history = train_from_diagram(tells, strike, 0.2, 60, 42)
    assert len(test_one_error_history) == len(test_two_error_history), f"Length of error histories do not match. Length One: {len(test_one_error_history)}\tLength Two: {len(test_two_error_history)}"
    for i in range(len(test_one_error_history)):
        np.testing.assert_allclose(test_one_error_history[i], test_two_error_history[i]), f"Errors do not match at epoch: {i+1}. {test_one_error_history[i]} to {test_two_error_history[i]}"

if __name__ == "__main__":
    tests = [name for name in dir() if name.startswith("test_")]
    for test_name in sorted(tests):
        test_func = globals()[test_name]
        try:
            test_func()
            print(f" PASS: {test_name}")
        except AssertionError as e:
            print(f" FAIL (AssertionError): {test_name} -- {e}")
        except ImportError as e:
            print(f" FAIL (ImportError): {test_name} -- {e}")