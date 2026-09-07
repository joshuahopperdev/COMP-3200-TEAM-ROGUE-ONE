import numpy as np

blade_angle = [8.5, 9.5, 9.9, 9.0]
clean = [1, 1, 0, 1]

def gradient_descent_alpha(input, goal, weight, alpha, iterations):
    errors = []
    for i in range(iterations):
        pred = input * weight
        error = (pred - goal) ** 2
        delta = pred - goal

        weight_delta = delta * input
        weight -= alpha * weight_delta
        errors.append(error)

    return errors


def np_gradient_descent_alpha(input, goal, weight, alpha, iterations):
    errors = []

    input = np.float64(input)
    goal = np.float64(goal)
    weight = np.float64(weight)
    alpha = np.float64(alpha)

    for _ in range(iterations):
        pred = input * weight
        error = np.square(pred - goal)
        delta = pred - goal

        weight_delta = delta * input
        weight -= alpha * weight_delta
        errors.append(error)

    return errors


if __name__ == '__main__': # runs only when YOU run this file
    # step 1
    print("--- 1. Reproducing Divergence (No Alpha) ---")
    input_ex = 2.0
    weight_ex = 0.5
    goal_ex = 0.8

    for i in range(20):
        pred = input_ex * weight_ex
        err = (pred - goal_ex) ** 2
        delta = pred - goal_ex
        weight_ex -= delta * input_ex
        print(f"Iter {i+1:2d} | Error: {err:12.4f} | Pred: {pred:12.4f} | Weight: {weight_ex:12.4f}")

    """
    --- WHY IT EXPLODES (Step 1 Comment) ---
    When input is large (input = 2.0), the update step weight_delta = (pred - goal) * input
    gets multiplied by input twice: once in the error/delta calculation, and again when 
    computing the step size. Without alpha dampening the step, the weight correction 
    vastly overshoots the target value on every single step, causing exponential 
    divergence.
    """

    # step 2 & 3
    print("\n--- 2 & 3. Taming with Alpha = 0.1 ---")
    scratch_errs = gradient_descent_alpha(2.0, 0.8, 0.5, alpha=0.1, iterations=20)
    np_errs = np_gradient_descent_alpha(2.0, 0.8, 0.5, alpha=0.1, iterations=20)
    
    print(f"Initial Error: {scratch_errs[0]:.6f} | Final Error: {scratch_errs[-1]:.6f}")
    
    # Tolerance check required by Part 1/Part 2 rules (abs diff < 1e-10)
    if np.allclose(scratch_errs, np_errs, atol=1e-10):
        print("SUCCESS: From-scratch and NumPy versions match within tolerance.")
    print("\n--- 4. Blade-Angle Dataset Experiments ---")
    
    # step 4
    test_alphas = [0.1, 0.01, 0.001]
    
    for a in [0.1, 0.01, 0.001]:
        errs = gradient_descent_alpha(blade_angle[0], clean[0], weight=0.5, alpha=a, iterations=20)
        print(f"Alpha {a:<5} | Start Err: {errs[0]:10.4f} | Final Err: {errs[-1]:10.6f}")

    """
    --- BLADE ANGLE SELECTION (Step 4 Comment) ---
    Selected Alpha: 0.01
    
    Reasoning:
    Blade angle values (~8.5-9.9) are ~10x larger than balance inputs (~0.65-0.90).
    Since updates scale with x^2, step sizes are ~100x larger. Alpha = 0.1 still 
    explodes here, while alpha = 0.01 dampens updates enough for smooth, steady 
    convergence across all 20 iterations.
    """