from part3_alpha import gradient_descent_alpha

 

def did_error_improve(errors):
    """
    When given an array of error values across iterations,
    checks whether we made roughly no progress: 
    is our final error not notably smaller than
    our initial error?

    Note that, if given 0s (perfect results),
    this will tell us it didn't improve, because, well,
    it didn't.

    Error is always positive, so there are no
    sign issues here; a larger error really
    does always mean that our error is, well,
    larger.

    Problem spec calls this function "detect_divergence()",
    but that seems like a silly name; is it really diverging
    if e.g. error is 0?

    Problem spec also says "Checking only whether the last 
    error is larger than the first would quietly report 
    “no problem” for a run whose error never shrinks at all",
    but I think that's a bit silly; "no error shrinkage" is
    probably only relevant in toy cases where you can get it
    exactly on the boundary.
    """
    return errors[-1] < 0.98 * errors[0]


def error_ratio(errors):
    """
    The cousin of did_error_improve(): when passed
    an array of error values, returns the ratio
    between the final and the initial errors.

    If passed a 0 for the initial error value,
    returns a 1; I really hope you didn't somehow
    manage to worsen from a perfect error,
    but if so, you deserve what you get.
    """
    if errors[0]:
        return errors[-1]/errors[0]
    else:
        return 1 




if __name__ == '__main__': 
    alphas_list = [.001, .01, .1, 0.45, 0.5, 0.55, 1.0]
    # for each result we will store a set of four values: 
    # the alpha, the final error, true/false did_error_improve, and error_ratio
    results = [0, 0, 0, 0]*len(alphas_list)
    for i in range(len(alphas_list)):
        errors = gradient_descent_alpha(input = 2.0, goal = 0.8, weight = 0.5, alpha = alphas_list[i], iterations = 20)
        results[i] = [alphas_list[i], errors[-1], did_error_improve(errors), error_ratio(errors)]
        print(f"Test {i+1}:\n  Alpha: {results[i][0]}\n  Final Error: {results[i][1]}\n  Did Error Imrpove: {results[i][2]}\n  Ratio of Final Error to Initial Error: {results[i][3]}")
