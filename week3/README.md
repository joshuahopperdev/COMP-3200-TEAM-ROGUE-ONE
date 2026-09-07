# Week 3: Gradient Descent

Gradient Descent describes a method of updating weights to find the optimal weight for a set of inputs and outputs. It uses a squared error to derive a weight update curated to the weight at that iteration. This is a full predict, compare, and learn loop--the starting point for the rest of the semester.

So far, our gradient descent uses a squared error, a mean of squared errors, a `weight_delta` derived from the error, and a learning weight to update the weights and minimize error.

## Part 1 -- Measuring the Miss -- Josiah Duvalian

This part focused on the **compare** step of the loop. This goes beyond the predictions of last week and is the first step to building a full deep learning loop.

### What was done

- Implemented `squared_error` function from scratch
- Implemented `mean_squared_error` function from scratch
- Compared the two functions to NumPy equivalents

When scratch was compared to NumPy, the output was identical.

```py
# error = (prediction - goal) ** 2
squared_error = lambda prediction, goal: (prediction - goal) ** 2

def mean_squared_error(predictions, goals):
  total = 0
  for i in range(len(predictions)):
    total += squared_error(predictions[i], goals[i])
  return total / len(predictions)
```

## Part 2 -- Gradient Descent: A Single Weight -- Caleb Hopper

Part 2 built a full predict-compare-learn loop. It contains only the bare minimum with nothing fancy, including
- One single input
- One single goal
- One single weight
- Some number of iterations
- Return value comprising a list of errors from each iteration

### What was done
- `gradient_descent`: from scratch
- `np_gradient_descent`: converts the parameters to `np.float64`
- `compare_np_scratch`: helper that compares errors from scratch and NumPy implementations -- when executed in main, it printed "From Scratch and Numpy Example are equal." all three times.

```py
def gradient_descent(sensing, goal, weight, iterations):
  errors = [0] * iterations
  for i in range(iterations):
    errors[i] = (sensing * weight - goal) ** 2
    weight -= (sensing * weight - goal) * sensing
  return errors
```
