# Week 3: Gradient Descent

Gradient Descent describes a method of updating weights to find the optimal weight for a set of inputs and outputs. It uses a squared error to derive a weight update curated to the weight at that iteration. This is a full predict, compare, and learn loop--the starting point for the rest of the semester.

So far, our gradient descent uses a squared error, a mean of squared errors, a `weight_delta` derived from the error, and a learning weight to update the weights and minimize error.

## Part 1: Measuring the Miss -- Josiah Duvalian

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