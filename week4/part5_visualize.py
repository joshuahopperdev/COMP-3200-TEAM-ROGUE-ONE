import matplotlib
matplotlib.use("Agg") # render to a file; no display needed
import matplotlib.pyplot as plt

from part1_multi_input import gradient_descent_multi
from part4_freeze import gradient_descent_frozen


# PART 1 Plotting
# weight_history[k] is the full weight vector after iteration k
weights, errors, weight_history = gradient_descent_multi(
    [8.5, 0.65, 1.2], [0.1, 0.2, -0.1], 1.0, 0.01, 20)

for j, name in enumerate(["blade_angle", "balance", "breath"]):
    plt.plot([w[j] for w in weight_history], label=name)

plt.xlabel("iteration")
plt.ylabel("weight value")
plt.title("Part 1 weights, alpha = 0.01")
plt.legend()
plt.savefig("week4/fig1_weights.png", dpi=150)
plt.close() # start a clean figure for the next plot


# PART 4 Plotting
_, _, baseline_history = gradient_descent_frozen(
    [8.5, 0.65, 1.2], [0.1, 0.2, -0.1], 1.0, 0.01, 5, []
    )
_, _, balance_history = gradient_descent_frozen(
    [8.5, 0.65, 1.2], [0.1, 0.2, -0.1], 1.0, 0.3, 5, [0, 2]
    )
_, _, breath_history = gradient_descent_frozen(
    [8.5, 0.65, 1.2], [0.1, 0.2, -0.1], 1.0, 0.3, 5, [0, 1]
    )


plt.plot([w[1] for w in baseline_history], label='balance (unfrozen, a=0.01)')
plt.plot([w[1] for w in balance_history], label='balance (frozen, a=0.3)')
plt.plot([w[2] for w in baseline_history], label='breath (unfrozen, a=0.01)')
plt.plot([w[2] for w in breath_history], label='breath (frozen, a=0.3)')


plt.xlabel("iteration")
plt.ylabel("weight value")
plt.title("Part 4 weights vs Unfrozen Weights")
plt.legend()
plt.savefig("week4/fig2_frozen.png", dpi=150)
plt.close()

"""
Part 5 Observations:

1. Free Weight Trajectory (Figure 2):
   Free weights move much faster and steeper because they must absorb all the slack 
   and correct the full error alone without help from frozen channels.

2. Steepest Line (Figure 1):
   'blade_angle' is the steepest because its large input (8.5) creates much larger 
   updates (delta * input), matching the prediction from Part 1.

3. Flat Lines (Figure 1):
   'balance' and 'breath' look flat because their updates are tiny relative to 
   'blade_angle'. They aren't actually flat; to verify, plot each weight on its own 
   y-axis scale or use a logarithmic scale.
"""