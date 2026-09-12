import matplotlib
matplotlib.use("Agg") # render to a file; no display needed
import matplotlib.pyplot as plt

from part1_multi_input import gradient_descent_multi
# from part4_freeze import gradient_descent_frozen


# PART 1 Plotting
# weight_history[k] is the full weight vector after iteration k
weights, errors, weight_history = gradient_descent_multi(
    [8.5, 0.65, 1.2], [0.1, 0.2, -0.1], 1.0, 0.01, 10)

for j, name in enumerate(["blade_angle", "balance", "breath"]):
    plt.plot([w[j] for w in weight_history], label=name)

plt.xlabel("iteration")
plt.ylabel("weight value")
plt.title("Part 1 weights, alpha = 0.01")
plt.legend()
plt.savefig("fig1_weights.png", dpi=150)
plt.close() # start a clean figure for the next plot


# PART 4 Plotting
# _, _, baseline_history = gradient_descent_frozen(
#     [8.5, 0.65, 1.2], [0.1, 0.2, -0.1], 1.0, 0.01, 5, [])
# _, _, balance_history = gradient_descent_frozen(
#     [8.5, 0.65, 1.2], [0.1, 0.2, -0.1], 1.0, 0.3, 5, [0, 2])
# _, _, breath_history = gradient_descent_frozen(
#     [8.5, 0.65, 1.2], [0.1, 0.2, -0.1], 1.0, 0.3, 5, [0, 1])


# plt.plot([w[1] for w in baseline_history], label='balance (unfrozen, a=0.01)')
# plt.plot([w[1] for w in balance_history], label='balance (frozen, a=0.3)')
# plt.plot([w[1] for w in baseline_history], label='balance (unfrozen, a=0.01)')
# plt.plot([w[1] for w in breath_history], label='balance (frozen, a=0.3)')


# plt.xlabel("iteration")
# plt.ylabel("weight value")
# plt.title("Part 4 weights vs Unfrozen Weights")
# plt.legend()
# plt.savefig("week4/fig2_frozen.png", dpi=150)
# plt.close()

