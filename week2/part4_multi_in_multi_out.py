# angle balance breath
weights = [[0.1, 0.1, -0.3], # -> opens_left?
[0.1, 0.2, 0.0], # -> strikes_high?
[0.0, 1.3, 0.1]] # -> feints?
input = [blade_angle[0], balance[0], breath[0]]
# Expected for sensing 0 (rounded): [0.555, 0.98, 0.965]