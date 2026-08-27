# angle balance breath
ih_wgt = [[0.1, 0.2, -0.1], # -> hid[0]
[-0.1, 0.1, 0.9], # -> hid[1]
[0.1, 0.4, 0.1]] # -> hid[2]

# hid0 hid1 hid2
hp_wgt = [[0.3, 1.1, -0.3], # -> opens_left?
[0.1, 0.2, 0.0], # -> strikes_high?
[0.0, 1.3, 0.1]] # -> feints?

weights = [ih_wgt, hp_wgt]
# Expected for sensing 0 (rounded): hidden = [0.86, 0.295, 1.23]
# pred = [0.2135, 0.145, 0.5065]