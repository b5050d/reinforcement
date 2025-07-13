# import math

# def distribute_signal_to_bins(angle, strength):
#     angle = angle % (2 * math.pi)
#     bins = [0.0] * 8

#     # scale angle to bin space (0 to 8)
#     index_f = angle / (math.pi / 4)  # 45° per bin
#     lower_bin = int(index_f) % 8
#     upper_bin = (lower_bin + 1) % 8
#     frac = index_f - lower_bin

#     # distribute signal proportionally
#     bins[lower_bin] = (1 - frac) * strength
#     bins[upper_bin] = frac * strength

#     return bins



# ans = distribute_signal_to_bins((math.pi/2)/3, 1)
# print(ans)

draw_dist = 20

# 1 = 1
# 20 = 0

