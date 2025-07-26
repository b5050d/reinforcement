difference = -1.4
ARENA_SIZE = 300
# min_euclid = 

from matplotlib import pyplot as plt

mines = [1, 10, 25, 50, 100, 150, 200, 250, 299]

rs = []
for m in mines:
    # reward = -.1 * difference * (((ARENA_SIZE*.5) - m)/(ARENA_SIZE*.5))
    # Linear
    reward = ((-.1/ARENA_SIZE) * m) + .1

    # Exponential Decay
    

    print(reward)

    rs.append(reward)

plt.plot(mines, rs)
plt.grid()
plt.show()





