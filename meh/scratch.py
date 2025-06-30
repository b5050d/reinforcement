






from matplotlib import pyplot as plt

def get_intensity(light_pos, player_pos, decay=.95):
    diff = abs(light_pos - player_pos)
    return decay**diff

a = []
size = 128
decay = .95

intensities = []
for i in range(size):
    # val = abs((size/2)-i)
    # intensities.append(1 * ((decay)**val))
    intensities.append(get_intensity(size/2, i, decay))


plt.plot(intensities)
plt.grid()
plt.xlabel("Light Intensity")
plt.xlabel("Position")
plt.title("Light Intensity Map (Light in middle)")
plt.show()