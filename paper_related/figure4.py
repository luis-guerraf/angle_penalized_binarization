import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

f = open("nonlazy_kernelwise_exponential_beta.txt", "r")
acc = []
angle = []

for x in f:
    x = x.split()

    # Take the angle from training but acc from validation (angle range is larger in training)
    if (x[0] == 'Epoch:'):
        angle.append(float(x[7]))
    if (x[0] == 'Test:'):
        acc.append(float(x[8]))

# Moving average
for i in range(1, len(acc)-1):
    acc[i] = (acc[i-1] + acc[i] + acc[i+1])/3
    angle[i] = (angle[i-1] + angle[i] + angle[i+1])/3

fig = plt.figure()
ax = plt.axes()

color = (0, 0, 0)
ax.set_xlabel('Epoch', fontsize=16)
ax.set_ylabel('Validation Accuracy (%)', fontsize=14, color=color)
ax.plot(acc, color=color)
ax.tick_params(axis='y', labelcolor=color)

ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:red'
ax2.set_ylabel('1 - Cosine distance', fontsize=14, color=color)  # we already handled the x-label with ax1
ax2.plot(angle, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('angle_vs_validation_nonlazy.pdf')
plt.show()
