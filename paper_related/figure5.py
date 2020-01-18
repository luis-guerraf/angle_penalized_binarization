import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

f1 = open("multistep_beta_SGD.txt", "r")
f2 = open("exponential_beta_SGD.txt", "r")
acc_multi = []
acc_exp = []
beta_multi = []
beta_exp = []

for x in f1:
    x = x.split()
    if (x[0] == 'Test:'):
        acc_multi.append(float(x[8]))
        beta_multi.append(float(x[10]))

for x in f2:
    x = x.split()
    if (x[0] == 'Test:'):
        acc_exp.append(float(x[8]))
        beta_exp.append(float(x[10]))

# Moving average
for i in range(1, len(acc_multi)-1):
    acc_multi[i] = (acc_multi[i-1] + acc_multi[i] + acc_multi[i+1])/3
    acc_exp[i] = (acc_exp[i-1] + acc_exp[i] + acc_exp[i+1])/3


###  Multistep beta ###
fig, axs = plt.subplots(2, 1, constrained_layout=True)      # Note: Avoid overlap in latex with constrained_layout
ax = axs[0]

color = (0, 0, 0)
ax.set_xlabel('Epoch', fontsize=16)
ax.set_ylabel('Validation Accuracy (%)', fontsize=14, color=color)
ax.plot(acc_multi, color=color)
ax.tick_params(axis='y', labelcolor=color)

ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:red'
ax2.set_ylabel('$\\beta$', fontsize=14, color=color)
ax2.plot(beta_multi, color=color)
ax2.tick_params(axis='y', labelcolor=color)


### Exponential beta ##
ax = axs[1]
color = (0, 0, 0)
ax.set_xlabel('Epoch', fontsize=16)
ax.set_ylabel('Validation Accuracy (%)', fontsize=14, color=color)
ax.plot(acc_exp, color=color)
ax.tick_params(axis='y', labelcolor=color)

ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:red'
ax2.set_ylabel('$\\beta$', fontsize=14, color=color)
ax2.plot(beta_exp, color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.savefig('beta_scheduler.pdf')
plt.show()
