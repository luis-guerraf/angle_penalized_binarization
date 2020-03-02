import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

files = ["lazy_kernelwise_scalings9bit.txt", "lazy_kernelwise_scalings8bit.txt", "lazy_kernelwise_scalings7bit.txt",
        "lazy_kernelwise_scalings6bit.txt"]

fig = plt.figure()
ax = plt.axes()

for file in files:
    f = open(file, "r")
    acc = []

    for x in f:
        x = x.split()
        if (x[0] == 'Test:'):
            i = x.index('Acc@1')
            acc.append(float(x[i+1]))

    # Moving average
    for i in range(1, len(acc)-1):
        acc[i] = (acc[i-1] + acc[i] + acc[i+1])/3

    ax.plot(acc)

txt = " bits " + r'$\alpha$'
ax.legend(["9"+txt, "8"+txt, "7"+txt, "6"+txt], frameon=True)

ax.set_xlabel('Epoch', fontsize=14)
ax.set_ylabel('Validation Accuracy (%)', fontsize=14)

plt.savefig('quantizing_scalings.pdf')
plt.show()


