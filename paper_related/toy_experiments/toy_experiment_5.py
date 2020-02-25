import numpy as np
import pylab as py
import torch
from torch.autograd import Function
import math

# Function that I'm going to plot (IF MODIFY, ALSO MODIFY THE LOSS)
def z_func(x, y):
    Z = ((x) ** 4 + (y - 3) ** 4) * ((x - 1) ** 2 + (y - 1) ** 2) + 1
    return np.log2(Z)

def r2m_x(x):
    return [x * 100 + 400]
def r2m_y(y):
    return [y * 100]


class Binarize_W(Function):
    @staticmethod
    def forward(ctx, x):
        return x.sign()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def l2_penalization(layer, beta):
    W_r = layer.view(-1)
    W_q = Binarize_W.forward(None, W_r).detach() * beta.detach()
    l2 = torch.norm(W_r - W_q, p=2)/W_r.nelement()
    return l2


def angle_penalization(layer):
    W_r = layer.view(-1)
    sqrt_n = math.sqrt(len(W_r))
    angle = 1 - (torch.norm(W_r, p=1) / (torch.norm(W_r, p=2) * sqrt_n))
    return angle


# Loss surface
x = np.arange(-4.0, 4.0, 0.01)
y = np.arange(0, 5.0, 0.01)
X, Y = py.meshgrid(x, y)  # grid of point
Z = z_func(X, Y)  # evaluation of the function on the grid
im = py.imshow(Z, cmap=py.cm.RdBu)  # drawing the function
cset = py.contour(Z, np.arange(0, 12, 1), linewidths=0.2, cmap=py.cm.Set2, linestyle='--')
# py.title('$\log2((x^4+(y-3)^4) * ((x-1)^2+(y-1)^2)+1)$')      # latex style title
py.xlabel("$W_1$")
py.ylabel("$W_2$")

# Background lines and initial points
py.scatter(r2m_x(1.5), r2m_y(3), c='green', edgecolors='black',  s=20)
py.scatter(r2m_x(2.25), r2m_y(2.25), c='green', edgecolors='blue',  s=20, zorder=2)

x = np.linspace(400, 799, 100)
py.plot(x, x-400, linestyle='-', c='g', zorder=1)
x = np.linspace(1, 400, 100)
py.plot(x, 400-x, linestyle='-', c='g', zorder=1)
x = np.linspace(1, 499, 100)
py.plot(np.ones(100)*400, x, linestyle='--', c='grey', zorder=1)


# Optimization
x = torch.nn.parameter.Parameter(torch.Tensor([1.5, 3]))
beta = torch.nn.parameter.Parameter(torch.Tensor([2.25]))
optimizer = torch.optim.SGD([x, beta], .05)

trajectory_x_float = []
trajectory_y_float = []
trajectory_x_quant = []
trajectory_y_quant = []
for i in range(20):
    x_ = Binarize_W.apply(x) * beta
    loss = torch.log2(((x_[0]) ** 4 + (x_[1] - 3) ** 4) * ((x_[0] - 1) ** 2 + (x_[1] - 1) ** 2) + 1)
    l2 = l2_penalization(x, beta)
    angle = angle_penalization(x)
    loss = loss + l2*6
    # loss = loss + angle*20
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    trajectory_x_float.append(r2m_x(x[0].item()))
    trajectory_y_float.append(r2m_y(x[1].item()))
    temp = Binarize_W.forward(None, x) * beta
    trajectory_x_quant.append(r2m_x(temp[0].item()))
    trajectory_y_quant.append(r2m_y(temp[1].item()))

py.scatter(trajectory_x_float, trajectory_y_float, c='black', s=10)
py.scatter(trajectory_x_quant, trajectory_y_quant, c='blue', s=10, zorder=2)

trajectory_x_float.append(r2m_x(x[0].item()))
trajectory_y_float.append(r2m_y(x[1].item()))
temp = Binarize_W.forward(None, x) * beta
trajectory_x_quant.append(r2m_x(temp[0].item()))
trajectory_y_quant.append(r2m_y(temp[1].item()))

py.scatter([r2m_x(x[0].item())], [r2m_y(x[1].item())], c='yellow', edgecolors='black', s=20)
py.scatter([r2m_x(temp[0].item())], [r2m_y(temp[1].item())], c='yellow', edgecolors='blue',  s=20, zorder=2)

py.gca().invert_yaxis()
py.savefig('toy_experiment_5.pdf')
py.savefig('toy_experiment_5.svg')
py.show()

