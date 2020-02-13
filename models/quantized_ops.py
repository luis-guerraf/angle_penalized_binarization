import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Function
from torch.nn.parameter import Parameter


bitW = None
learnable_scalings = None
mode = None


class Binarize_W(Function):
    @staticmethod
    def forward(ctx, x):
        # ctx.save_for_backward(x)
        return x.sign()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Xnorize_W(Function):
    @staticmethod
    def forward(self, tensor, _mode):
        s = tensor.size()
        if _mode == "layerwise":
            n = tensor.nelement()
            m = tensor.norm(1, keepdim=True).div(n)
        if _mode == "channelwise":
            n = tensor[0].nelement()
            m = tensor.norm(1, dim=[1,2,3], keepdim=True).div(n)
        if _mode == "kernelwise":
            n = tensor[0, 0].nelement()
            m = tensor.norm(1, dim=[2,3], keepdim=True).div(n)

        tensor = tensor.sign().mul(m.expand(s))

        return tensor

    @staticmethod
    def backward(self, grad_output):
        return grad_output, None


def DoReFa_W(x, numBits):
    # Assumed symmetric distribution of weights (i.e. range [-val, val])
    if numBits == 32:
        return x

    # Bring to range [0, 1] reducing impact of large values
    w_q = torch.tanh(x).div(2 * torch.max(torch.abs(torch.tanh(x)))) + 0.5

    # Quantize to k bits in range [0, 1]
    w_q = quantize(w_q, numBits)

    # Affine to bring to range [-1, 1]
    w_q *= 2
    w_q -= 1

    return w_q


def quantize(x, k):
    n = float(2**k - 1.0)
    x = RoundNoGradient.apply(x, n)
    return x


class RoundNoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, n):
        return torch.round(x*n)/n

    @staticmethod
    def backward(ctx, g):
        return g, None


class QuantizedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        self.padding = padding
        self.stride = stride
        super(QuantizedConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                        stride=self.stride, padding=self.padding,
                                        dilation=dilation, groups=groups, bias=bias)
        if learnable_scalings and mode =='layerwise':
            self.scale = Parameter(torch.Tensor(1))
        elif learnable_scalings and mode =='channelwise':
            self.scale = Parameter(torch.Tensor(out_channels))
        elif learnable_scalings and mode == 'kernelwise':
            self.scale = Parameter(torch.Tensor(out_channels, in_channels))

    def forward(self, input):
        if bitW == 32:
            weight = self.weight
        elif bitW == 1:
            if learnable_scalings:
                # For learnable scalings
                weight = Binarize_W.apply(self.weight)
                if mode == 'layerwise':
                    weight *= self.scale
                elif mode =='channelwise':
                    weight *= self.scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                elif mode == 'kernelwise':
                    weight *= self.scale.unsqueeze(-1).unsqueeze(-1)
            else:
                # For APSQ
                weight = Xnorize_W.apply(self.weight, mode)
        else:
            weight = DoReFa_W(self.weight, bitW)

        output = F.conv2d(input, weight, self.bias,
                          self.stride, self.padding, self.dilation, self.groups)
        return output

    def init_scalings(self):
        tensor = self.weight
        if mode == 'layerwise':
            n = tensor.nelement()
            m = tensor.norm(1).div(n)
        elif mode == 'channelwise':
            n = tensor[0].nelement()
            m = tensor.norm(1, dim=[1,2,3]).div(n)
        elif mode == 'kernelwise':
            n = tensor[0, 0].nelement()
            m = tensor.norm(1, dim=[2,3]).div(n)

        self.scale.data = m

# Conv1d can only have channel-wise scaling, not kernel-wise
class QuantizedConv1d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        self.padding = padding
        self.stride = stride
        super(QuantizedConv1d, self).__init__(in_channels, out_channels, kernel_size,
                                        stride=self.stride, padding=self.padding,
                                        dilation=dilation, groups=groups, bias=bias)
        if learnable_scalings:
            self.scale = Parameter(torch.Tensor(out_channels).uniform_(0, 1))

    def forward(self, input):
        if bitW == 32:
            weight = self.weight
        elif bitW == 1:
            if learnable_scalings:
                # For learnable kernels
                weight = Binarize_W.apply(self.weight)
                weight *= self.scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            else:
                # For APSQ
                weight = Xnorize_W.apply(self.weight, 'channelwise')
        else:
            weight = DoReFa_W(self.weight, bitW)

        output = F.conv2d(input, weight, self.bias,
                          self.stride, self.padding, self.dilation, self.groups)
        return output

    def init_scalings(self):
        tensor = self.weight
        n = tensor[0].nelement()
        m = tensor.norm(1, 3).sum(2).sum(1).div(n)
        self.scale.data = m
