import torch
import torch.nn as nn
import torch.nn.functional as F
import math

alpha = 2.0
class atan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0.0).to(x.dtype)
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_x = (alpha / 2) / (1 + ((alpha * math.pi / 2) * (x)).square()) * grad_output
        return grad_x

class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama=1.0):
        out = (input > 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None

surrograte_type = 'MG'
print('gradient type: ', surrograte_type)
gamma = 0.5
lens = 0.5
def gaussian(x, mu=0., sigma=.5):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(math.pi)) / sigma
class ActFun_adp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):  # input = membrane potential- threshold
        ctx.save_for_backward(input)
        return input.gt(0).float()  # is firing ???
    @staticmethod
    def backward(ctx, grad_output):  # approximate the gradients
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # temp = abs(input) < lens
        scale = 6.0
        hight = .15
        if surrograte_type == 'G':
            temp = torch.exp(-(input**2)/(2*lens**2))/torch.sqrt(2*torch.tensor(math.pi))/lens
        #multi gaussian
        elif surrograte_type == 'MG':
            temp = gaussian(input, mu=0., sigma=lens) * (1. + hight) \
                - gaussian(input, mu=lens, sigma=scale * lens) * hight \
                - gaussian(input, mu=-lens, sigma=scale * lens) * hight
        elif surrograte_type =='linear':
            temp = F.relu(1-input.abs())
        elif surrograte_type == 'slayer':
            temp = torch.exp(-5*input.abs())
        elif surrograte_type == 'rect':
            temp = input.abs() < 0.5
        return grad_input * temp.float()*gamma

class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias = True):
        super(Linear, self).__init__(in_features, out_features, bias)
    def forward(self, input):
        if input.shape.__len__() == 3:
            B, T = input.shape[0:2]
            output = F.linear(input.flatten(0, 1).contiguous(), self.weight, self.bias)
            N = output.shape[-1]
            output = output.view([B,T,N])
            return output
        elif input.shape.__len__() == 4:
            B, P, T = input.shape[0:3]
            output = F.linear(input.flatten(0, 2).contiguous(), self.weight, self.bias)
            N = output.shape[-1]
            output = output.view([B, P, T, N])
            return output

class Conv1d(nn.Conv1d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros'):
        super(Conv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)
    def forward(self, input):
        B, T = input.shape[0:2]
        output = self._conv_forward(input.flatten(0, 1).contiguous(), self.weight, self.bias)
        C, L = output.shape[1:]
        output = output.view([B,T,C,L])
        return output

class Conv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'):
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)
    def forward(self, input):
        B, T = input.shape[0:2]
        output = self._conv_forward(input.flatten(0, 1).contiguous(), self.weight, self.bias)
        C, H, W = output.shape[1:]
        output = output.view([B,T,C,H,W])
        return output

class Dropout(nn.Dropout):
    def __init__(self, p = 0.5, inplace = False):
        super(Dropout, self).__init__(p, inplace)
    def forward(self, input):
        neuron = list([])
        for t in range(input.shape[1]):
            neuron.append(F.dropout(input[:, t, ...], self.p, self.training, self.inplace))
        return torch.stack(neuron, dim=1)

class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features):
        super(BatchNorm2d, self).__init__(
            num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
    def forward(self, input):
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)
        output = []
        for t in range(input.shape[1]):
            output.append(F.batch_norm(input[:,t,...],
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight, self.bias, bn_training, exponential_average_factor, self.eps))
        return torch.stack(output, dim=1)

class AdaptiveMaxPool1d(nn.AdaptiveMaxPool1d):
    def __init__(self, output_size):
        super(AdaptiveMaxPool1d, self).__init__(output_size)
    def forward(self, input):
        B, T = input.shape[0:2]
        output = F.adaptive_max_pool1d(input.flatten(0,1).contiguous(), self.output_size, self.return_indices)
        C, L = output.shape[1:]
        output = output.view([B,T,C,L])
        return output

class AdaptiveMaxPool2d(nn.AdaptiveMaxPool2d):
    def __init__(self, output_size):
        super(AdaptiveMaxPool2d, self).__init__(output_size)
    def forward(self, input):
        B, T = input.shape[0:2]
        output = F.adaptive_max_pool2d(input.flatten(0, 1).contiguous(), self.output_size, self.return_indices)
        C, H, W = output.shape[1:]
        output = output.view([B,T,C,H,W])
        return output

class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def __init__(self, output_size):
        super(AdaptiveAvgPool2d, self).__init__(output_size)
    def forward(self, input):
        B, T = input.shape[0:2]
        output = F.adaptive_avg_pool2d(input.flatten(0, 1).contiguous(), self.output_size)
        C, H, W = output.shape[1:]
        output = output.view([B,T,C,H,W])
        return output

# class LIF(nn.Module):
#     def __init__(self, thresh, tau, hete_th, hete_tau, train_th, train_tau):
#         super(LIF, self).__init__()
#         # self.act = atan.apply
#         self.act = ActFun_adp.apply
#         if type(tau) is torch.Tensor:
#             tau = hete_tau*torch.randn_like(tau) + tau
#             self.tau = torch.sigmoid(nn.Parameter(-torch.log(tau-1), requires_grad=train_tau))
#         elif type(tau) is float:
#             self.tau = 1/(1+math.exp(-tau))
#         if type(thresh) is torch.Tensor:
#             self.thresh = nn.Parameter(hete_th*torch.randn_like(thresh)+thresh, requires_grad=train_th)
#         elif type(thresh) is float:
#             self.thresh = thresh
#
#     def forward(self, input):
#         T = input.shape[1]
#         mem = 0
#         spike_pot = []
#         mem_pot = []
#         for t in range(T):
#             mem = (input[:, t, ...]-mem) * self.tau + mem
#             spike = self.act(mem - self.thresh)
#             mem = (1 - spike) * mem           ## hard reset
#             # mem = mem - spike * self.thresh     ## soft reset
#             spike_pot.append(spike)
#             mem_pot.append(mem)
#         return torch.stack(spike_pot, dim=1), torch.stack(mem_pot, dim=1)

class LIF(nn.Module):
    def __init__(self, thresh, tau, hete_th, hete_tau, train_th, train_tau):
        super(LIF, self).__init__()
        self.act = atan.apply
        if type(tau) is torch.Tensor:
            tau = hete_tau*torch.randn_like(tau) + tau
            self.tau_x = nn.Parameter(-torch.log(tau-1), requires_grad=train_tau)
        elif type(tau) is float:
            self.tau_x = -math.log2(tau-1)
        if type(thresh) is torch.Tensor:
            self.thresh = nn.Parameter(hete_th*torch.randn_like(thresh)+thresh, requires_grad=train_th)
        elif type(thresh) is float:
            self.thresh = thresh

    def forward(self, input):
        T = input.shape[1]
        mem = 0
        spike_pot = []
        mem_pot = []
        for t in range(T):
            if type(self.tau_x) is float:
                mem = (input[:, t, ...]-mem) * (1/(1+math.exp(-self.tau_x))) + mem
                spike = self.act(mem - self.thresh)
                mem = (1 - spike) * mem         ## hard reset
                # mem = mem - spike * self.thresh   ## soft reset
                spike_pot.append(spike)
                mem_pot.append(mem)
            else:
                mem = (input[:, t, ...]-mem) * self.tau_x.sigmoid() + mem
                spike = self.act(mem - self.thresh)
                mem = (1 - spike) * mem         ## hard reset
                # mem = mem - spike * self.thresh   ## soft reset
                spike_pot.append(spike)
                mem_pot.append(mem)
        return torch.stack(spike_pot, dim=1), torch.stack(mem_pot, dim=1)


# class LIF(nn.Module):
#     def __init__(self, thresh, tau, hete_th, hete_tau, train_th, train_tau):
#         super(LIF, self).__init__()
#         # self.act = atan.apply
#         self.act = ActFun_adp.apply
#         if type(tau) is torch.Tensor:
#             tau = hete_tau*torch.randn_like(tau) + tau
#             self.tau = torch.sigmoid(nn.Parameter(-torch.log(tau-1), requires_grad=train_tau))
#         elif type(tau) is float:
#             self.tau = 1/(1+math.exp(-tau))
#         if type(thresh) is torch.Tensor:
#             self.thresh = nn.Parameter(hete_th*torch.randn_like(thresh)+thresh, requires_grad=train_th)
#         elif type(thresh) is float:
#             self.thresh = thresh
#
#     def forward(self, input):
#         T = input.shape[1]
#         mem = 0
#         spike_pot = []
#         mem_pot = []
#         for t in range(T):
#             mem = (input[:, t, ...]-mem) * self.tau + mem
#             spike = self.act(mem - self.thresh)
#             # mem = (1 - spike) * mem           ## hard reset
#             mem = mem - spike * self.thresh     ## soft reset
#             spike_pot.append(spike)
#             mem_pot.append(mem)
#         return torch.stack(spike_pot, dim=1), torch.stack(mem_pot, dim=1)






