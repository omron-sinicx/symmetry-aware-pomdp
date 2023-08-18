"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from torchkit import pytorch_utils as ptu
from torchkit.core import PyTorchModule
from torchkit.modules import LayerNorm

from escnn import gspaces
from escnn import nn as enn

relu_name = "relu"
elu_name = "elu"
ACTIVATIONS = {
    relu_name: nn.ReLU,
    elu_name: nn.ELU,
}


class Mlp(PyTorchModule):
    def __init__(
        self,
        hidden_sizes,
        output_size,
        input_size,
        init_w=3e-3,
        hidden_activation=F.relu,
        output_activation=ptu.identity,
        hidden_init=ptu.fanin_init,
        b_init_value=0.1,
        layer_norm=False,
        layer_norm_kwargs=None,
    ):
        self.save_init_params(locals())
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output


class EquiMlp(nn.Module):
    """
    Equi Linear MLP (usually used at the end of actor/critics)
    """
    def __init__(
        self,
        depth,
        in_type,
        hid_type,
        out_type,
        activation='relu',
    ):
        super().__init__()

        self.in_type = in_type
        self.hid_type = hid_type
        self.out_type = out_type
        self.fcs = []
        self.acts = []

        assert depth >= 2

        for i in range(depth):
            if i == 0:
                fc = enn.R2Conv(self.in_type, self.hid_type, kernel_size=1).to(ptu.device)
                act = enn.PointwiseNonLinearity(self.hid_type, f'p_{activation}')
            elif i <= depth - 2:
                fc = enn.R2Conv(self.hid_type, self.hid_type, kernel_size=1).to(ptu.device)
                act = enn.PointwiseNonLinearity(self.hid_type, f'p_{activation}')
            else:
                fc = enn.R2Conv(self.hid_type, self.out_type, kernel_size=1).to(ptu.device)
                act = None

            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)
            self.acts.append(act)

    def forward(self, input):
        dim_0 = input.shape[0]
        input = input.reshape((-1, self.in_type.size, 1, 1))
        h = self.in_type(input)
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.acts[i] is not None:
                h = self.acts[i](h)
        if dim_0 == 1:
            return h.tensor.reshape(1, self.out_type.size)
        else:
            return h.tensor.reshape(dim_0, -1, self.out_type.size)


class FlattenMlp(Mlp):
    """
    if there are multiple inputs, concatenate along last dim
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=-1)
        return super().forward(flat_inputs, **kwargs)


class EquiFlattenMlp(EquiMlp):
    """
    if there are multiple inputs, concatenate along last dim
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=-1)
        return super().forward(flat_inputs, **kwargs)


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    from math import floor

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor(
        ((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1
    )
    w = floor(
        ((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1
    )
    return h, w