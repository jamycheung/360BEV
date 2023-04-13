import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from timm.models.layers import trunc_normal_
from torch import Tensor
from torch.nn import init
from torch.nn.modules.utils import _pair


class DWConv2d(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size=3, padding=1, bias=False):
        super(DWConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_chans, in_chans, kernel_size=kernel_size,
                                   padding=padding, groups=in_chans, bias=bias)
        self.pointwise = nn.Conv2d(in_chans, out_chans, kernel_size=1, bias=bias)

        nn.init.kaiming_uniform_(self.depthwise.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.pointwise.weight, a=math.sqrt(5))

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class DeformableProjEmbed(nn.Module):
    """ feature map to Projected Embedding
    """
    def __init__(self, in_chans=512, emb_chans=128):
        super().__init__()
        self.kernel_size = kernel_size = 3
        self.stride = stride = 1
        self.padding = padding = 1
        self.proj = nn.Conv2d(in_chans, emb_chans, kernel_size=kernel_size, stride=stride,
                              padding=padding)
        self.offset_conv = nn.Conv2d(in_chans, 2 * kernel_size * kernel_size, kernel_size=kernel_size,
                                     stride=stride, padding=padding)
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        self.modulator_conv = nn.Conv2d(in_chans, 1 * kernel_size * kernel_size, kernel_size=kernel_size,
                                     stride=stride, padding=padding)
        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        self.norm = nn.BatchNorm2d(emb_chans)
        self.act = nn.GELU()

    def deform_proj(self, x):
        # h, w = x.shape[2:]
        max_offset = min(x.shape[-2], x.shape[-1]) // 4
        offset = self.offset_conv(x).clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.proj.weight,
                                          bias=self.proj.bias,
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          )
        return x

    def forward(self, x):
        x = self.deform_proj(x)
        x = self.act(self.norm(x))
        return x


class DeformableMLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super(DeformableMLP, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        if stride != 1:
            raise ValueError('stride must be 1')
        if padding != 0:
            raise ValueError('padding must be 0')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, 1, 1))  # kernel size == 1

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.offset_modulator_conv = DWConv2d(in_channels, 3 * in_channels)

        self.norm = nn.BatchNorm2d(in_channels)
        self.act = nn.GELU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        B, C, H, W = input.size()
        offset_modulator = self.offset_modulator_conv(input)
        offset_y, offset_x, modulator = torch.chunk(offset_modulator, 3, dim=1)
        modulator = 2. * torch.sigmoid(modulator)
        offset = torch.cat((offset_y, offset_x), dim=1)
        max_offset = max(H, W) // 4
        offset = offset.clamp(-max_offset, max_offset)
        x = torchvision.ops.deform_conv2d(input=input,
                                          offset=offset,
                                          weight=self.weight,
                                          bias=self.bias,
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          dilation=self.dilation
                                          )

        x = self.act(self.norm(x))
        return x

    def extra_repr(self) -> str:
        # s = self.__class__.__name__ + '('
        s = ''
        s += '{in_channels}'
        s += ', {out_channels}'
        s += ', kernel_size={kernel_size}'
        s += ', stride={stride}'
        s += ', padding={padding}' if self.padding != (0, 0) else ''
        s += ', dilation={dilation}' if self.dilation != (1, 1) else ''
        s += ', groups={groups}' if self.groups != 1 else ''
        s += ', bias=False' if self.bias is None else ''
        # s += ')'
        return s.format(**self.__dict__)


class DeformableMLPBlock(nn.Module):
    def __init__(self, in_chans=512, emb_chans=64):
        super().__init__()
        # spatial deformable proj
        self.sdp = DeformableProjEmbed(in_chans=in_chans, emb_chans=emb_chans)
        self.h_mlp = DeformableMLP(emb_chans, emb_chans, (1, 3), 1, 0)
        self.w_mlp = DeformableMLP(emb_chans, emb_chans, (3, 1), 1, 0)
        self.c_mlp = nn.Linear(emb_chans, emb_chans)
        self.proj = nn.Linear(emb_chans, emb_chans)

    def forward(self, x):
        x = self.sdp(x)
        # B, C, H, W = x.shape
        h = self.h_mlp(x).permute(0, 2, 3, 1)
        w = self.w_mlp(x).permute(0, 2, 3, 1)
        x = x.permute(0, 2, 3, 1)
        x = x + h + w
        c = self.c_mlp(x)
        x = x + c
        x = self.proj(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class DMLP(nn.Module):
    def __init__(self, vit_params):
        super().__init__()
        emb_chans = vit_params['emb_chans']
        self.head1 = DeformableMLPBlock(in_chans=64, emb_chans=emb_chans)
        self.head2 = DeformableMLPBlock(in_chans=128, emb_chans=emb_chans)
        self.head3 = DeformableMLPBlock(in_chans=320, emb_chans=emb_chans)
        self.head4 = DeformableMLPBlock(in_chans=512, emb_chans=emb_chans)
        self.pred = nn.Conv2d(emb_chans, vit_params['nclass'], 1)

    def forward(self, c1, c2, c3, c4):
        size = c1.size()[2:]
        c4 = self.head4(c4)
        c4 = F.interpolate(c4, size, mode='bilinear', align_corners=True)

        c3 = self.head3(c3)
        c3 = F.interpolate(c3, size, mode='bilinear', align_corners=True)

        c2 = self.head2(c2)
        c2 = F.interpolate(c2, size, mode='bilinear', align_corners=True)

        c1 = self.head1(c1)
        out = c1 + c2 + c3 + c4
        out = self.pred(out)
        return out

