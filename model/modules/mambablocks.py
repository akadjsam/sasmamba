import os
import pdb
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import torch_dct as dct
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
# from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from torchvision.models import VisionTransformer
from dc1d.nn import PackedDeformConv1d, DeformConv1d
try:
    from Dwconv.dwconv_layer import DepthwiseFunction
except:
    DepthwiseFunction = None
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
from torch.nn.init import constant_
from torch_dct import dct, idct
try:
    from .csm_triton import CrossScanTriton, CrossMergeTriton, CrossScanTriton1b1, getCSM
    from .csm_triton import CrossScanTritonF, CrossMergeTritonF, CrossScanTriton1b1F
    from .csms6s import CrossScan, CrossMerge, CrossScan_fs_ft, CrossScan_fs_bt, CrossScan_bs_ft, CrossScan_bs_bt, \
        CrossMerge_bs_bt, CrossMerge_bs_ft, CrossMerge_fs_bt, CrossMerge_fs_ft, CrossScan_plus_poselimbs, \
        CrossMerge_plus_poselimbs,CrossScan_plus_poselimbsv2, CrossScan_plus_poselimbs_he, CrossMerge_plus_poselimbs_he, CrossScan_ds_fb, CrossMerge_ds_fb
    from .csms6s import CrossScan_Ab_1direction, CrossMerge_Ab_1direction, CrossScan_Ab_2direction, \
        CrossMerge_Ab_2direction
    from .csms6s import SelectiveScanMamba, SelectiveScanCore, SelectiveScanOflex, SelectiveScanStateFn
    from .csms6s import flops_selective_scan_fn, flops_selective_scan_ref, selective_scan_flop_jit
except:
    from csm_triton import CrossScanTriton, CrossMergeTriton, CrossScanTriton1b1, getCSM
    from csm_triton import CrossScanTritonF, CrossMergeTritonF, CrossScanTriton1b1F
    from csms6s import CrossScan, CrossMerge, CrossScan_fs_ft, CrossScan_fs_bt, CrossScan_bs_ft, CrossScan_bs_bt, \
        CrossMerge_bs_bt, CrossMerge_bs_ft, CrossMerge_fs_bt, CrossMerge_fs_ft, CrossScan_plus_poselimbs, \
        CrossMerge_plus_poselimbs,CrossScan_plus_poselimbsv2, CrossScan_plus_poselimbs_he, CrossMerge_plus_poselimbs_he,CrossScan_ds_fb, CrossMerge_ds_fb
    from csms6s import CrossScan_Ab_1direction, CrossMerge_Ab_1direction, CrossScan_Ab_2direction, \
        CrossMerge_Ab_2direction
    from csms6s import SelectiveScanMamba, SelectiveScanCore, SelectiveScanOflex, SelectiveScanStateFn
    from csms6s import flops_selective_scan_fn, flops_selective_scan_ref, selective_scan_flop_jit
try :
    from graph import GCN
except:
    from .graph import GCN
from torchvision.ops import DeformConv2d
import einops
# from zeta.nn.modules.p_scan import pscan
import selective_scan_cuda_oflex_rh
# try:
#     from model.ops_dcnv3.functions import DCNv3Function
# except:
#     from ..ops_dcnv3.functions import DCNv3Function

# try:
#     from ..ops_dcnv3.functions.dcnv3_func import DCNv3Function
# except:
#     try:
#         from .ops_dcnv3.functions.dcnv3_func import DCNv3Function
#     except:
#         from ops_dcnv3.functions.dcnv3_func import DCNv3Function
# =====================================================
# we have this class as linear and conv init differ from each other
# this function enable loading from both conv2d or linear
class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        # B, C, H, W = x.shape
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                             error_msgs)


def da_selective_scan(x, delta, A, B):
    """
    Perform selective scan operation on the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (B, L, ED).
        delta (torch.Tensor): Delta tensor of shape (B, L, ED).
        A (torch.Tensor): A tensor of shape (ED, N).
        B (torch.Tensor): B tensor of shape (B, L, N).
        C (torch.Tensor): C tensor of shape (B, L, N).
        D (torch.Tensor): D tensor of shape (ED).

    Returns:
        torch.Tensor: Output tensor of shape (B, L, ED).
    """

    _, L, _ = x.shape

    deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)
    deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)

    BX = deltaB * x.unsqueeze(-1)  # (B, L, ED, N)

    hs = pscan(deltaA, BX)


    return hs

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm, channel_first=False):
        super().__init__()
        self.dim = dim
        Linear = Linear2d if channel_first else nn.Linear
        self._patch_merging_pad = self._patch_merging_pad_channel_first if channel_first else self._patch_merging_pad_channel_last
        self.reduction = Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad_channel_last(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    @staticmethod
    def _patch_merging_pad_channel_first(x: torch.Tensor):
        H, W = x.shape[-2:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2]  # ... H/2 W/2
        x1 = x[..., 1::2, 0::2]  # ... H/2 W/2
        x2 = x[..., 0::2, 1::2]  # ... H/2 W/2
        x3 = x[..., 1::2, 1::2]  # ... H/2 W/2
        x = torch.cat([x0, x1, x2, x3], 1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class FANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True, with_gate=True):
        super(FANLayer, self).__init__()
        self.input_linear_p = nn.Linear(input_dim, output_dim // 4, bias=bias)
        self.input_linear_g = nn.Linear(input_dim, (output_dim - output_dim // 2))
        self.activation = nn.GELU()
        if with_gate:
            self.gate = nn.Parameter(torch.randn(1, dtype=torch.float32))

    def forward(self, src):
        g = self.activation(self.input_linear_g(src))
        p = self.input_linear_p(src)

        if not hasattr(self, 'gate'):
            output = torch.cat((torch.cos(p), torch.sin(p), g), dim=-1)
        else:
            gate = torch.sigmoid(self.gate)
            output = torch.cat((gate * torch.cos(p), gate * torch.sin(p), (1 - gate) * g), dim=-1)
        return output

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Mlp2(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class gMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        self.channel_first = channels_first
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, 2 * hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
        x = self.fc2(x * self.act(z))
        x = self.drop(x)
        return x

class FreqMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        b, f, _ = x.shape
        x = dct.dct(x.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = dct.idct(x.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        return x


class FreqTMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # 输入 (b, f, v, c)
        b, f, v, c = x.shape

        # 变换到 (b, v, c, f) 以便在时间维度 (f) 上应用 DCT
        x = rearrange(x, 'b f v c -> b v c f')
        x = dct(x, norm='ortho')
        x = rearrange(x, 'b v c f -> b f v c')
        # MLP 操作
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = rearrange(x, 'b f v c -> b v c f')
        # 反变换
        x = idct(x, norm='ortho')
        x = rearrange(x, 'b v c f -> b f v c')

        return x

class FreqMlp1DTime(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # Assume input is (n, t, v, c)
        b, t, v, c = x.shape

        # Reshape for time FFT (t dimension only)
        x = rearrange(x, 'b t v c -> b (v c) t')

        # Apply FFT along the time dimension
        x_fft = torch.fft.fft(x.float(), dim=-1)

        # Real and Imaginary components
        x_real, x_imag = x_fft.real, x_fft.imag

        # MLP on real and imaginary separately
        x_real = self.fc1(x_real)
        x_real = self.act(x_real)
        x_real = self.drop(x_real)
        x_real = self.fc2(x_real)
        x_real = self.drop(x_real)

        x_imag = self.fc1(x_imag)
        x_imag = self.act(x_imag)
        x_imag = self.drop(x_imag)
        x_imag = self.fc2(x_imag)
        x_imag = self.drop(x_imag)

        # Combine real and imaginary parts
        x_fft_out = torch.complex(x_real, x_imag)

        # Inverse FFT
        x_out = torch.fft.ifft(x_fft_out, dim=-1).real

        # Reshape back to original shape
        x_out = rearrange(x_out, 'b (v c) t -> b t v c', v=v)

        return x_out


# Test the module
# x = torch.randn(2, 243, 17, 64)
# model = FreqMlp1DTime(in_features=64, hidden_features=128, out_features=64)
# y = model(x)
# print(y.shape)  # Should be (2, 243, 17, 64)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MixedBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm3 = norm_layer(dim)
        self.mlp2 = FreqMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        b, f, c = x.shape
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x1 = x[:, :f//2] + self.drop_path(self.mlp1(self.norm2(x[:, :f//2])))
        x2 = x[:, f//2:] + self.drop_path(self.mlp2(self.norm3(x[:, f//2:])))
        return torch.cat((x1, x2), dim=1)

class SoftmaxSpatial(nn.Softmax):
    def forward(self, x: torch.Tensor):
        if self.dim == -1:
            B, C, H, W = x.shape
            return super().forward(x.view(B, C, -1)).view(B, C, H, W)
        elif self.dim == 1:
            B, H, W, C = x.shape
            return super().forward(x.view(B, -1, C)).view(B, H, W, C)
        else:
            raise NotImplementedError


# =====================================================
class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D


class BiSTSSM_v2:
    def __initv2__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v2",
            channel_first=False,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear
        self.forward = self.forwardv2

        # tags for forward_type ==============================
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        if out_norm_none:
            self.out_norm = nn.Identity()
        elif out_norm_dwconv3:
            self.out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            self.out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            self.out_norm = nn.Sigmoid()
        else:
            LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm
            self.out_norm = LayerNorm(d_inner)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanMamba),
            # will be deleted in the future
            v02=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanMamba,
                        CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            v03=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanOflex,
                        CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            v04=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex,
                        CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            v05=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True,
                        CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            # ===============================
            v051d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True,
                          CrossScan=getCSM(1)[0], CrossMerge=getCSM(1)[1],
                          ),
            v052d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True,
                          CrossScan=getCSM(2)[0], CrossMerge=getCSM(2)[1],
                          ),
            v052dc=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True,
                           cascade2d=True),
            # ===============================
            v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanCore),
            v2_fs_ft=partial(self.forward_corev2, force_fp32=(not self.disable_force32), CrossScan=CrossScan_fs_ft,
                             SelectiveScan=SelectiveScanCore, CrossMerge=CrossMerge_fs_ft),
            v2_fs_bt=partial(self.forward_corev2, force_fp32=(not self.disable_force32), CrossScan=CrossScan_fs_bt,
                             SelectiveScan=SelectiveScanCore, CrossMerge=CrossMerge_fs_bt),
            v2_bs_ft=partial(self.forward_corev2, force_fp32=(not self.disable_force32), CrossScan=CrossScan_bs_ft,
                             SelectiveScan=SelectiveScanCore, CrossMerge=CrossMerge_bs_ft),
            v2_bs_bt=partial(self.forward_corev2, force_fp32=(not self.disable_force32), CrossScan=CrossScan_bs_bt,
                             SelectiveScan=SelectiveScanCore, CrossMerge=CrossMerge_bs_bt),
            v2_plus_poselimbs=partial(self.forward_corev2, force_fp32=(not self.disable_force32),
                                      CrossScan=CrossScan_plus_poselimbs, SelectiveScan=SelectiveScanCore,
                                      CrossMerge=CrossMerge_plus_poselimbs),
            v2_plus_poselimbs_v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32),
                                      CrossScan=CrossScan_plus_poselimbs, SelectiveScan=SelectiveScanOflex,
                                      CrossMerge=CrossMerge_plus_poselimbs),
            v2_plus_poselimbs_v3=partial(self.forward_corev2, force_fp32=(not self.disable_force32),
                                         CrossScan=CrossScan_plus_poselimbsv2, SelectiveScan=SelectiveScanOflex,
                                         CrossMerge=CrossMerge_plus_poselimbs),
            v2_plus_poselimbs_he = partial(self.forward_corev2, force_fp32=(not self.disable_force32),
                                       CrossScan=CrossScan_plus_poselimbs_he, SelectiveScan=SelectiveScanOflex,
                                       CrossMerge=CrossScan_plus_poselimbs_he),
            v3=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex),
            # v1=partial(self.forward_corev2, force_fp32=True, SelectiveScan=SelectiveScanOflex),
            # v4=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True, CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            # ===============================
            v31d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex,
                         CrossScan=CrossScan_Ab_1direction, CrossMerge=CrossMerge_Ab_1direction,
                         ),
            v32d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex,
                         CrossScan=CrossScan_Ab_2direction, CrossMerge=CrossMerge_Ab_2direction,
                         ),
            v32dc=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, cascade2d=True),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)
        k_group = 4  # 双向的或者单向的话 控制该值即可。 #todo.

        # in proj =======================================
        d_proj = d_inner if self.disable_z else (d_inner * 2)
        self.in_proj = Linear(d_model, d_proj, bias=bias)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.with_dconv:
            # self.conv2d = nn.Conv2d(
            #     in_channels=d_inner,
            #     out_channels=d_inner,
            #     groups=d_inner,
            #     bias=conv_bias,
            #     kernel_size=d_conv,
            #     padding=(d_conv - 1) // 2,
            #     **factory_kwargs,
            # )
            self.conv2d = DepthwiseDCN(d_inner, d_conv=d_conv, conv_bias=conv_bias)

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            # dt proj ============================
            self.dt_projs = [
                self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
                for _ in range(k_group)
            ]
            self.dt_projs_weight = nn.Parameter(
                torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
            self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
            del self.dt_projs

            # A, D =======================================
            self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)
            self.Ds = self.D_init(d_inner, copies=k_group, merge=True)  # (K * D)
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(
                torch.randn((k_group * d_inner, d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((k_group, d_inner)))
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(
                torch.zeros((k_group * d_inner, d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, d_inner)))

    # 总是使用core v2
    def forward_corev2(
            self,
            x: torch.Tensor = None,
            # ==============================
            to_dtype=True,  # True: final out to dtype
            force_fp32=False,  # True: input fp32
            # ==============================
            ssoflex=True,  # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
            # ==============================
            SelectiveScan=SelectiveScanOflex,  # 使用SelectiveScanOflex
            CrossScan=CrossScan,  # CrossScan_plus_poselimbs
            CrossMerge=CrossMerge, # CrossMerge_plus_poselimbs
            no_einsum=False,  # replace einsum with linear or conv1d to raise throughput
            # ==============================
            cascade2d=False,
            **kwargs,
    ):
        # print(SelectiveScan, CrossScan, CrossMerge, cascade2d)
        # <class 'lib.model.csms6s.SelectiveScanCore'> <class 'lib.model.csms6s.CrossScan'> <class 'lib.model.csms6s.CrossMerge'> False
        x_proj_weight = self.x_proj_weight
        x_proj_bias = getattr(self, "x_proj_bias", None)
        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds
        delta_softplus = True
        out_norm = getattr(self, "out_norm", None)
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, -1, -1, ssoflex)

        if cascade2d:
            def scan_rowcol(
                    x: torch.Tensor,
                    proj_weight: torch.Tensor,
                    proj_bias: torch.Tensor,
                    dt_weight: torch.Tensor,
                    dt_bias: torch.Tensor,  # (2*c)
                    _As: torch.Tensor,  # As = -torch.exp(A_logs.to(torch.float))[:2,] # (2*c, d_state)
                    _Ds: torch.Tensor,
                    width=True,
            ):
                # x: (B, D, H, W)
                # proj_weight: (2 * D, (R+N+N))
                XB, XD, XH, XW = x.shape
                if width:
                    _B, _D, _L = XB * XH, XD, XW
                    xs = x.permute(0, 2, 1, 3).contiguous()
                else:
                    _B, _D, _L = XB * XW, XD, XH
                    xs = x.permute(0, 3, 1, 2).contiguous()
                xs = torch.stack([xs, xs.flip(dims=[-1])], dim=2)  # (B, H, 2, D, W)
                if no_einsum:
                    x_dbl = F.conv1d(xs.view(_B, -1, _L), proj_weight.view(-1, _D, 1),
                                     bias=(proj_bias.view(-1) if proj_bias is not None else None), groups=2)
                    dts, Bs, Cs = torch.split(x_dbl.view(_B, 2, -1, _L), [R, N, N], dim=2)
                    dts = F.conv1d(dts.contiguous().view(_B, -1, _L), dt_weight.view(2 * _D, -1, 1), groups=2)
                else:
                    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, proj_weight)
                    if x_proj_bias is not None:
                        x_dbl = x_dbl + x_proj_bias.view(1, 2, -1, 1)
                    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_weight)

                xs = xs.view(_B, -1, _L)
                dts = dts.contiguous().view(_B, -1, _L)
                As = _As.view(-1, N).to(torch.float)
                Bs = Bs.contiguous().view(_B, 2, N, _L)
                Cs = Cs.contiguous().view(_B, 2, N, _L)
                Ds = _Ds.view(-1)
                delta_bias = dt_bias.view(-1).to(torch.float)

                if force_fp32:
                    xs = xs.to(torch.float)
                dts = dts.to(xs.dtype)
                Bs = Bs.to(xs.dtype)
                Cs = Cs.to(xs.dtype)

                ys: torch.Tensor = selective_scan(
                    xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
                ).view(_B, 2, -1, _L)
                return ys

            As = -torch.exp(A_logs.to(torch.float)).view(4, -1, N)
            y_row = scan_rowcol(
                x,
                proj_weight=x_proj_weight.view(4, -1, D)[:2].contiguous(),
                proj_bias=(x_proj_bias.view(4, -1)[:2].contiguous() if x_proj_bias is not None else None),
                dt_weight=dt_projs_weight.view(4, D, -1)[:2].contiguous(),
                dt_bias=(dt_projs_bias.view(4, -1)[:2].contiguous() if dt_projs_bias is not None else None),
                _As=As[:2].contiguous().view(-1, N),
                _Ds=Ds.view(4, -1)[:2].contiguous().view(-1),
                width=True,
            ).view(B, H, 2, -1, W).sum(dim=2).permute(0, 2, 1, 3)
            y_col = scan_rowcol(
                y_row,
                proj_weight=x_proj_weight.view(4, -1, D)[2:].contiguous().to(y_row.dtype),
                proj_bias=(
                    x_proj_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if x_proj_bias is not None else None),
                dt_weight=dt_projs_weight.view(4, D, -1)[2:].contiguous().to(y_row.dtype),
                dt_bias=(
                    dt_projs_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if dt_projs_bias is not None else None),
                _As=As[2:].contiguous().view(-1, N),
                _Ds=Ds.view(4, -1)[2:].contiguous().view(-1),
                width=False,
            ).view(B, W, 2, -1, H).sum(dim=2).permute(0, 2, 3, 1)
            y = y_col
        else:
            xs = CrossScan.apply(x)
            if no_einsum:
                x_dbl = F.conv1d(xs.view(B, -1, L), x_proj_weight.view(-1, D, 1),
                                 bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
                dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
                dts = F.conv1d(dts.contiguous().view(B, -1, L), dt_projs_weight.view(K * D, -1, 1), groups=K)
            else:
                x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
                if x_proj_bias is not None:
                    x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
                dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

            xs = xs.view(B, -1, L)
            dts = dts.contiguous().view(B, -1, L)
            As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
            Bs = Bs.contiguous().view(B, K, N, L)
            Cs = Cs.contiguous().view(B, K, N, L)
            Ds = Ds.to(torch.float)  # (K * c)
            delta_bias = dt_projs_bias.view(-1).to(torch.float)

            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

            ys: torch.Tensor = selective_scan(
                xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
            ).view(B, K, -1, H, W)

            y: torch.Tensor = CrossMerge.apply(ys)

            if getattr(self, "__DEBUG__", False):
                setattr(self, "__data__", dict(
                    A_logs=A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                    us=xs, dts=dts, delta_bias=delta_bias,
                    ys=ys, y=y,
                ))

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)  # (B, L, C)
        y = out_norm(y)

        return (y.to(x.dtype) if to_dtype else y)

    def forwardv2(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        # print(self.disable_z, self.disable_z_act, self.channel_first, self.with_dconv)
        # False False False True
        if not self.disable_z:
            # print(x.shape)
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))  # (b, h, w, d)
            # print(x.shape, z.shape)
            if not self.disable_z_act:
                z = self.act(z)
        if not self.channel_first:
            # print(x.shape)
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)
        # torch.Size([1, 256, 243, 17])   input shape : n c t v
        y = self.forward_core(x)   # out shape : n t v c
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        # print('this is forwardv2')
        return out


class BiSTSSM(nn.Module, mamba_init, BiSTSSM_v2):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v2",   # v2_plus_poselimbs
            channel_first=False,
            # ======================
            **kwargs,
    ):
        super().__init__()
        kwargs.update(
            d_model=d_model, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, dt_init_floor=dt_init_floor,
            initialize=initialize, forward_type=forward_type, channel_first=channel_first,
        )
        self.__initv2__(**kwargs)

class BiSTSSMBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
            channel_first=False,
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",  # v2_plus_poselimbs
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            gmlp=False,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = BiSTSSM(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                forward_type=forward_type,  # v2_plus_poselimbs
                channel_first=channel_first,
            )

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                            drop=mlp_drop_rate, channels_first=channel_first)

    def _forward(self, input: torch.Tensor):
        x = input
        x = x + self.drop_path(self.op(self.norm(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)





class SelectiveScanStateFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, A, B, D=None, z=None, delta_bias=None, delta_softplus=False,
                return_last_state=False, lag=0):
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if z is not None and z.stride(-1) != 1:
            z = z.contiguous()
        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
            ctx.squeeze_B = True

        out, x, *rest = selective_scan_cuda_oflex_rh.fwd(u, delta, A, B, D, delta_bias, delta_softplus, 1, True)
        ctx.delta_softplus = delta_softplus
        ctx.has_z = z is not None
        last_state = x[:, :, -1, 1::2]  # (batch, dim, dstate)
        if not ctx.has_z:
            ctx.save_for_backward(u, delta, A, B, D, delta_bias, x)
            return out if not return_last_state else (out, last_state)
        else:
            ctx.save_for_backward(u, delta, A, B, D, z, delta_bias, x, out)
            out_z = rest[0]
            return out_z if not return_last_state else (out_z, last_state)

    @staticmethod
    def backward(ctx, dout, *args):
        if not ctx.has_z:
            u, delta, A, B, D, delta_bias, x = ctx.saved_tensors
            z = None
            out = None
        else:
            u, delta, A, B, D, z, delta_bias, x, out = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        # Here we just pass in None and dz will be allocated in the C++ code.
        du, ddelta, dA, dB, dD, ddelta_bias, *rest = selective_scan_cuda_oflex_rh.bwd(
            u, delta, A, B, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        dz = rest[0] if ctx.has_z else None
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        return (du, ddelta, dA, dB,
                dD if D is not None else None,
                dz,
                ddelta_bias if delta_bias is not None else None,
                None,
                None,
                None)

def selective_scan_fn(u, delta, A, B, D=None, z=None, delta_bias=None, delta_softplus=False,
                     return_last_state=False):
    """if return_last_state is True, returns (out, last_state)
    last_state has shape (batch, dim, dstate). Note that the gradient of the last state is
    not considered in the backward pass.
    """

    return SelectiveScanStateFn.apply(u, delta, A, B, D, z, delta_bias, delta_softplus, return_last_state)

class StateFusion(nn.Module):
    def __init__(self, dim):
        super(StateFusion, self).__init__()

        self.dim = dim
        self.kernel_3   = nn.Parameter(torch.ones(dim, 1, 3, 3))
        self.kernel_3_1 = nn.Parameter(torch.ones(dim, 1, 3, 3))
        self.kernel_3_2 = nn.Parameter(torch.ones(dim, 1, 3, 3))
        self.alpha = nn.Parameter(torch.ones(3), requires_grad=True)

    @staticmethod
    def padding(input_tensor, padding):
        return torch.nn.functional.pad(input_tensor, padding, mode='replicate')

    def forward(self, h):

        if self.training:
            h1 = F.conv2d(self.padding(h, (1,1,1,1)), self.kernel_3,   padding=0, dilation=1, groups=self.dim)
            h2 = F.conv2d(self.padding(h, (3,3,3,3)), self.kernel_3_1, padding=0, dilation=3, groups=self.dim)
            h3 = F.conv2d(self.padding(h, (5,5,5,5)), self.kernel_3_2, padding=0, dilation=5, groups=self.dim)
            out = self.alpha[0]*h1 + self.alpha[1]*h2 + self.alpha[2]*h3
            return out

        else:
            if not hasattr(self, "_merge_weight"):
                self._merge_weight = torch.zeros((self.dim, 1, 11, 11), device=h.device)
                self._merge_weight[:, :, 4:7, 4:7] = self.alpha[0]*self.kernel_3

                self._merge_weight[:, :, 2:3, 2:3] = self.alpha[1]*self.kernel_3_1[:,:,0:1,0:1]
                self._merge_weight[:, :, 2:3, 5:6] = self.alpha[1]*self.kernel_3_1[:,:,0:1,1:2]
                self._merge_weight[:, :, 2:3, 8:9] = self.alpha[1]*self.kernel_3_1[:,:,0:1,2:3]
                self._merge_weight[:, :, 5:6, 2:3] = self.alpha[1]*self.kernel_3_1[:,:,1:2,0:1]
                self._merge_weight[:, :, 5:6, 5:6] += self.alpha[1]*self.kernel_3_1[:,:,1:2,1:2]
                self._merge_weight[:, :, 5:6, 8:9] = self.alpha[1]*self.kernel_3_1[:,:,1:2,2:3]
                self._merge_weight[:, :, 8:9, 2:3] = self.alpha[1]*self.kernel_3_1[:,:,2:3,0:1]
                self._merge_weight[:, :, 8:9, 5:6] = self.alpha[1]*self.kernel_3_1[:,:,2:3,1:2]
                self._merge_weight[:, :, 8:9, 8:9] = self.alpha[1]*self.kernel_3_1[:,:,2:3,2:3]

                self._merge_weight[:, :, 0:1, 0:1] = self.alpha[2]*self.kernel_3_2[:,:,0:1,0:1]
                self._merge_weight[:, :, 0:1, 5:6] = self.alpha[2]*self.kernel_3_2[:,:,0:1,1:2]
                self._merge_weight[:, :, 0:1, 10:11] = self.alpha[2]*self.kernel_3_2[:,:,0:1,2:3]
                self._merge_weight[:, :, 5:6, 0:1] = self.alpha[2]*self.kernel_3_2[:,:,1:2,0:1]
                self._merge_weight[:, :, 5:6, 5:6] += self.alpha[2]*self.kernel_3_2[:,:,1:2,1:2]
                self._merge_weight[:, :, 5:6, 10:11] = self.alpha[2]*self.kernel_3_2[:,:,1:2,2:3]
                self._merge_weight[:, :, 10:11, 0:1] = self.alpha[2]*self.kernel_3_2[:,:,2:3,0:1]
                self._merge_weight[:, :, 10:11, 5:6] = self.alpha[2]*self.kernel_3_2[:,:,2:3,1:2]
                self._merge_weight[:, :, 10:11, 10:11] = self.alpha[2]*self.kernel_3_2[:,:,2:3,2:3]

            out = DepthwiseFunction.apply(h, self._merge_weight, None, 11//2, 11//2, False)

            return out

class StructureAwareSSM(nn.Module):
    def __init__(
            self,
            d_model=96,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
        self.x_proj_weight = nn.Parameter(self.x_proj.weight)
        del self.x_proj

        self.dt_projs = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                                     **factory_kwargs)
        self.dt_projs_weight = nn.Parameter(self.dt_projs.weight)
        self.dt_projs_bias = nn.Parameter(self.dt_projs.bias)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, dt_init)
        self.Ds = self.D_init(self.d_inner, dt_init)

        self.selective_scan = selective_scan_fn

        self.state_fusion = StateFusion(self.d_inner)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                bias=True, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=bias, **factory_kwargs)

        if bias:
            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            dt = torch.exp(
                torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))

            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)
            # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
            dt_proj.bias._no_reinit = True

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        elif dt_init == "simple":
            with torch.no_grad():
                dt_proj.weight.copy_(0.1 * torch.randn((d_inner, dt_rank)))
                dt_proj.bias.copy_(0.1 * torch.randn((d_inner)))
                dt_proj.bias._no_reinit = True
        elif dt_init == "zero":
            with torch.no_grad():
                dt_proj.weight.copy_(0.1 * torch.rand((d_inner, dt_rank)))
                dt_proj.bias.copy_(0.1 * torch.rand((d_inner)))
                dt_proj.bias._no_reinit = True
        else:
            raise NotImplementedError

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, init, device=None):
        if init == "random" or "constant":
            # S4D real initialization
            A = repeat(
                torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=d_inner,
            ).contiguous()
            A_log = torch.log(A)
            A_log = nn.Parameter(A_log)
            A_log._no_weight_decay = True
        elif init == "simple":
            A_log = nn.Parameter(torch.randn((d_inner, d_state)))
        elif init == "zero":
            A_log = nn.Parameter(torch.zeros((d_inner, d_state)))
        else:
            raise NotImplementedError
        return A_log

    @staticmethod
    def D_init(d_inner, init="random", device=None):
        if init == "random" or "constant":
            # D "skip" parameter
            D = torch.ones(d_inner, device=device)
            D = nn.Parameter(D)
            D._no_weight_decay = True
        elif init == "simple" or "zero":
            D = nn.Parameter(torch.ones(d_inner))
        else:
            raise NotImplementedError
        return D

    def ssm(self, x: torch.Tensor):
        B, C, L = x.shape
        # L = H * W

        xs = x.view(B, -1, L)  # b d l

        x_dbl = torch.matmul(self.x_proj_weight.view(1, -1, C), xs)  # b D d, b d l -> b D l
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=1)
        dts = torch.matmul(self.dt_projs_weight.view(1, C, -1), dts)

        As = -torch.exp(self.A_logs)
        Ds = self.Ds
        dts = dts.contiguous()
        dt_projs_bias = self.dt_projs_bias
        #
        # print(f"xs shape : {xs.shape}, dts shape : {dts.shape}, As shape : {As.shape}, Bs shape : {Bs.shape}, Cs shape : {Cs.shape}, Ds shape : {Ds.shape}, dt_projs_bias shape : {dt_projs_bias.shape}")
        # xs
        # shape: torch.Size([32, 128, 4131]), dts
        # shape: torch.Size([32, 128, 4131]), As
        # shape: torch.Size([128, 16]), Bs
        # shape: torch.Size([32, 16, 4131]), Cs
        # shape: torch.Size([32, 16, 4131]), Ds
        # shape: torch.Size([128]), dt_projs_bias
        # shape: torch.Size([128])
        h = self.selective_scan(
            xs, dts,
            As, Bs, None,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        )

        # h = rearrange(h, "b d s (h w) -> b (d s) h w", h=H, w=W)  # s=16    reshape to : b l d n
        h = rearrange(h, "b d s l -> b l d s")  # s=16    reshape to : b l d n
        # h = self.state_fusion(h)
        # h = rearrange(h, "b d h w -> b d (h w)")  # b d n l
        # Cs shape : b n l
        Cs = rearrange(Cs, "b dstate l -> b l dstate")  # b n l -> b l n
        y = (h @ Cs.unsqueeze(-1)).squeeze()  # b l n 1   # y = h * Cs  # 这里因为state = 1
        # print(y.shape)
        # y shape : b l d
        # y = h * Cs  # 这里因为state = 1   # cs shape : b n l   reshape to : b l n 1  y = h @ cs -> b l d
        y = rearrange(y, "b l d -> b d l")  # b l d -> b d l
        y = y + xs * Ds.view(-1, 1)   # xs shape : b d l, ds shape : d 1

        return y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        # print(x.shape)
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = rearrange(x, 'b h w d -> b d h w').contiguous()
        x = self.act(self.conv2d(x))
        x1 = rearrange(x, 'b d h w -> b d (h w)').contiguous()  # b d l
        # x2 = torch.flip(x1, dims=[-1])  # b d l
        # x3 = rearrange(x, 'b d h w -> b d (w h)').contiguous()  # b d w h
        # x4 = torch.flip(x3, dims=[-1])  # b d w h

        y1 = self.ssm(x1)  # h w
        # y2 = self.ssm(x2)
        # y3 = self.ssm(x3)
        # y4 = self.ssm(x4)

        # y2 = torch.flip(y2, dims=[-1])  # b d l
        # y4 = torch.flip(y4, dims=[-1])  # b d l
        # y11 = y1 + y2
        # y22 = y3 + y4
        # y11 = rearrange(y11, 'b d (h w) -> b d h w', h=H)  # b d l -> b l d
        # y22 = rearrange(y22, 'b d (w h) -> b d h w', h=H)  # b d l -> b l
        # y = y11 + y22
        y = rearrange(y1, 'b d (h w) -> b h w d', h=H).contiguous()  # b d l -> b l d
        # y = rearrange(y11+y22, 'b d (h w)-> b h w d', h=H, w=W)

        y = self.out_norm(y)
        y = y * F.silu(z)
        y = self.out_proj(y)
        if self.dropout is not None:
            y = self.dropout(y)
        return y

class BiSTSSMBlock_plus(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
            channel_first=False,
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",  # v2_plus_poselimbs
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            gmlp=False,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = StructureAwareSSM(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                expand=ssm_ratio,
                dt_rank=ssm_dt_rank,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate
            )

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                            drop=mlp_drop_rate, channels_first=channel_first)

    def _forward(self, input: torch.Tensor):
        x = input
        x = x + self.drop_path(self.op(self.norm(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)

class DepthwiseDCN(nn.Module):
    def __init__(self, d_inner, d_conv=3, conv_bias=True):
        super().__init__()
        self.d_inner = d_inner
        self.d_conv = d_conv

        # offset 通道数: 2 * kH * kW
        offset_channels = 2 * d_conv * d_conv

        # offset 层
        self.offset_conv = nn.Conv2d(
            in_channels=d_inner,
            out_channels=offset_channels,
            kernel_size=d_conv,
            stride=1,
            padding=(d_conv - 1) // 2,
            bias=True,

        )

        # Deformable Convolution (depthwise)
        self.dcn = DeformConv2d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            bias=conv_bias,
            groups=d_inner,  # depthwise

        )

    def forward(self, x):
        # 计算 offset
        offset = self.offset_conv(x)
        # deformable convolution
        out = self.dcn(x, offset)
        return out

class BiSTSSM_v3:
    def __initv2__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v2",
            channel_first=False,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear
        self.forward = self.forwardv2

        # tags for forward_type ==============================
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        if out_norm_none:
            self.out_norm = nn.Identity()
        elif out_norm_dwconv3:
            self.out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            self.out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            self.out_norm = nn.Sigmoid()
        else:
            LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm
            self.out_norm = LayerNorm(d_inner)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanMamba),
            # will be deleted in the future
            v02=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanMamba,
                        CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            v03=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanOflex,
                        CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            v04=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex,
                        CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            v05=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True,
                        CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            # ===============================
            v051d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True,
                          CrossScan=getCSM(1)[0], CrossMerge=getCSM(1)[1],
                          ),
            v052d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True,
                          CrossScan=getCSM(2)[0], CrossMerge=getCSM(2)[1],
                          ),
            v052dc=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True,
                           cascade2d=True),
            # ===============================
            v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanCore),
            v2_fs_ft=partial(self.forward_corev2, force_fp32=(not self.disable_force32), CrossScan=CrossScan_fs_ft,
                             SelectiveScan=SelectiveScanCore, CrossMerge=CrossMerge_fs_ft),
            v2_fs_bt=partial(self.forward_corev2, force_fp32=(not self.disable_force32), CrossScan=CrossScan_fs_bt,
                             SelectiveScan=SelectiveScanCore, CrossMerge=CrossMerge_fs_bt),
            v2_bs_ft=partial(self.forward_corev2, force_fp32=(not self.disable_force32), CrossScan=CrossScan_bs_ft,
                             SelectiveScan=SelectiveScanCore, CrossMerge=CrossMerge_bs_ft),
            v2_bs_bt=partial(self.forward_corev2, force_fp32=(not self.disable_force32), CrossScan=CrossScan_bs_bt,
                             SelectiveScan=SelectiveScanCore, CrossMerge=CrossMerge_bs_bt),
            v2_plus_poselimbs=partial(self.forward_corev2, force_fp32=(not self.disable_force32),
                                      CrossScan=CrossScan_plus_poselimbs, SelectiveScan=SelectiveScanCore,
                                      CrossMerge=CrossMerge_plus_poselimbs),
            v2_plus_poselimbs_v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32),
                                      CrossScan=CrossScan_plus_poselimbs, SelectiveScan=SelectiveScanOflex,
                                      CrossMerge=CrossMerge_plus_poselimbs),
            v2_plus_poselimbs_v3=partial(self.forward_corev2, force_fp32=(not self.disable_force32),
                                         CrossScan=CrossScan_plus_poselimbsv2, SelectiveScan=SelectiveScanOflex,
                                         CrossMerge=CrossMerge_plus_poselimbs),
            v2_plus_poselimbs_he = partial(self.forward_corev2, force_fp32=(not self.disable_force32),
                                       CrossScan=CrossScan_plus_poselimbs_he, SelectiveScan=SelectiveScanOflex,
                                       CrossMerge=CrossScan_plus_poselimbs_he),
            v3=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex),
            # v1=partial(self.forward_corev2, force_fp32=True, SelectiveScan=SelectiveScanOflex),
            # v4=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True, CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            # ===============================
            v31d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex,
                         CrossScan=CrossScan_Ab_1direction, CrossMerge=CrossMerge_Ab_1direction,
                         ),
            v32d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex,
                         CrossScan=CrossScan_Ab_2direction, CrossMerge=CrossMerge_Ab_2direction,
                         ),
            v32dc=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, cascade2d=True),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)
        k_group = 4  # 双向的或者单向的话 控制该值即可。 #todo.

        # in proj =======================================
        d_proj = d_inner if self.disable_z else (d_inner * 2)
        self.in_proj = Linear(d_model, d_proj, bias=bias)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,
                groups=d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            # self.gcn = GCN(d_inner, d_inner, 17)  # b t j c
            # self.conv2d = DepthwiseDCN(d_inner, d_conv=d_conv, conv_bias=conv_bias)

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            # dt proj ============================
            self.dt_projs = [
                self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
                for _ in range(k_group)
            ]
            self.dt_projs_weight = nn.Parameter(
                torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
            self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
            del self.dt_projs

            # A, D =======================================
            self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)
            self.Ds = self.D_init(d_inner, copies=k_group, merge=True)  # (K * D)
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(
                torch.randn((k_group * d_inner, d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((k_group, d_inner)))
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(
                torch.zeros((k_group * d_inner, d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, d_inner)))

    # 总是使用core v2
    def forward_corev2(
            self,
            x: torch.Tensor = None,
            # ==============================
            to_dtype=True,  # True: final out to dtype
            force_fp32=False,  # True: input fp32
            # ==============================
            ssoflex=True,  # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
            # ==============================
            SelectiveScan=SelectiveScanOflex,  # 使用SelectiveScanOflex
            CrossScan=CrossScan,  # CrossScan_plus_poselimbs
            CrossMerge=CrossMerge, # CrossMerge_plus_poselimbs
            no_einsum=False,  # replace einsum with linear or conv1d to raise throughput
            # ==============================
            cascade2d=False,
            **kwargs,
    ):
        # print(SelectiveScan, CrossScan, CrossMerge, cascade2d)
        # <class 'lib.model.csms6s.SelectiveScanCore'> <class 'lib.model.csms6s.CrossScan'> <class 'lib.model.csms6s.CrossMerge'> False
        x_proj_weight = self.x_proj_weight
        x_proj_bias = getattr(self, "x_proj_bias", None)
        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds
        delta_softplus = True
        out_norm = getattr(self, "out_norm", None)
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, -1, -1, ssoflex)

        # def selective_scan(u, delta, A, B, D=None, z=None, delta_bias=None, delta_softplus=False,
        #                       return_last_state=False):
        #     """if return_last_state is True, returns (out, last_state)
        #     last_state has shape (batch, dim, dstate). Note that the gradient of the last state is
        #     not considered in the backward pass.
        #     """
        #     return SelectiveScanStateFn.apply(u, delta, A, B, D, z, delta_bias, delta_softplus, return_last_state)

        if cascade2d:
            def scan_rowcol(
                    x: torch.Tensor,
                    proj_weight: torch.Tensor,
                    proj_bias: torch.Tensor,
                    dt_weight: torch.Tensor,
                    dt_bias: torch.Tensor,  # (2*c)
                    _As: torch.Tensor,  # As = -torch.exp(A_logs.to(torch.float))[:2,] # (2*c, d_state)
                    _Ds: torch.Tensor,
                    width=True,
            ):
                # x: (B, D, H, W)
                # proj_weight: (2 * D, (R+N+N))
                XB, XD, XH, XW = x.shape
                if width:
                    _B, _D, _L = XB * XH, XD, XW
                    xs = x.permute(0, 2, 1, 3).contiguous()
                else:
                    _B, _D, _L = XB * XW, XD, XH
                    xs = x.permute(0, 3, 1, 2).contiguous()
                xs = torch.stack([xs, xs.flip(dims=[-1])], dim=2)  # (B, H, 2, D, W)
                if no_einsum:
                    x_dbl = F.conv1d(xs.view(_B, -1, _L), proj_weight.view(-1, _D, 1),
                                     bias=(proj_bias.view(-1) if proj_bias is not None else None), groups=2)
                    dts, Bs, Cs = torch.split(x_dbl.view(_B, 2, -1, _L), [R, N, N], dim=2)
                    dts = F.conv1d(dts.contiguous().view(_B, -1, _L), dt_weight.view(2 * _D, -1, 1), groups=2)
                else:
                    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, proj_weight)
                    if x_proj_bias is not None:
                        x_dbl = x_dbl + x_proj_bias.view(1, 2, -1, 1)
                    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_weight)

                xs = xs.view(_B, -1, _L)
                dts = dts.contiguous().view(_B, -1, _L)
                As = _As.view(-1, N).to(torch.float)
                Bs = Bs.contiguous().view(_B, 2, N, _L)
                Cs = Cs.contiguous().view(_B, 2, N, _L)
                Ds = _Ds.view(-1)
                delta_bias = dt_bias.view(-1).to(torch.float)

                if force_fp32:
                    xs = xs.to(torch.float)
                dts = dts.to(xs.dtype)
                Bs = Bs.to(xs.dtype)
                Cs = Cs.to(xs.dtype)

                ys: torch.Tensor = selective_scan(
                    xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
                ).view(_B, 2, -1, _L)
                return ys

            As = -torch.exp(A_logs.to(torch.float)).view(4, -1, N)
            y_row = scan_rowcol(
                x,
                proj_weight=x_proj_weight.view(4, -1, D)[:2].contiguous(),
                proj_bias=(x_proj_bias.view(4, -1)[:2].contiguous() if x_proj_bias is not None else None),
                dt_weight=dt_projs_weight.view(4, D, -1)[:2].contiguous(),
                dt_bias=(dt_projs_bias.view(4, -1)[:2].contiguous() if dt_projs_bias is not None else None),
                _As=As[:2].contiguous().view(-1, N),
                _Ds=Ds.view(4, -1)[:2].contiguous().view(-1),
                width=True,
            ).view(B, H, 2, -1, W).sum(dim=2).permute(0, 2, 1, 3)
            y_col = scan_rowcol(
                y_row,
                proj_weight=x_proj_weight.view(4, -1, D)[2:].contiguous().to(y_row.dtype),
                proj_bias=(
                    x_proj_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if x_proj_bias is not None else None),
                dt_weight=dt_projs_weight.view(4, D, -1)[2:].contiguous().to(y_row.dtype),
                dt_bias=(
                    dt_projs_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if dt_projs_bias is not None else None),
                _As=As[2:].contiguous().view(-1, N),
                _Ds=Ds.view(4, -1)[2:].contiguous().view(-1),
                width=False,
            ).view(B, W, 2, -1, H).sum(dim=2).permute(0, 2, 3, 1)
            y = y_col
        else:
            xs = CrossScan.apply(x)
            if no_einsum:
                x_dbl = F.conv1d(xs.view(B, -1, L), x_proj_weight.view(-1, D, 1),
                                 bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
                dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
                dts = F.conv1d(dts.contiguous().view(B, -1, L), dt_projs_weight.view(K * D, -1, 1), groups=K)
            else:
                x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
                if x_proj_bias is not None:
                    x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
                dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

            xs = xs.view(B, -1, L)
            dts = dts.contiguous().view(B, -1, L)
            As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
            Bs = Bs.contiguous().view(B, K, N, L)
            Cs = Cs.contiguous().view(B, K, N, L)
            Ds = Ds.to(torch.float)  # (K * c)
            delta_bias = dt_projs_bias.view(-1).to(torch.float)

            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

            ys: torch.Tensor = selective_scan(
                xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
            ).view(B, K, -1, H, W)

            y: torch.Tensor = CrossMerge.apply(ys)

            if getattr(self, "__DEBUG__", False):
                setattr(self, "__data__", dict(
                    A_logs=A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                    us=xs, dts=dts, delta_bias=delta_bias,
                    ys=ys, y=y,
                ))

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)  # (B, L, C)
        y = out_norm(y)

        return (y.to(x.dtype) if to_dtype else y)



        #     xs = CrossScan.apply(x)
        #     if no_einsum:
        #         x_dbl = F.conv1d(xs.view(B, -1, L), x_proj_weight.view(-1, D, 1),
        #                          bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
        #         dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
        #         dts = F.conv1d(dts.contiguous().view(B, -1, L), dt_projs_weight.view(K * D, -1, 1), groups=K)
        #     else:
        #         x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
        #         if x_proj_bias is not None:
        #             x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        #         dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        #         dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
        #
        #     xs = xs.view(B, -1, L)
        #     dts = dts.contiguous().view(B, -1, L)
        #     As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
        #     Bs = Bs.contiguous().view(B, K, N, L)
        #     Cs = Cs.contiguous().view(B, K, N, L)
        #     Ds = Ds.to(torch.float)  # (K * c)
        #     delta_bias = dt_projs_bias.view(-1).to(torch.float)
        #
        #     if force_fp32:
        #         xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)
        #
        #     # todo进行修改, 然后在后边添加相关的图卷积操作。
        #     # xs shape: b kd l
        #     # dts shape : b kd l
        #     # as shape : kd state
        #     # bs shape : b state l 或者 b k state l
        #     h = selective_scan(
        #         xs, dts,
        #         As, Bs, None,
        #         z=None,
        #         delta_bias=delta_bias,
        #         delta_softplus=True,
        #         return_last_state=False,
        #     )  # h shape : b d k l
        #     # print(h.shape)  # 32,512,16,4131  b  kd state l,
        #
        #     h = rearrange(h, "b (k d) s l -> b k l d s", k=K, d=D)  # =16    reshape to : b l d n
        #     # h = self.state_fusion(h)
        #     # h = rearrange(h, "b d h w -> b d (h w)")  # b d n l
        #     # Cs shape : b n l
        #
        #     # Cs = rearrange(Cs, "b k s l -> b (k l) s")  #
        #     # ys = (h @ Cs.unsqueeze(-1)).squeeze()  # b (k l) d 1
        #     # ys = rearrange(ys, "b (k l) d -> b k d l", k=K)  # b k d l
        #     # ys = rearrange(ys, "b k d (h w) -> b k d h w", h=H, w=W)  # b k d h w
        #
        #     ys = h.mean(-1)
        #     ys = rearrange(ys, "b k (h w) d  -> b k d h w", h=H, w=W)  # b d k l
        #     # ys: torch.Tensor = selective_scan(
        #     #     xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
        #     # ).view(B, K, -1, H, W)
        #     # print(ys.shape)  # torch.Size([32, 4, 128, 243, 17])   b d k t v
        #     # pdb.set_trace()
        #     y: torch.Tensor = CrossMerge.apply(ys)
        #
        #     if getattr(self, "__DEBUG__", False):
        #         setattr(self, "__data__", dict(
        #             A_logs=A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
        #             us=xs, dts=dts, delta_bias=delta_bias,
        #             ys=ys, y=y,
        #         ))
        #
        # y = y.view(B, -1, H, W)
        # if not channel_first:
        #     y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)  # (B, L, C)
        # y = out_norm(y)
        #
        # return (y.to(x.dtype) if to_dtype else y)

    def forwardv2(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        # print(self.disable_z, self.disable_z_act, self.channel_first, self.with_dconv)
        # False False False True
        if not self.disable_z:
            # print(x.shape)
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))  # (b, h, w, d)
            # print(x.shape, z.shape)
            if not self.disable_z_act:
                z = self.act(z)
        if not self.channel_first:
            # print(x.shape)
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            # x = rearrange(x, 'b d h w -> b h w d').contiguous()
            # x = self.gcn(x)  # (b, d, h, w)
            # x = rearrange(x, 'b h w d -> b d h w').contiguous()  # b d h w
            x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)
        # torch.Size([1, 256, 243, 17])   input shape : n c t v
        y = self.forward_core(x)   # out shape : n t v c
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        # print('this is forwardv2')
        return out

class BiSTSSM_vv3(nn.Module, mamba_init, BiSTSSM_v3):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v2",   # v2_plus_poselimbs
            channel_first=False,
            # ======================
            **kwargs,
    ):
        super().__init__()
        kwargs.update(
            d_model=d_model, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, dt_init_floor=dt_init_floor,
            initialize=initialize, forward_type=forward_type, channel_first=channel_first,
        )
        self.__initv2__(**kwargs)

class BiSTSSMBlockv3(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
            channel_first=False,
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",  # v2_plus_poselimbs
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            gmlp=False,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = BiSTSSM_vv3(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                forward_type=forward_type,  # v2_plus_poselimbs
                channel_first=channel_first,
            )

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                            drop=mlp_drop_rate, channels_first=channel_first)

    def _forward(self, input: torch.Tensor):
        x = input
        x = x + self.drop_path(self.op(self.norm(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)

class to_channels_first(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)

class to_channels_last(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)

def build_norm_layer(dim,
                     norm_layer,
                     in_format='channels_last',
                     out_format='channels_last',
                     eps=1e-6):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(
            f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)


def build_act_layer(act_layer):
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()

    raise NotImplementedError(f'build_act_layer does not support {act_layer}')

class CenterFeatureScaleModule(nn.Module):
    def forward(self,
                query,
                center_feature_scale_proj_weight,
                center_feature_scale_proj_bias):
        center_feature_scale = F.linear(query,
                                        weight=center_feature_scale_proj_weight,
                                        bias=center_feature_scale_proj_bias).sigmoid()
        return center_feature_scale


class Dynamic_Adaptive_Scan(nn.Module):
    def __init__(
            self,
            channels=64,
            kernel_size=1,
            dw_kernel_size=3,
            stride=1,
            pad=0,
            dilation=1,
            group=1,
            offset_scale=1.0,
            act_layer='GELU',
            norm_layer='LN',
            center_feature_scale=False,
            remove_center=False,
    ):
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size


        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = (kernel_size-1)//2
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.center_feature_scale = center_feature_scale
        self.remove_center = int(remove_center)

        if self.remove_center and self.kernel_size % 2 == 0:
            raise ValueError('remove_center is only compatible with odd kernel size.')

        self.dw_conv = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=dw_kernel_size,
                stride=1,
                padding=(dw_kernel_size - 1) // 2,
                groups=channels),
            build_norm_layer(
                channels,
                norm_layer,
                'channels_first',
                'channels_last'),
            build_act_layer(act_layer))
        self.offset = nn.Linear(
            channels,
            group * (kernel_size * kernel_size - remove_center) * 2)  # 每个group 一个offset

        self._reset_parameters()

        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(
                torch.zeros((group, channels), dtype=torch.float))
            self.center_feature_scale_proj_bias = nn.Parameter(
                torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group, ))
            self.center_feature_scale_module = CenterFeatureScaleModule()

    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.)
        constant_(self.offset.bias.data, 0.)


    def forward(self, input, x):
        #  # input shape : n c t v;  x shape : n t v c
        N, _, H, W = input.shape   # input 为输入， x 为经过conv 之后的特征，局部特征。
        x_proj = x
        x1 = input
        x1 = self.dw_conv(x1)   # x1 shape : n t v c
        offset = self.offset(x1)  # shape : N, H, W, groups * (kernel_size * kernel_size - remove_center) * 2
        mask = torch.ones(N, H, W, self.group, device=x.device, dtype=x.dtype)
        # print(x.shape)
        x = DCNv3Function.apply(
            x, offset, mask,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale,
            256,
            self.remove_center)  # kernal size =1 的化，不会收集其他位置的信息，只会进行位置调整。

        if self.center_feature_scale:
            center_feature_scale = self.center_feature_scale_module(
                x1, self.center_feature_scale_proj_weight, self.center_feature_scale_proj_bias)
            # N, H, W, groups -> N, H, W, groups, 1 -> N, H, W, groups, _d_per_group -> N, H, W, channels
            center_feature_scale = center_feature_scale[..., None].repeat(
                1, 1, 1, 1, self.channels // self.group).flatten(-2)
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale

        x = x.permute(0, 3, 1, 2).contiguous()  # output : n c t v
        # print(x.shape)
        return x

# test dynamic adaptive scan
# input = torch.randn(1, 64, 243, 17).cuda()
# x = torch.randn(1, 243, 17, 64).cuda()
# net = Dynamic_Adaptive_Scan(channels=64, kernel_size=1).cuda()
# output = net(input, x)
# print(output.shape)   # 1 64 243 17
# pdb.set_trace()

class OffsetConv1D(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.offset_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=kernel_size,  # 1 offset per kernel position
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

    def forward(self, x):
        return self.offset_conv(x)

class OffsetNet(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        hidden_channels = 128
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, kernel_size, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.net(x)

class TransformerOffset1D(nn.Module):
    def __init__(self, in_channels, kernel_size, num_heads=8):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=in_channels, nhead=num_heads),
            num_layers=2
        )
        self.linear = nn.Linear(in_channels, kernel_size)

    def forward(self, x):
        # x: [B, C, T] → [T, B, C]
        x = x.permute(2, 0, 1)
        x = self.encoder(x)  # [T, B, C]
        x = x.permute(1, 2, 0)  # back to [B, C, T]
        offsets = self.linear(x.permute(0, 2, 1))  # [B, T, k]
        return offsets.permute(0, 2, 1)  # [B, k, T]


class JointStruct_Adaptive_Scan(nn.Module):
    # different frames have different offsets
    def __init__(
            self,
            channels=64,
            kernel_size=3,
            dw_kernel_size=3,
            stride=1,
            pad=0,
            dilation=1,
            group=1,   # here the group denotes different temporal group, not channel group, must be in [1, 3, 9, 27]
            offset_scale=1.0,
            act_layer='GELU',
            norm_layer='LN',
            center_feature_scale=False,
            remove_center=False,
    ):
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f'frames (27, 81, 243) must be divisible by groups (1, 3, 9, 27), but got {group}')
        local_num_frames = channels // group
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size


        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.center_feature_scale = center_feature_scale
        self.remove_center = int(remove_center)

        # local information:
        self.dwt_conv = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(dw_kernel_size,1),
                stride=1,
                padding=((dw_kernel_size - 1) // 2,0),
                groups=channels),
            build_norm_layer(
                channels,
                norm_layer,
                'channels_first',
                'channels_first'),
            build_act_layer(act_layer))

        # channel to offset x,y
        self.offset = nn.Linear(
            channels,
            kernel_size)  # 每个group 一个offset

        self._reset_parameters()
        self.dc1d = DeformConv1d(in_channels = channels, out_channels=channels, kernel_size=kernel_size,stride=stride,
                                 padding=pad, dilation=dilation, groups=group)

    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.)
        constant_(self.offset.bias.data, 0.)


    def forward(self, input, x):
        N, C, T, V = input.shape   # input 为原始输入， x 为经过gcn or conv 之后的特征，局部特征。 这里他们形状相同。
        x_proj = x
        # compute offset from orignal input
        x1 = input
        x1 = self.dwt_conv(x1)   # shape : n c t v  tcn, kernel=3 # todo: 改成 sp-gcn  或者直接stgcn
        x1 = rearrange(x1, 'n c t v -> (n t) v c')  # x1 shape : n t v c
        # offset shape : n t v 1  compute the offset of v
        offset = self.offset(x1)  # shape : Nt,v,k
        offset = rearrange(offset, 'nt v k -> nt k v 1')
        x = rearrange(x, 'n c t v -> (n t) c v')  # x shape : n t v c
        # offset
        # print(x.shape, offset.shape)
        x = self.dc1d(x, offset)  # x shape : nt c v
        x = rearrange(x, '(n t) c v -> n c t v', n=N, t=T)  # x shape : n c t v
        # kernal size =1 的化，不会收集其他位置的信息，只会进行位置调整。
        return x

# test dynamic adaptive scan
# input = torch.randn(1, 64, 243, 17).cuda()
# x = torch.randn(1, 243, 17, 64).cuda()
# net = Dynamic_Adaptive_Scan(channels=64, kernel_size=1).cuda()
# output = net(input, x)
# print(output.shape)   # 1 64 243 17
# pdb.set_trace()


class JointStruct_Adaptive_Scan_v2(nn.Module):
    # different frames have shared offsets
    def __init__(
            self,
            channels=64,
            kernel_size=1,
            dw_kernel_size=3,
            stride=1,
            pad=0,
            dilation=1,
            group=1,   # here the group denotes different temporal group, not channel group, must be in [1, 3, 9, 27]
            offset_scale=1.0,
            act_layer='GELU',
            norm_layer='LN',
            center_feature_scale=False,
            remove_center=False,
    ):
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f'frames (27, 81, 243) must be divisible by groups (1, 3, 9, 27), but got {group}')
        local_num_frames = channels // group
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size


        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.center_feature_scale = center_feature_scale
        self.remove_center = int(remove_center)

        # local information:
        self.dwt_conv = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(dw_kernel_size,1),
                stride=1,
                padding=((dw_kernel_size - 1) // 2,0),
                groups=channels),
            build_norm_layer(
                channels,
                norm_layer,
                'channels_first',
                'channels_first'),
            build_act_layer(act_layer))

        # channel to offset x,y
        self.offset = nn.Linear(
            channels,
            kernel_size)  # 每个group 一个offset

        self._reset_parameters()
        self.dc1d = DeformConv1d(in_channels = channels, out_channels=channels, kernel_size=kernel_size,stride=stride,
                                 padding=pad, dilation=dilation, groups=group)

    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.)
        constant_(self.offset.bias.data, 0.)


    def forward(self, input, x):
        N, C, T, V = input.shape   # input 为原始输入， x 为经过gcn or conv 之后的特征，局部特征。 这里他们形状相同。
        x_proj = x
        # compute offset from orignal input
        x1 = input
        x1 = self.dwt_conv(x1)   # shape : n c t v  tcn, kernel=3 # todo: 改成 sp-gcn  或者直接stgcn
        x1 = x1.mean(dim=2)         # 在时间维 T 上求均值 → shape: (N, C, V)
        x1 = rearrange(x1, 'n c v -> n v c')       # shape: N, V, C
        # offset shape : n t v 1  compute the offset of v
        offset = self.offset(x1)  # shape : Nt,v,k
        offset = offset.unsqueeze(1)  # shape: N, 1, V, K
        offset = offset.expand(-1, T, -1, -1)  # → shape: N, T, V, K
        offset = rearrange(offset, 'n t v k -> (n t) k v 1')
        x = rearrange(x, 'n c t v -> (n t) c v')
        x = self.dc1d(x, offset)
        x = rearrange(x, '(n t) c v -> n c t v', n=N, t=T)
        return x
# test dynamic adaptive scan
# input = torch.randn(1, 64, 243, 17).cuda()
# x = torch.randn(1, 64, 243, 17).cuda()
# net = JointStruct_Adaptive_Scan_v2(channels=64, kernel_size=1).cuda()
# output = net(input, x)
# print(output.shape)   # 1 64 243 17
# pdb.set_trace()

class D3D_STSSM_v2:
    def __initv2__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v3",
            channel_first=False,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        self.ssm_ratio = ssm_ratio
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear
        self.forward = self.forwardv2

        # tags for forward_type ==============================
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        if out_norm_none:
            self.out_norm = nn.Identity()
        elif out_norm_dwconv3:
            self.out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            self.out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            self.out_norm = nn.Sigmoid()
        else:
            LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm
            self.out_norm = LayerNorm(d_inner)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanMamba),
            # will be deleted in the future
            v02=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanMamba,
                        CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            v03=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanOflex,
                        CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            v04=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex,
                        CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            v05=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True,
                        CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            # ===============================
            v051d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True,
                          CrossScan=getCSM(1)[0], CrossMerge=getCSM(1)[1],
                          ),
            v052d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True,
                          CrossScan=getCSM(2)[0], CrossMerge=getCSM(2)[1],
                          ),
            v052dc=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True,
                           cascade2d=True),
            # ===============================
            v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanCore),
            v2_fs_ft=partial(self.forward_corev2, force_fp32=(not self.disable_force32), CrossScan=CrossScan_fs_ft,
                             SelectiveScan=SelectiveScanCore, CrossMerge=CrossMerge_fs_ft),
            v2_fs_bt=partial(self.forward_corev2, force_fp32=(not self.disable_force32), CrossScan=CrossScan_fs_bt,
                             SelectiveScan=SelectiveScanCore, CrossMerge=CrossMerge_fs_bt),
            v2_bs_ft=partial(self.forward_corev2, force_fp32=(not self.disable_force32), CrossScan=CrossScan_bs_ft,
                             SelectiveScan=SelectiveScanCore, CrossMerge=CrossMerge_bs_ft),
            v2_bs_bt=partial(self.forward_corev2, force_fp32=(not self.disable_force32), CrossScan=CrossScan_bs_bt,
                             SelectiveScan=SelectiveScanCore, CrossMerge=CrossMerge_bs_bt),
            v2_plus_poselimbs=partial(self.forward_corev2, force_fp32=(not self.disable_force32),
                                      CrossScan=CrossScan_plus_poselimbs, SelectiveScan=SelectiveScanCore,
                                      CrossMerge=CrossMerge_plus_poselimbs),

            # v2_four_direction 和 v3 等价
            v2_four_direction=partial(self.forward_corev2, force_fp32=(not self.disable_force32),
                                      CrossScan=CrossScan, SelectiveScan=SelectiveScanOflex,
                                      CrossMerge=CrossMerge),

            v3=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex),
            # v1=partial(self.forward_corev2, force_fp32=True, SelectiveScan=SelectiveScanOflex),
            # v4=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True, CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            # ===============================
            v31d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex,
                         CrossScan=CrossScan_Ab_1direction, CrossMerge=CrossMerge_Ab_1direction,
                         ),
            v32d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex,
                         CrossScan=CrossScan_Ab_2direction, CrossMerge=CrossMerge_Ab_2direction,
                         ),
            v32dc=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, cascade2d=True),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)
        k_group = 4

        # in proj =======================================
        d_proj = d_inner if self.disable_z else (d_inner * 2)
        self.in_proj = Linear(d_model, d_proj, bias=bias)
        self.offsetproj = Linear(d_model, d_inner, bias=bias)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,
                groups=d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            # dt proj ============================
            self.dt_projs = [
                self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
                for _ in range(k_group)
            ]
            self.dt_projs_weight = nn.Parameter(
                torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
            self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
            del self.dt_projs

            # A, D =======================================
            self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)
            self.Ds = self.D_init(d_inner, copies=k_group, merge=True)  # (K * D)
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(
                torch.randn((k_group * d_inner, d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((k_group, d_inner)))
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(
                torch.zeros((k_group * d_inner, d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, d_inner)))
        num_group = d_inner // d_state
        self.da_scan = Dynamic_Adaptive_Scan(channels=d_inner, group=num_group, kernel_size=1)  # 核心
    def forward_corev2(
            self,
            x: torch.Tensor = None,
            # ==============================
            to_dtype=True,  # True: final out to dtype
            force_fp32=False,  # True: input fp32
            # ==============================
            ssoflex=True,  # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
            # ==============================
            SelectiveScan=SelectiveScanOflex,
            CrossScan=CrossScan,
            CrossMerge=CrossMerge,
            no_einsum=False,  # replace einsum with linear or conv1d to raise throughput
            # ==============================
            cascade2d=False,
            **kwargs,
    ):
        # print(SelectiveScan, CrossScan, CrossMerge, cascade2d)
        # <class 'lib.model.csms6s.SelectiveScanCore'> <class 'lib.model.csms6s.CrossScan'> <class 'lib.model.csms6s.CrossMerge'> False
        x_proj_weight = self.x_proj_weight
        x_proj_bias = getattr(self, "x_proj_bias", None)
        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds
        delta_softplus = True
        out_norm = getattr(self, "out_norm", None)
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, -1, -1, ssoflex)

        if cascade2d:
            def scan_rowcol(
                    x: torch.Tensor,
                    proj_weight: torch.Tensor,
                    proj_bias: torch.Tensor,
                    dt_weight: torch.Tensor,
                    dt_bias: torch.Tensor,  # (2*c)
                    _As: torch.Tensor,  # As = -torch.exp(A_logs.to(torch.float))[:2,] # (2*c, d_state)
                    _Ds: torch.Tensor,
                    width=True,
            ):
                # x: (B, D, H, W)
                # proj_weight: (2 * D, (R+N+N))
                XB, XD, XH, XW = x.shape
                if width:
                    _B, _D, _L = XB * XH, XD, XW
                    xs = x.permute(0, 2, 1, 3).contiguous()
                else:
                    _B, _D, _L = XB * XW, XD, XH
                    xs = x.permute(0, 3, 1, 2).contiguous()
                xs = torch.stack([xs, xs.flip(dims=[-1])], dim=2)  # (B, H, 2, D, W)
                if no_einsum:
                    x_dbl = F.conv1d(xs.view(_B, -1, _L), proj_weight.view(-1, _D, 1),
                                     bias=(proj_bias.view(-1) if proj_bias is not None else None), groups=2)
                    dts, Bs, Cs = torch.split(x_dbl.view(_B, 2, -1, _L), [R, N, N], dim=2)
                    dts = F.conv1d(dts.contiguous().view(_B, -1, _L), dt_weight.view(2 * _D, -1, 1), groups=2)
                else:
                    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, proj_weight)
                    if x_proj_bias is not None:
                        x_dbl = x_dbl + x_proj_bias.view(1, 2, -1, 1)
                    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_weight)

                xs = xs.view(_B, -1, _L)
                dts = dts.contiguous().view(_B, -1, _L)
                As = _As.view(-1, N).to(torch.float)
                Bs = Bs.contiguous().view(_B, 2, N, _L)
                Cs = Cs.contiguous().view(_B, 2, N, _L)
                Ds = _Ds.view(-1)
                delta_bias = dt_bias.view(-1).to(torch.float)

                if force_fp32:
                    xs = xs.to(torch.float)
                dts = dts.to(xs.dtype)
                Bs = Bs.to(xs.dtype)
                Cs = Cs.to(xs.dtype)

                ys: torch.Tensor = selective_scan(
                    xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
                ).view(_B, 2, -1, _L)
                return ys

            As = -torch.exp(A_logs.to(torch.float)).view(4, -1, N)
            y_row = scan_rowcol(
                x,
                proj_weight=x_proj_weight.view(4, -1, D)[:2].contiguous(),
                proj_bias=(x_proj_bias.view(4, -1)[:2].contiguous() if x_proj_bias is not None else None),
                dt_weight=dt_projs_weight.view(4, D, -1)[:2].contiguous(),
                dt_bias=(dt_projs_bias.view(4, -1)[:2].contiguous() if dt_projs_bias is not None else None),
                _As=As[:2].contiguous().view(-1, N),
                _Ds=Ds.view(4, -1)[:2].contiguous().view(-1),
                width=True,
            ).view(B, H, 2, -1, W).sum(dim=2).permute(0, 2, 1, 3)
            y_col = scan_rowcol(
                y_row,
                proj_weight=x_proj_weight.view(4, -1, D)[2:].contiguous().to(y_row.dtype),
                proj_bias=(
                    x_proj_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if x_proj_bias is not None else None),
                dt_weight=dt_projs_weight.view(4, D, -1)[2:].contiguous().to(y_row.dtype),
                dt_bias=(
                    dt_projs_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if dt_projs_bias is not None else None),
                _As=As[2:].contiguous().view(-1, N),
                _Ds=Ds.view(4, -1)[2:].contiguous().view(-1),
                width=False,
            ).view(B, W, 2, -1, H).sum(dim=2).permute(0, 2, 3, 1)
            y = y_col
        else:
            xs = CrossScan.apply(x)
            if no_einsum:
                x_dbl = F.conv1d(xs.view(B, -1, L), x_proj_weight.view(-1, D, 1),
                                 bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
                dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
                dts = F.conv1d(dts.contiguous().view(B, -1, L), dt_projs_weight.view(K * D, -1, 1), groups=K)
            else:
                x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
                if x_proj_bias is not None:
                    x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
                dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

            xs = xs.view(B, -1, L)
            dts = dts.contiguous().view(B, -1, L)
            As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
            Bs = Bs.contiguous().view(B, K, N, L)
            Cs = Cs.contiguous().view(B, K, N, L)
            Ds = Ds.to(torch.float)  # (K * c)
            delta_bias = dt_projs_bias.view(-1).to(torch.float)

            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

            ys: torch.Tensor = selective_scan(
                xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
            ).view(B, K, -1, H, W)

            y: torch.Tensor = CrossMerge.apply(ys)

            if getattr(self, "__DEBUG__", False):
                setattr(self, "__data__", dict(
                    A_logs=A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                    us=xs, dts=dts, delta_bias=delta_bias,
                    ys=ys, y=y,
                ))

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)  # (B, L, C)
        y = out_norm(y)

        return (y.to(x.dtype) if to_dtype else y)

    def forwardv2(self, x: torch.Tensor, **kwargs):
        input = self.offsetproj(x) if self.ssm_ratio != 1.0 else x  # input shape: n t v c   # todo: 目前只能expand=1
        x = self.in_proj(x)
        # print(self.disable_z, self.disable_z_act, self.channel_first, self.with_dconv)
        # False False False True
        # print(self.disable_z)
        if not self.disable_z:
            # print(x.shape)
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
            # print(x.shape, z.shape)
            if not self.disable_z_act:
                z = self.act(z)     # n t v c
        if not self.channel_first:
            # print(x.shape)
            x = x.permute(0, 3, 1, 2).contiguous()   # n t v c -> n c t v
            input = input.permute(0, 3, 1, 2).contiguous()  # n t v c -> n c t v
        if self.with_dconv:
            x = self.conv2d(x)  # (b, d, h, w)   # n c t v
        x = self.act(x)
        # torch.Size([1, 256, 243, 17])   # n c t v
        # global offset
        # input of dascan: # input1 : n c t v;  input2 : n t v c
        # print(input.shape, x.shape)  # torch.Size([1, 64, 243, 17]) torch.Size([1, 128, 243, 17])

        x = self.da_scan(input, x.permute(0, 2, 3, 1).contiguous())+x  # output shape: n c t v
        y = self.forward_core(x)  # x shape n c t v   outputshape : n t v c
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        # print('this is forwardv2')
        return out   # n t v c


class D3D_STSSM(nn.Module, mamba_init, D3D_STSSM_v2):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v2",
        channel_first=False,
        # ======================
        **kwargs,
    ):
        super().__init__()
        kwargs.update(
            d_model=d_model, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, dt_init_floor=dt_init_floor,
            initialize=initialize, forward_type=forward_type, channel_first=channel_first,
        )
        self.__initv2__(**kwargs)

# test
# x = torch.randn(1, 243, 17, 64).cuda()
# net = D3D_STSSM(d_model=64, forward_type='v2_four_direction').cuda()   # todo: 单向 需要使用原始的ssm
# output = net(x)
# print(output.shape)   # 1,243,17,64
# pdb.set_trace()

class D3D_SSMBlock(nn.Module):
    # global dynamic adaptive SSM block
    # 不区分空间和时间
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
            channel_first=False,
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            gmlp=False,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = D3D_STSSM(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                forward_type=forward_type,
                channel_first=channel_first,
            )

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                            drop=mlp_drop_rate, channels_first=channel_first)
            # self.mlp = FMPL_v1(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
            #                    drop=mlp_drop_rate, channels_first=channel_first, v=17)
    def _forward(self, input: torch.Tensor):
        x = input
        x = x + self.drop_path(self.op(self.norm(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)

# test
# x = torch.randn(1, 243, 17, 64).cuda()
# net = D3D_SSMBlock(hidden_dim=64, forward_type='v2_four_direction').cuda()   # todo: 单向 需要使用原始的ssm
# output = net(x)
# print(output.shape)   # 1,243,17,64
# pdb.set_trace()

class D3D_SSMBlock_dffv(nn.Module):
    # global dynamic adaptive SSM block
    # 不区分空间和时间
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
            channel_first=False,
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            gmlp=False,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = D3D_STSSM(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                forward_type=forward_type,
                channel_first=channel_first,
            )

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            # self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
            #                 drop=mlp_drop_rate, channels_first=channel_first)
            self.mlp = FMPL_v1(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                               drop=mlp_drop_rate, channels_first=channel_first, v=17)
    def _forward(self, input: torch.Tensor):
        x = input
        x = x + self.drop_path(self.op(self.norm(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)

# test
# x = torch.randn(1, 243, 17, 64).cuda()
# net = D3D_SSMBlock(hidden_dim=64, forward_type='v2_four_direction').cuda()   # todo: 单向 需要使用原始的ssm
# output = net(x)
# print(output.shape)   # 1,243,17,64
# pdb.set_trace()

class D3D_SSMBlock_dfft(nn.Module):
    # global dynamic adaptive SSM block
    # 不区分空间和时间
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
            channel_first=False,
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            gmlp=False,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = D3D_STSSM(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                forward_type=forward_type,
                channel_first=channel_first,
            )

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            # self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
            #                 drop=mlp_drop_rate, channels_first=channel_first)
            self.mlp = FMPL_v2(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                               drop=mlp_drop_rate, channels_first=channel_first, v=17)
    def _forward(self, input: torch.Tensor):
        x = input
        x = x + self.drop_path(self.op(self.norm(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)

# test
# x = torch.randn(1, 243, 17, 64).cuda()
# net = D3D_SSMBlock(hidden_dim=64, forward_type='v2_four_direction').cuda()   # todo: 单向 需要使用原始的ssm
# output = net(x)
# print(output.shape)   # 1,243,17,64
# pdb.set_trace()

class D3D_SSMBlock_dffvt(nn.Module):
    # global dynamic adaptive SSM block
    # 不区分空间和时间
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
            channel_first=False,
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            gmlp=False,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            freq_l=9,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = D3D_STSSM(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                forward_type=forward_type,
                channel_first=channel_first,
            )

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            # self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
            #                 drop=mlp_drop_rate, channels_first=channel_first)
            self.mlp = FMPL_combined(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                               drop=mlp_drop_rate, channels_first=channel_first, v=17, temporal_t=freq_l)
    def _forward(self, input: torch.Tensor):
        x = input
        x = x + self.drop_path(self.op(self.norm(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)

# test
# x = torch.randn(1, 243, 17, 64).cuda()
# net = D3D_SSMBlock(hidden_dim=64, forward_type='v2_four_direction').cuda()   # todo: 单向 需要使用原始的ssm
# output = net(x)
# print(output.shape)   # 1,243,17,64
# pdb.set_trace()
class D3D_SSMBlock_freqmlp(nn.Module):
    # global dynamic adaptive SSM block
    # 不区分空间和时间
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
            channel_first=False,
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            gmlp=False,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = D3D_STSSM(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                forward_type=forward_type,
                channel_first=channel_first,
            )

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            # self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
            #                 drop=mlp_drop_rate, channels_first=channel_first)
            self.mlp = FreqTMlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer)
    def _forward(self, input: torch.Tensor):
        x = input
        x = x + self.drop_path(self.op(self.norm(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)

# test
# x = torch.randn(1, 243, 17, 64).cuda()
# net = D3D_SSMBlock(hidden_dim=64, forward_type='v2_four_direction').cuda()   # todo: 单向 需要使用原始的ssm
# output = net(x)
# print(output.shape)   # 1,243,17,64
# pdb.set_trace()

class D3D_SSMBlock_freq_plus_temporal(nn.Module):
    # global dynamic adaptive SSM block
    # 不区分空间和时间
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
            channel_first=False,
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            gmlp=False,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = D3D_STSSM(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                forward_type=forward_type,
                channel_first=channel_first,
            )

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm1 = norm_layer(hidden_dim)
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp1 = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                            drop=mlp_drop_rate, channels_first=channel_first)
            self.mlp2 = FreqTMlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer)
    def _forward(self, input: torch.Tensor):
        # n t v c
        n,t,v,c = input.shape
        x = input
        x = x + self.drop_path(self.op(self.norm(x)))
        x1 = x[:,:t//2,::] + self.drop_path(self.mlp1(self.norm1(x[:,:t//2,::])))
        x2 = x[:,t//2:,:] + self.drop_path(self.mlp2(self.norm2(x[:,t//2:,:])))
        x = torch.cat([x1,x2], dim=1)
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)

# test
# x = torch.randn(1, 243, 17, 64).cuda()
# net = D3D_SSMBlock(hidden_dim=64, forward_type='v2_four_direction').cuda()   # todo: 单向 需要使用原始的ssm
# output = net(x)
# print(output.shape)   # 1,243,17,64
# pdb.set_trace()


class SD_STSSM_v2:
    # 四个方向扫描  通道 dynamic temporal-specific 与 D3D 区别是 offsets 只在空间上变化。也就是v 维度上进行。
    def __initv2__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v3",
            channel_first=False,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        self.ssm_ratio = ssm_ratio
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear
        self.forward = self.forwardv2

        # tags for forward_type ==============================
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        if out_norm_none:
            self.out_norm = nn.Identity()
        elif out_norm_dwconv3:
            self.out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            self.out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            self.out_norm = nn.Sigmoid()
        else:
            LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm
            self.out_norm = LayerNorm(d_inner)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanMamba),
            # will be deleted in the future
            v02=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanMamba,
                        CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            v03=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanOflex,
                        CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            v04=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex,
                        CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            v05=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True,
                        CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            # ===============================
            v051d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True,
                          CrossScan=getCSM(1)[0], CrossMerge=getCSM(1)[1],
                          ),
            v052d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True,
                          CrossScan=getCSM(2)[0], CrossMerge=getCSM(2)[1],
                          ),
            v052dc=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True,
                           cascade2d=True),
            # ===============================
            v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanCore),
            v2_fs_ft=partial(self.forward_corev2, force_fp32=(not self.disable_force32), CrossScan=CrossScan_fs_ft,
                             SelectiveScan=SelectiveScanCore, CrossMerge=CrossMerge_fs_ft),
            v2_fs_bt=partial(self.forward_corev2, force_fp32=(not self.disable_force32), CrossScan=CrossScan_fs_bt,
                             SelectiveScan=SelectiveScanCore, CrossMerge=CrossMerge_fs_bt),
            v2_bs_ft=partial(self.forward_corev2, force_fp32=(not self.disable_force32), CrossScan=CrossScan_bs_ft,
                             SelectiveScan=SelectiveScanCore, CrossMerge=CrossMerge_bs_ft),
            v2_bs_bt=partial(self.forward_corev2, force_fp32=(not self.disable_force32), CrossScan=CrossScan_bs_bt,
                             SelectiveScan=SelectiveScanCore, CrossMerge=CrossMerge_bs_bt),
            v2_plus_poselimbs=partial(self.forward_corev2, force_fp32=(not self.disable_force32),
                                      CrossScan=CrossScan_plus_poselimbs, SelectiveScan=SelectiveScanCore,
                                      CrossMerge=CrossMerge_plus_poselimbs),

            # v2_four_direction 和 v3 等价
            v2_four_direction=partial(self.forward_corev2, force_fp32=(not self.disable_force32),
                                      CrossScan=CrossScan, SelectiveScan=SelectiveScanOflex,
                                      CrossMerge=CrossMerge),
            # 只进行 st fb 两个方向的扫描。
            v2_st_fb_direction=partial(self.forward_corev2, force_fp32=(not self.disable_force32),
                                      CrossScan=CrossScan, SelectiveScan=SelectiveScanOflex,
                                      CrossMerge=CrossMerge),

            v3=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex),
            # v1=partial(self.forward_corev2, force_fp32=True, SelectiveScan=SelectiveScanOflex),
            # v4=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True, CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            # ===============================
            v31d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex,
                         CrossScan=CrossScan_Ab_1direction, CrossMerge=CrossMerge_Ab_1direction,
                         ),
            v32d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex,
                         CrossScan=CrossScan_Ab_2direction, CrossMerge=CrossMerge_Ab_2direction,
                         ),
            v32dc=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, cascade2d=True),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)
        k_group = 4

        # in proj =======================================
        d_proj = d_inner if self.disable_z else (d_inner * 2)
        self.in_proj = Linear(d_model, d_proj, bias=bias)
        self.offsetproj = Linear(d_model, d_inner, bias=bias)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,
                groups=d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            # dt proj ============================
            self.dt_projs = [
                self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
                for _ in range(k_group)
            ]
            self.dt_projs_weight = nn.Parameter(
                torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
            self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
            del self.dt_projs

            # A, D =======================================
            self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)
            self.Ds = self.D_init(d_inner, copies=k_group, merge=True)  # (K * D)
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(
                torch.randn((k_group * d_inner, d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((k_group, d_inner)))
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(
                torch.zeros((k_group * d_inner, d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, d_inner)))
        num_group = d_inner // d_state
        self.da_scan = JointStruct_Adaptive_Scan(channels=d_inner, group=num_group, kernel_size=1)  # 核心
    def forward_corev2(
            self,
            x: torch.Tensor = None,
            # ==============================
            to_dtype=True,  # True: final out to dtype
            force_fp32=False,  # True: input fp32
            # ==============================
            ssoflex=True,  # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
            # ==============================
            SelectiveScan=SelectiveScanOflex,
            CrossScan=CrossScan,
            CrossMerge=CrossMerge,
            no_einsum=False,  # replace einsum with linear or conv1d to raise throughput
            # ==============================
            cascade2d=False,
            **kwargs,
    ):
        # print(SelectiveScan, CrossScan, CrossMerge, cascade2d)
        # <class 'lib.model.csms6s.SelectiveScanCore'> <class 'lib.model.csms6s.CrossScan'> <class 'lib.model.csms6s.CrossMerge'> False
        x_proj_weight = self.x_proj_weight
        x_proj_bias = getattr(self, "x_proj_bias", None)
        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds
        delta_softplus = True
        out_norm = getattr(self, "out_norm", None)
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, -1, -1, ssoflex)

        if cascade2d:
            def scan_rowcol(
                    x: torch.Tensor,
                    proj_weight: torch.Tensor,
                    proj_bias: torch.Tensor,
                    dt_weight: torch.Tensor,
                    dt_bias: torch.Tensor,  # (2*c)
                    _As: torch.Tensor,  # As = -torch.exp(A_logs.to(torch.float))[:2,] # (2*c, d_state)
                    _Ds: torch.Tensor,
                    width=True,
            ):
                # x: (B, D, H, W)
                # proj_weight: (2 * D, (R+N+N))
                XB, XD, XH, XW = x.shape
                if width:
                    _B, _D, _L = XB * XH, XD, XW
                    xs = x.permute(0, 2, 1, 3).contiguous()
                else:
                    _B, _D, _L = XB * XW, XD, XH
                    xs = x.permute(0, 3, 1, 2).contiguous()
                xs = torch.stack([xs, xs.flip(dims=[-1])], dim=2)  # (B, H, 2, D, W)
                if no_einsum:
                    x_dbl = F.conv1d(xs.view(_B, -1, _L), proj_weight.view(-1, _D, 1),
                                     bias=(proj_bias.view(-1) if proj_bias is not None else None), groups=2)
                    dts, Bs, Cs = torch.split(x_dbl.view(_B, 2, -1, _L), [R, N, N], dim=2)
                    dts = F.conv1d(dts.contiguous().view(_B, -1, _L), dt_weight.view(2 * _D, -1, 1), groups=2)
                else:
                    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, proj_weight)
                    if x_proj_bias is not None:
                        x_dbl = x_dbl + x_proj_bias.view(1, 2, -1, 1)
                    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_weight)

                xs = xs.view(_B, -1, _L)
                dts = dts.contiguous().view(_B, -1, _L)
                As = _As.view(-1, N).to(torch.float)
                Bs = Bs.contiguous().view(_B, 2, N, _L)
                Cs = Cs.contiguous().view(_B, 2, N, _L)
                Ds = _Ds.view(-1)
                delta_bias = dt_bias.view(-1).to(torch.float)

                if force_fp32:
                    xs = xs.to(torch.float)
                dts = dts.to(xs.dtype)
                Bs = Bs.to(xs.dtype)
                Cs = Cs.to(xs.dtype)

                ys: torch.Tensor = selective_scan(
                    xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
                ).view(_B, 2, -1, _L)
                return ys

            As = -torch.exp(A_logs.to(torch.float)).view(4, -1, N)
            y_row = scan_rowcol(
                x,
                proj_weight=x_proj_weight.view(4, -1, D)[:2].contiguous(),
                proj_bias=(x_proj_bias.view(4, -1)[:2].contiguous() if x_proj_bias is not None else None),
                dt_weight=dt_projs_weight.view(4, D, -1)[:2].contiguous(),
                dt_bias=(dt_projs_bias.view(4, -1)[:2].contiguous() if dt_projs_bias is not None else None),
                _As=As[:2].contiguous().view(-1, N),
                _Ds=Ds.view(4, -1)[:2].contiguous().view(-1),
                width=True,
            ).view(B, H, 2, -1, W).sum(dim=2).permute(0, 2, 1, 3)
            y_col = scan_rowcol(
                y_row,
                proj_weight=x_proj_weight.view(4, -1, D)[2:].contiguous().to(y_row.dtype),
                proj_bias=(
                    x_proj_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if x_proj_bias is not None else None),
                dt_weight=dt_projs_weight.view(4, D, -1)[2:].contiguous().to(y_row.dtype),
                dt_bias=(
                    dt_projs_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if dt_projs_bias is not None else None),
                _As=As[2:].contiguous().view(-1, N),
                _Ds=Ds.view(4, -1)[2:].contiguous().view(-1),
                width=False,
            ).view(B, W, 2, -1, H).sum(dim=2).permute(0, 2, 3, 1)
            y = y_col
        else:
            xs = CrossScan.apply(x)
            if no_einsum:
                x_dbl = F.conv1d(xs.view(B, -1, L), x_proj_weight.view(-1, D, 1),
                                 bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
                dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
                dts = F.conv1d(dts.contiguous().view(B, -1, L), dt_projs_weight.view(K * D, -1, 1), groups=K)
            else:
                x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
                if x_proj_bias is not None:
                    x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
                dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

            xs = xs.view(B, -1, L)
            dts = dts.contiguous().view(B, -1, L)
            As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
            Bs = Bs.contiguous().view(B, K, N, L)
            Cs = Cs.contiguous().view(B, K, N, L)
            Ds = Ds.to(torch.float)  # (K * c)
            delta_bias = dt_projs_bias.view(-1).to(torch.float)

            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

            ys: torch.Tensor = selective_scan(
                xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
            ).view(B, K, -1, H, W)

            y: torch.Tensor = CrossMerge.apply(ys)

            if getattr(self, "__DEBUG__", False):
                setattr(self, "__data__", dict(
                    A_logs=A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                    us=xs, dts=dts, delta_bias=delta_bias,
                    ys=ys, y=y,
                ))

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)  # (B, L, C)
        y = out_norm(y)

        return (y.to(x.dtype) if to_dtype else y)

    def forwardv2(self, x: torch.Tensor, **kwargs):
        input = self.offsetproj(x) if self.ssm_ratio != 1.0 else x  # input shape: n t v c   # todo: 目前只能expand=1
        x = self.in_proj(x)
        # print(self.disable_z, self.disable_z_act, self.channel_first, self.with_dconv)
        # False False False True
        # print(self.disable_z)
        if not self.disable_z:
            # print(x.shape)
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
            # print(x.shape, z.shape)
            if not self.disable_z_act:
                z = self.act(z)     # n t v c
        if not self.channel_first:
            # print(x.shape)
            x = x.permute(0, 3, 1, 2).contiguous()   # n t v c -> n c t v
            input = input.permute(0, 3, 1, 2).contiguous()  # n t v c -> n c t v
        if self.with_dconv:
            x = self.conv2d(x)  #   # n c t v
        x = self.act(x)
        # torch.Size([1, 256, 243, 17])   # n c t v
        # global offset
        # input of dascan: # input1 : n c t v;  input2 : n t v c
        # print(input.shape, x.shape)  # torch.Size([1, 64, 243, 17]) torch.Size([1, 128, 243, 17])
        x = self.da_scan(input, x) + x  #input, x shape : n c t v;  output shape: n c t v  # TODO 只进行 v offset
        y = self.forward_core(x)  # x shape n c t v   outputshape : n t v c   # todo 当进行 v offset 后，只进行时间前向， 时间后向的话需要重新 v offset
        # 空间动态扫描后， 1. 进行四方向实验， 2. 进行st 前向和后向， 单向。。  3. 时间动态扫描：ts 前向后向，单向。
        # 时间动态 + 空间动态，并联。  时间使用时间ffn增强，空间使用空间ffn 增强。
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        # print('this is forwardv2')
        return out   # n t v c


class SD_STSSM(nn.Module, mamba_init, SD_STSSM_v2):
    # 四个方向扫描  通道 dynamic channels-specific
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v2",
        channel_first=False,
        # ======================
        **kwargs,
    ):
        super().__init__()
        kwargs.update(
            d_model=d_model, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, dt_init_floor=dt_init_floor,
            initialize=initialize, forward_type=forward_type, channel_first=channel_first,
        )
        self.__initv2__(**kwargs)


class SD_SSMBlock(nn.Module):
    # Structure dynamic adaptive SSM block
    # 区分空间和时间 order # 四个方向扫描  通道 dynamic temporal-specific
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
            channel_first=False,
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            gmlp=False,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SD_STSSM(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                forward_type=forward_type,
                channel_first=channel_first,
            )

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                            drop=mlp_drop_rate, channels_first=channel_first)

    def _forward(self, input: torch.Tensor):
        x = input
        x = x + self.drop_path(self.op(self.norm(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)

# test
# x = torch.randn(1, 243, 17, 64).cuda()
# net = SD_SSMBlock(hidden_dim=64, forward_type='v2_four_direction').cuda()   # todo: 单向 需要使用原始的ssm
# output = net(x)
# print(output.shape)   # 1,243,17,64
# pdb.set_trace()

class SD_SSMBlock_freqT_MLP(nn.Module):
    # Structure dynamic adaptive SSM block
    # 区分空间和时间 order # 四个方向扫描  通道 dynamic temporal-specific
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
            channel_first=False,
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            gmlp=False,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SD_STSSM(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                forward_type=forward_type,
                channel_first=channel_first,
            )

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm1 = norm_layer(hidden_dim)
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = FreqTMlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer)

    def _forward(self, input: torch.Tensor):
        x = input
        x = x + self.drop_path(self.op(self.norm(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)

# test
# x = torch.randn(1, 243, 17, 64).cuda()
# net = SD_SSMBlock(hidden_dim=64, forward_type='v2_four_direction').cuda()   # todo: 单向 需要使用原始的ssm
# output = net(x)
# print(output.shape)   # 1,243,17,64
# pdb.set_trace()

class SD_SSMBlock_freqT_plusMLP(nn.Module):
    # Structure dynamic adaptive SSM block
    # 区分空间和时间 order # 四个方向扫描  通道 dynamic temporal-specific
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
            channel_first=False,
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            gmlp=False,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SD_STSSM(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                forward_type=forward_type,
                channel_first=channel_first,
            )

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm1 = norm_layer(hidden_dim)
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp1 = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                             drop=mlp_drop_rate, channels_first=channel_first)
            self.mlp2 = FreqTMlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer)

    def _forward(self, input: torch.Tensor):
        # n t v c
        n, t, v, c = input.shape
        x = input
        x = x + self.drop_path(self.op(self.norm(x)))
        x1 = x[:, :t // 2, ::] + self.drop_path(self.mlp1(self.norm1(x[:, :t // 2, ::])))
        x2 = x[:, t // 2:, :] + self.drop_path(self.mlp2(self.norm2(x[:, t // 2:, :])))
        x = torch.cat([x1, x2], dim=1)
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)

# test
# x = torch.randn(1, 243, 17, 64).cuda()
# net = SD_SSMBlock(hidden_dim=64, forward_type='v2_four_direction').cuda()   # todo: 单向 需要使用原始的ssm
# output = net(x)
# print(output.shape)   # 1,243,17,64
# pdb.set_trace()

class SD_SSMBlock_ddfv(nn.Module):
    # Structure dynamic adaptive SSM block
    # 区分空间和时间 order # 四个方向扫描  通道 dynamic temporal-specific
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
            channel_first=False,
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            gmlp=False,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SD_STSSM(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                forward_type=forward_type,
                channel_first=channel_first,
            )

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = FMPL_v1(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                               drop=mlp_drop_rate, channels_first=channel_first, v=17)

    def _forward(self, input: torch.Tensor):
        x = input
        x = x + self.drop_path(self.op(self.norm(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)

# test
# x = torch.randn(1, 243, 17, 64).cuda()
# net = SD_SSMBlock(hidden_dim=64, forward_type='v2_four_direction').cuda()   # todo: 单向 需要使用原始的ssm
# output = net(x)
# print(output.shape)   # 1,243,17,64
# pdb.set_trace()

class SD_SSMBlock_ddft(nn.Module):
    # Structure dynamic adaptive SSM block
    # 区分空间和时间 order # 四个方向扫描  通道 dynamic temporal-specific
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
            channel_first=False,
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            gmlp=False,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SD_STSSM(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                forward_type=forward_type,
                channel_first=channel_first,
            )

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = FMPL_v2(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                               drop=mlp_drop_rate, channels_first=channel_first, v=17)

    def _forward(self, input: torch.Tensor):
        x = input
        x = x + self.drop_path(self.op(self.norm(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)

# test
# x = torch.randn(1, 243, 17, 64).cuda()
# net = SD_SSMBlock(hidden_dim=64, forward_type='v2_four_direction').cuda()   # todo: 单向 需要使用原始的ssm
# output = net(x)
# print(output.shape)   # 1,243,17,64
# pdb.set_trace()

class SD_SSMBlock_ddftv(nn.Module):
    # Structure dynamic adaptive SSM block
    # 区分空间和时间 order # 四个方向扫描  通道 dynamic temporal-specific
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
            channel_first=False,
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            gmlp=False,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SD_STSSM(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                forward_type=forward_type,
                channel_first=channel_first,
            )

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = FMPL_combined(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                                     drop=mlp_drop_rate, channels_first=channel_first, v=17, temporal_t=9)

    def _forward(self, input: torch.Tensor):
        x = input
        x = x + self.drop_path(self.op(self.norm(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)

# test
# x = torch.randn(1, 243, 17, 64).cuda()
# net = SD_SSMBlock(hidden_dim=64, forward_type='v2_four_direction').cuda()   # todo: 单向 需要使用原始的ssm
# output = net(x)
# print(output.shape)   # 1,243,17,64
# pdb.set_trace()

class SD_STSSM_v3:
    # 只进行两个方向的扫描, dynamic 依然是temporal-specific
    def __initv2__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v2_ds_fb_direction",
            channel_first=False,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        self.ssm_ratio = ssm_ratio
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear
        self.forward = self.forwardv2

        # tags for forward_type ==============================
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        if out_norm_none:
            self.out_norm = nn.Identity()
        elif out_norm_dwconv3:
            self.out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            self.out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            self.out_norm = nn.Sigmoid()
        else:
            LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm
            self.out_norm = LayerNorm(d_inner)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanMamba),
            # will be deleted in the future
            v02=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanMamba,
                        CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            v03=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanOflex,
                        CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            v04=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex,
                        CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            v05=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True,
                        CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            # ===============================
            v051d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True,
                          CrossScan=getCSM(1)[0], CrossMerge=getCSM(1)[1],
                          ),
            v052d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True,
                          CrossScan=getCSM(2)[0], CrossMerge=getCSM(2)[1],
                          ),
            v052dc=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True,
                           cascade2d=True),
            # ===============================
            v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanCore),
            v2_fs_ft=partial(self.forward_corev2, force_fp32=(not self.disable_force32), CrossScan=CrossScan_fs_ft,
                             SelectiveScan=SelectiveScanCore, CrossMerge=CrossMerge_fs_ft),
            v2_fs_bt=partial(self.forward_corev2, force_fp32=(not self.disable_force32), CrossScan=CrossScan_fs_bt,
                             SelectiveScan=SelectiveScanCore, CrossMerge=CrossMerge_fs_bt),
            v2_bs_ft=partial(self.forward_corev2, force_fp32=(not self.disable_force32), CrossScan=CrossScan_bs_ft,
                             SelectiveScan=SelectiveScanCore, CrossMerge=CrossMerge_bs_ft),
            v2_bs_bt=partial(self.forward_corev2, force_fp32=(not self.disable_force32), CrossScan=CrossScan_bs_bt,
                             SelectiveScan=SelectiveScanCore, CrossMerge=CrossMerge_bs_bt),
            v2_plus_poselimbs=partial(self.forward_corev2, force_fp32=(not self.disable_force32),
                                      CrossScan=CrossScan_plus_poselimbs, SelectiveScan=SelectiveScanCore,
                                      CrossMerge=CrossMerge_plus_poselimbs),

            # v2_four_direction 和 v3 等价
            v2_four_direction=partial(self.forward_corev2, force_fp32=(not self.disable_force32),
                                      CrossScan=CrossScan, SelectiveScan=SelectiveScanOflex,
                                      CrossMerge=CrossMerge),
            # 只进行 st fb 两个方向的扫描。
            v2_ds_fb_direction=partial(self.forward_corev2, force_fp32=(not self.disable_force32),
                                      CrossScan=CrossScan_ds_fb, SelectiveScan=SelectiveScanOflex,
                                      CrossMerge=CrossMerge_ds_fb),   # CrossScan_ds_fb, CrossMerge_ds_fb
            # 只进行 st f 前向扫描
            v2_st_f_direction=partial(self.forward_corev2, force_fp32=(not self.disable_force32),
                                       CrossScan=CrossScan, SelectiveScan=SelectiveScanOflex,
                                       CrossMerge=CrossMerge),

            v3=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex),
            # v1=partial(self.forward_corev2, force_fp32=True, SelectiveScan=SelectiveScanOflex),
            # v4=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True, CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            # ===============================
            v31d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex,
                         CrossScan=CrossScan_Ab_1direction, CrossMerge=CrossMerge_Ab_1direction,
                         ),
            v32d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex,
                         CrossScan=CrossScan_Ab_2direction, CrossMerge=CrossMerge_Ab_2direction,
                         ),
            v32dc=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, cascade2d=True),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)
        k_group = 2   # k_group = 2

        # in proj =======================================
        d_proj = d_inner if self.disable_z else (d_inner * 2)
        self.in_proj = Linear(d_model, d_proj, bias=bias)
        self.offsetproj = Linear(d_model, d_inner, bias=bias)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,
                groups=d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            # dt proj ============================
            self.dt_projs = [
                self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
                for _ in range(k_group)
            ]
            self.dt_projs_weight = nn.Parameter(
                torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
            self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
            del self.dt_projs

            # A, D =======================================
            self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)
            self.Ds = self.D_init(d_inner, copies=k_group, merge=True)  # (K * D)
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(
                torch.randn((k_group * d_inner, d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((k_group, d_inner)))
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(
                torch.zeros((k_group * d_inner, d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, d_inner)))
        num_group = d_inner // d_state
        self.da_scan = JointStruct_Adaptive_Scan(channels=d_inner, group=num_group, kernel_size=1)  # 核心
    def forward_corev2(
            self,
            x: torch.Tensor = None,
            # ==============================
            to_dtype=True,  # True: final out to dtype
            force_fp32=False,  # True: input fp32
            # ==============================
            ssoflex=True,  # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
            # ==============================
            SelectiveScan=SelectiveScanOflex,
            CrossScan=CrossScan,
            CrossMerge=CrossMerge,
            no_einsum=False,  # replace einsum with linear or conv1d to raise throughput
            # ==============================
            cascade2d=False,
            **kwargs,
    ):
        # print(SelectiveScan, CrossScan, CrossMerge, cascade2d)
        # <class 'lib.model.csms6s.SelectiveScanCore'> <class 'lib.model.csms6s.CrossScan'> <class 'lib.model.csms6s.CrossMerge'> False
        x_proj_weight = self.x_proj_weight
        x_proj_bias = getattr(self, "x_proj_bias", None)
        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds
        delta_softplus = True
        out_norm = getattr(self, "out_norm", None)
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, -1, -1, ssoflex)

        if cascade2d:
            def scan_rowcol(
                    x: torch.Tensor,
                    proj_weight: torch.Tensor,
                    proj_bias: torch.Tensor,
                    dt_weight: torch.Tensor,
                    dt_bias: torch.Tensor,  # (2*c)
                    _As: torch.Tensor,  # As = -torch.exp(A_logs.to(torch.float))[:2,] # (2*c, d_state)
                    _Ds: torch.Tensor,
                    width=True,
            ):
                # x: (B, D, H, W)
                # proj_weight: (2 * D, (R+N+N))
                XB, XD, XH, XW = x.shape
                if width:
                    _B, _D, _L = XB * XH, XD, XW
                    xs = x.permute(0, 2, 1, 3).contiguous()
                else:
                    _B, _D, _L = XB * XW, XD, XH
                    xs = x.permute(0, 3, 1, 2).contiguous()
                xs = torch.stack([xs, xs.flip(dims=[-1])], dim=2)  # (B, H, 2, D, W)
                if no_einsum:
                    x_dbl = F.conv1d(xs.view(_B, -1, _L), proj_weight.view(-1, _D, 1),
                                     bias=(proj_bias.view(-1) if proj_bias is not None else None), groups=2)
                    dts, Bs, Cs = torch.split(x_dbl.view(_B, 2, -1, _L), [R, N, N], dim=2)
                    dts = F.conv1d(dts.contiguous().view(_B, -1, _L), dt_weight.view(2 * _D, -1, 1), groups=2)
                else:
                    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, proj_weight)
                    if x_proj_bias is not None:
                        x_dbl = x_dbl + x_proj_bias.view(1, 2, -1, 1)
                    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_weight)

                xs = xs.view(_B, -1, _L)
                dts = dts.contiguous().view(_B, -1, _L)
                As = _As.view(-1, N).to(torch.float)
                Bs = Bs.contiguous().view(_B, 2, N, _L)
                Cs = Cs.contiguous().view(_B, 2, N, _L)
                Ds = _Ds.view(-1)
                delta_bias = dt_bias.view(-1).to(torch.float)

                if force_fp32:
                    xs = xs.to(torch.float)
                dts = dts.to(xs.dtype)
                Bs = Bs.to(xs.dtype)
                Cs = Cs.to(xs.dtype)

                ys: torch.Tensor = selective_scan(
                    xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
                ).view(_B, 2, -1, _L)
                return ys

            As = -torch.exp(A_logs.to(torch.float)).view(4, -1, N)
            y_row = scan_rowcol(
                x,
                proj_weight=x_proj_weight.view(4, -1, D)[:2].contiguous(),
                proj_bias=(x_proj_bias.view(4, -1)[:2].contiguous() if x_proj_bias is not None else None),
                dt_weight=dt_projs_weight.view(4, D, -1)[:2].contiguous(),
                dt_bias=(dt_projs_bias.view(4, -1)[:2].contiguous() if dt_projs_bias is not None else None),
                _As=As[:2].contiguous().view(-1, N),
                _Ds=Ds.view(4, -1)[:2].contiguous().view(-1),
                width=True,
            ).view(B, H, 2, -1, W).sum(dim=2).permute(0, 2, 1, 3)
            y_col = scan_rowcol(
                y_row,
                proj_weight=x_proj_weight.view(4, -1, D)[2:].contiguous().to(y_row.dtype),
                proj_bias=(
                    x_proj_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if x_proj_bias is not None else None),
                dt_weight=dt_projs_weight.view(4, D, -1)[2:].contiguous().to(y_row.dtype),
                dt_bias=(
                    dt_projs_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if dt_projs_bias is not None else None),
                _As=As[2:].contiguous().view(-1, N),
                _Ds=Ds.view(4, -1)[2:].contiguous().view(-1),
                width=False,
            ).view(B, W, 2, -1, H).sum(dim=2).permute(0, 2, 3, 1)
            y = y_col
        else:
            xs = CrossScan.apply(x)
            if no_einsum:
                x_dbl = F.conv1d(xs.view(B, -1, L), x_proj_weight.view(-1, D, 1),
                                 bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
                dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
                dts = F.conv1d(dts.contiguous().view(B, -1, L), dt_projs_weight.view(K * D, -1, 1), groups=K)
            else:
                x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
                if x_proj_bias is not None:
                    x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
                dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

            xs = xs.view(B, -1, L)
            dts = dts.contiguous().view(B, -1, L)
            As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
            Bs = Bs.contiguous().view(B, K, N, L)
            Cs = Cs.contiguous().view(B, K, N, L)
            Ds = Ds.to(torch.float)  # (K * c)
            delta_bias = dt_projs_bias.view(-1).to(torch.float)

            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

            ys: torch.Tensor = selective_scan(
                xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
            ).view(B, K, -1, H, W)
            # print(ys.shape)
            y: torch.Tensor = CrossMerge.apply(ys)

            if getattr(self, "__DEBUG__", False):
                setattr(self, "__data__", dict(
                    A_logs=A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                    us=xs, dts=dts, delta_bias=delta_bias,
                    ys=ys, y=y,
                ))
        # print(y.shape)
        y = y.view(B, -1, H, W)
        # print(y.shape)
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)  # (B, L, C)
        # print(y.shape)
        y = out_norm(y)

        return (y.to(x.dtype) if to_dtype else y)

    def forwardv2(self, x: torch.Tensor, **kwargs):
        input = self.offsetproj(x) if self.ssm_ratio != 1.0 else x  # input shape: n t v c   # todo: 目前只能expand=1
        x = self.in_proj(x)
        # print(self.disable_z, self.disable_z_act, self.channel_first, self.with_dconv)
        # False False False True
        # print(self.disable_z)
        if not self.disable_z:
            # print(x.shape)
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
            # print(x.shape, z.shape)
            if not self.disable_z_act:
                z = self.act(z)     # n t v c
        if not self.channel_first:
            # print(x.shape)
            x = x.permute(0, 3, 1, 2).contiguous()   # n t v c -> n c t v
            input = input.permute(0, 3, 1, 2).contiguous()  # n t v c -> n c t v
        if self.with_dconv:
            x = self.conv2d(x)  #   # n c t v
        x = self.act(x)
        # torch.Size([1, 256, 243, 17])   # n c t v
        # global offset
        # input of dascan: # input1 : n c t v;  input2 : n t v c
        # print(input.shape, x.shape)  # torch.Size([1, 64, 243, 17]) torch.Size([1, 128, 243, 17])
        x = self.da_scan(input, x) + x  #input, x shape : n c t v;  output shape: n c t v  # TODO 只进行 v offset
        y = self.forward_core(x)  # x shape n c t v   outputshape : n t v c   # todo 当进行 v offset 后，只进行时间前向， 时间后向的话需要重新 v offset
        # 空间动态扫描后， 1. 进行四方向实验， 2. 进行st 前向和后向， 单向。。  3. 时间动态扫描：ts 前向后向，单向。
        # 时间动态 + 空间动态，并联。  时间使用时间ffn增强，空间使用空间ffn 增强。
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        # print('this is forwardv2')
        return out   # n t v c

class SD_STSSM_vv3(nn.Module, mamba_init, SD_STSSM_v3):
    # 只进行两个方向的扫描, dynamic
    # 依然是temporal-specific
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v2_ds_fb_direction",
        channel_first=False,
        # ======================
        **kwargs,
    ):
        super().__init__()
        kwargs.update(
            d_model=d_model, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, dt_init_floor=dt_init_floor,
            initialize=initialize, forward_type=forward_type, channel_first=channel_first,
        )
        self.__initv2__(**kwargs)


class SD_SSMBlock_v3(nn.Module):
    # Structure dynamic adaptive SSM block
    # 区分空间和时间 order 只进行两个方向的扫描, dynamic 依然是temporal-specific
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
            channel_first=False,
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2_ds_fb_direction",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            gmlp=False,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SD_STSSM_vv3(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                forward_type=forward_type,
                channel_first=channel_first,
            )

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                            drop=mlp_drop_rate, channels_first=channel_first)

    def _forward(self, input: torch.Tensor):
        x = input
        x = x + self.drop_path(self.op(self.norm(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)

# test
# x = torch.randn(1, 243, 17, 64).cuda()
# net = SD_SSMBlock_v3(hidden_dim=64, forward_type='v2_ds_fb_direction').cuda()   # todo: 单向 需要使用原始的ssm
# output = net(x)
# print(output.shape)   # 1,243,17,64
# pdb.set_trace()


class SD_STSSM_v4:
    # 4方向的扫描, dynamic是temporal-shared
    def __initv2__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v2_four_direction",
            channel_first=False,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        self.ssm_ratio = ssm_ratio
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear
        self.forward = self.forwardv2

        # tags for forward_type ==============================
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        if out_norm_none:
            self.out_norm = nn.Identity()
        elif out_norm_dwconv3:
            self.out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            self.out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            self.out_norm = nn.Sigmoid()
        else:
            LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm
            self.out_norm = LayerNorm(d_inner)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanMamba),
            # will be deleted in the future
            v02=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanMamba,
                        CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            v03=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanOflex,
                        CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            v04=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex,
                        CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            v05=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True,
                        CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            # ===============================
            v051d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True,
                          CrossScan=getCSM(1)[0], CrossMerge=getCSM(1)[1],
                          ),
            v052d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True,
                          CrossScan=getCSM(2)[0], CrossMerge=getCSM(2)[1],
                          ),
            v052dc=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True,
                           cascade2d=True),
            # ===============================
            v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanCore),
            v2_fs_ft=partial(self.forward_corev2, force_fp32=(not self.disable_force32), CrossScan=CrossScan_fs_ft,
                             SelectiveScan=SelectiveScanCore, CrossMerge=CrossMerge_fs_ft),
            v2_fs_bt=partial(self.forward_corev2, force_fp32=(not self.disable_force32), CrossScan=CrossScan_fs_bt,
                             SelectiveScan=SelectiveScanCore, CrossMerge=CrossMerge_fs_bt),
            v2_bs_ft=partial(self.forward_corev2, force_fp32=(not self.disable_force32), CrossScan=CrossScan_bs_ft,
                             SelectiveScan=SelectiveScanCore, CrossMerge=CrossMerge_bs_ft),
            v2_bs_bt=partial(self.forward_corev2, force_fp32=(not self.disable_force32), CrossScan=CrossScan_bs_bt,
                             SelectiveScan=SelectiveScanCore, CrossMerge=CrossMerge_bs_bt),
            v2_plus_poselimbs=partial(self.forward_corev2, force_fp32=(not self.disable_force32),
                                      CrossScan=CrossScan_plus_poselimbs, SelectiveScan=SelectiveScanCore,
                                      CrossMerge=CrossMerge_plus_poselimbs),

            # v2_four_direction 和 v3 等价
            v2_four_direction=partial(self.forward_corev2, force_fp32=(not self.disable_force32),
                                      CrossScan=CrossScan, SelectiveScan=SelectiveScanOflex,
                                      CrossMerge=CrossMerge),
            # 只进行 st fb 两个方向的扫描。
            v2_ds_fb_direction=partial(self.forward_corev2, force_fp32=(not self.disable_force32),
                                      CrossScan=CrossScan_ds_fb, SelectiveScan=SelectiveScanOflex,
                                      CrossMerge=CrossMerge_ds_fb),   # CrossScan_ds_fb, CrossMerge_ds_fb
            # 只进行 st f 前向扫描
            v2_st_f_direction=partial(self.forward_corev2, force_fp32=(not self.disable_force32),
                                       CrossScan=CrossScan, SelectiveScan=SelectiveScanOflex,
                                       CrossMerge=CrossMerge),

            v3=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex),
            # v1=partial(self.forward_corev2, force_fp32=True, SelectiveScan=SelectiveScanOflex),
            # v4=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True, CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            # ===============================
            v31d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex,
                         CrossScan=CrossScan_Ab_1direction, CrossMerge=CrossMerge_Ab_1direction,
                         ),
            v32d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex,
                         CrossScan=CrossScan_Ab_2direction, CrossMerge=CrossMerge_Ab_2direction,
                         ),
            v32dc=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, cascade2d=True),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)
        k_group = 4   # k_group = 2

        # in proj =======================================
        d_proj = d_inner if self.disable_z else (d_inner * 2)
        self.in_proj = Linear(d_model, d_proj, bias=bias)
        self.offsetproj = Linear(d_model, d_inner, bias=bias)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,
                groups=d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            # dt proj ============================
            self.dt_projs = [
                self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
                for _ in range(k_group)
            ]
            self.dt_projs_weight = nn.Parameter(
                torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
            self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
            del self.dt_projs

            # A, D =======================================
            self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)
            self.Ds = self.D_init(d_inner, copies=k_group, merge=True)  # (K * D)
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(
                torch.randn((k_group * d_inner, d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((k_group, d_inner)))
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(
                torch.zeros((k_group * d_inner, d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, d_inner)))
        num_group = d_inner // d_state
        self.da_scan = JointStruct_Adaptive_Scan_v2(channels=d_inner, group=num_group)  # 核心  v1 为temporal-spcefic v2: temporal-shared
    def forward_corev2(
            self,
            x: torch.Tensor = None,
            # ==============================
            to_dtype=True,  # True: final out to dtype
            force_fp32=False,  # True: input fp32
            # ==============================
            ssoflex=True,  # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
            # ==============================
            SelectiveScan=SelectiveScanOflex,
            CrossScan=CrossScan,
            CrossMerge=CrossMerge,
            no_einsum=False,  # replace einsum with linear or conv1d to raise throughput
            # ==============================
            cascade2d=False,
            **kwargs,
    ):
        # print(SelectiveScan, CrossScan, CrossMerge, cascade2d)
        # <class 'lib.model.csms6s.SelectiveScanCore'> <class 'lib.model.csms6s.CrossScan'> <class 'lib.model.csms6s.CrossMerge'> False
        x_proj_weight = self.x_proj_weight
        x_proj_bias = getattr(self, "x_proj_bias", None)
        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds
        delta_softplus = True
        out_norm = getattr(self, "out_norm", None)
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, -1, -1, ssoflex)

        if cascade2d:
            def scan_rowcol(
                    x: torch.Tensor,
                    proj_weight: torch.Tensor,
                    proj_bias: torch.Tensor,
                    dt_weight: torch.Tensor,
                    dt_bias: torch.Tensor,  # (2*c)
                    _As: torch.Tensor,  # As = -torch.exp(A_logs.to(torch.float))[:2,] # (2*c, d_state)
                    _Ds: torch.Tensor,
                    width=True,
            ):
                # x: (B, D, H, W)
                # proj_weight: (2 * D, (R+N+N))
                XB, XD, XH, XW = x.shape
                if width:
                    _B, _D, _L = XB * XH, XD, XW
                    xs = x.permute(0, 2, 1, 3).contiguous()
                else:
                    _B, _D, _L = XB * XW, XD, XH
                    xs = x.permute(0, 3, 1, 2).contiguous()
                xs = torch.stack([xs, xs.flip(dims=[-1])], dim=2)  # (B, H, 2, D, W)
                if no_einsum:
                    x_dbl = F.conv1d(xs.view(_B, -1, _L), proj_weight.view(-1, _D, 1),
                                     bias=(proj_bias.view(-1) if proj_bias is not None else None), groups=2)
                    dts, Bs, Cs = torch.split(x_dbl.view(_B, 2, -1, _L), [R, N, N], dim=2)
                    dts = F.conv1d(dts.contiguous().view(_B, -1, _L), dt_weight.view(2 * _D, -1, 1), groups=2)
                else:
                    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, proj_weight)
                    if x_proj_bias is not None:
                        x_dbl = x_dbl + x_proj_bias.view(1, 2, -1, 1)
                    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_weight)

                xs = xs.view(_B, -1, _L)
                dts = dts.contiguous().view(_B, -1, _L)
                As = _As.view(-1, N).to(torch.float)
                Bs = Bs.contiguous().view(_B, 2, N, _L)
                Cs = Cs.contiguous().view(_B, 2, N, _L)
                Ds = _Ds.view(-1)
                delta_bias = dt_bias.view(-1).to(torch.float)

                if force_fp32:
                    xs = xs.to(torch.float)
                dts = dts.to(xs.dtype)
                Bs = Bs.to(xs.dtype)
                Cs = Cs.to(xs.dtype)

                ys: torch.Tensor = selective_scan(
                    xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
                ).view(_B, 2, -1, _L)
                return ys

            As = -torch.exp(A_logs.to(torch.float)).view(4, -1, N)
            y_row = scan_rowcol(
                x,
                proj_weight=x_proj_weight.view(4, -1, D)[:2].contiguous(),
                proj_bias=(x_proj_bias.view(4, -1)[:2].contiguous() if x_proj_bias is not None else None),
                dt_weight=dt_projs_weight.view(4, D, -1)[:2].contiguous(),
                dt_bias=(dt_projs_bias.view(4, -1)[:2].contiguous() if dt_projs_bias is not None else None),
                _As=As[:2].contiguous().view(-1, N),
                _Ds=Ds.view(4, -1)[:2].contiguous().view(-1),
                width=True,
            ).view(B, H, 2, -1, W).sum(dim=2).permute(0, 2, 1, 3)
            y_col = scan_rowcol(
                y_row,
                proj_weight=x_proj_weight.view(4, -1, D)[2:].contiguous().to(y_row.dtype),
                proj_bias=(
                    x_proj_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if x_proj_bias is not None else None),
                dt_weight=dt_projs_weight.view(4, D, -1)[2:].contiguous().to(y_row.dtype),
                dt_bias=(
                    dt_projs_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if dt_projs_bias is not None else None),
                _As=As[2:].contiguous().view(-1, N),
                _Ds=Ds.view(4, -1)[2:].contiguous().view(-1),
                width=False,
            ).view(B, W, 2, -1, H).sum(dim=2).permute(0, 2, 3, 1)
            y = y_col
        else:
            xs = CrossScan.apply(x)
            if no_einsum:
                x_dbl = F.conv1d(xs.view(B, -1, L), x_proj_weight.view(-1, D, 1),
                                 bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
                dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
                dts = F.conv1d(dts.contiguous().view(B, -1, L), dt_projs_weight.view(K * D, -1, 1), groups=K)
            else:
                x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
                if x_proj_bias is not None:
                    x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
                dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

            xs = xs.view(B, -1, L)
            dts = dts.contiguous().view(B, -1, L)
            As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
            Bs = Bs.contiguous().view(B, K, N, L)
            Cs = Cs.contiguous().view(B, K, N, L)
            Ds = Ds.to(torch.float)  # (K * c)
            delta_bias = dt_projs_bias.view(-1).to(torch.float)

            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

            ys: torch.Tensor = selective_scan(
                xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
            ).view(B, K, -1, H, W)

            y: torch.Tensor = CrossMerge.apply(ys)

            if getattr(self, "__DEBUG__", False):
                setattr(self, "__data__", dict(
                    A_logs=A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                    us=xs, dts=dts, delta_bias=delta_bias,
                    ys=ys, y=y,
                ))

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)  # (B, L, C)
        y = out_norm(y)

        return (y.to(x.dtype) if to_dtype else y)

    def forwardv2(self, x: torch.Tensor, **kwargs):
        input = self.offsetproj(x) if self.ssm_ratio != 1.0 else x  # input shape: n t v c   # todo: 目前只能expand=1
        x = self.in_proj(x)
        # print(self.disable_z, self.disable_z_act, self.channel_first, self.with_dconv)
        # False False False True
        # print(self.disable_z)
        if not self.disable_z:
            # print(x.shape)
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
            # print(x.shape, z.shape)
            if not self.disable_z_act:
                z = self.act(z)     # n t v c
        if not self.channel_first:
            # print(x.shape)
            x = x.permute(0, 3, 1, 2).contiguous()   # n t v c -> n c t v
            input = input.permute(0, 3, 1, 2).contiguous()  # n t v c -> n c t v
        if self.with_dconv:
            x = self.conv2d(x)  #   # n c t v
        x = self.act(x)
        # torch.Size([1, 256, 243, 17])   # n c t v
        # global offset
        # input of dascan: # input1 : n c t v;  input2 : n t v c
        # print(input.shape, x.shape)  # torch.Size([1, 64, 243, 17]) torch.Size([1, 128, 243, 17])
        x = self.da_scan(input, x)+x  #input, x shape : n c t v;  output shape: n c t v  # TODO 只进行 v offset
        y = self.forward_core(x)  # x shape n c t v   outputshape : n t v c   # todo 当进行 v offset 后，只进行时间前向， 时间后向的话需要重新 v offset
        # 空间动态扫描后， 1. 进行四方向实验， 2. 进行st 前向和后向， 单向。。  3. 时间动态扫描：ts 前向后向，单向。
        # 时间动态 + 空间动态，并联。  时间使用时间ffn增强，空间使用空间ffn 增强。
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        # print('this is forwardv2')
        return out   # n t v c

class SD_STSSM_vv4(nn.Module, mamba_init, SD_STSSM_v4):
    # 4方向的扫描, dynamic是temporal-shared
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v2_four_direction",
        channel_first=False,
        # ======================
        **kwargs,
    ):
        super().__init__()
        kwargs.update(
            d_model=d_model, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, dt_init_floor=dt_init_floor,
            initialize=initialize, forward_type=forward_type, channel_first=channel_first,
        )
        self.__initv2__(**kwargs)


class SD_SSMBlock_v4(nn.Module):
    # Structure dynamic adaptive SSM block
    # 4方向的扫描, dynamic是temporal-shared
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
            channel_first=False,
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2_four_direction",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            gmlp=False,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SD_STSSM_vv4(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                forward_type=forward_type,
                channel_first=channel_first,
            )

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                            drop=mlp_drop_rate, channels_first=channel_first)

    def _forward(self, input: torch.Tensor):
        x = input
        x = x + self.drop_path(self.op(self.norm(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)

# test
# x = torch.randn(1, 243, 17, 64).cuda()
# net = SD_SSMBlock_v4(hidden_dim=64, forward_type='v2_ds_fb_direction').cuda()   # todo: 单向 需要使用原始的ssm
# output = net(x)
# print(output.shape)   # 1,243,17,64
# pdb.set_trace()


# channels shared 2 direction
class SD_STSSM_v5:
    # 2方向的扫描, dynamic是temporal-shared
    def __initv2__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v2_four_direction",
            channel_first=False,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        self.ssm_ratio = ssm_ratio
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear
        self.forward = self.forwardv2

        # tags for forward_type ==============================
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        if out_norm_none:
            self.out_norm = nn.Identity()
        elif out_norm_dwconv3:
            self.out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            self.out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            self.out_norm = nn.Sigmoid()
        else:
            LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm
            self.out_norm = LayerNorm(d_inner)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanMamba),
            # will be deleted in the future
            v02=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanMamba,
                        CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            v03=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanOflex,
                        CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            v04=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex,
                        CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            v05=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True,
                        CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            # ===============================
            v051d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True,
                          CrossScan=getCSM(1)[0], CrossMerge=getCSM(1)[1],
                          ),
            v052d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True,
                          CrossScan=getCSM(2)[0], CrossMerge=getCSM(2)[1],
                          ),
            v052dc=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True,
                           cascade2d=True),
            # ===============================
            v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanCore),
            v2_fs_ft=partial(self.forward_corev2, force_fp32=(not self.disable_force32), CrossScan=CrossScan_fs_ft,
                             SelectiveScan=SelectiveScanCore, CrossMerge=CrossMerge_fs_ft),
            v2_fs_bt=partial(self.forward_corev2, force_fp32=(not self.disable_force32), CrossScan=CrossScan_fs_bt,
                             SelectiveScan=SelectiveScanCore, CrossMerge=CrossMerge_fs_bt),
            v2_bs_ft=partial(self.forward_corev2, force_fp32=(not self.disable_force32), CrossScan=CrossScan_bs_ft,
                             SelectiveScan=SelectiveScanCore, CrossMerge=CrossMerge_bs_ft),
            v2_bs_bt=partial(self.forward_corev2, force_fp32=(not self.disable_force32), CrossScan=CrossScan_bs_bt,
                             SelectiveScan=SelectiveScanCore, CrossMerge=CrossMerge_bs_bt),
            v2_plus_poselimbs=partial(self.forward_corev2, force_fp32=(not self.disable_force32),
                                      CrossScan=CrossScan_plus_poselimbs, SelectiveScan=SelectiveScanCore,
                                      CrossMerge=CrossMerge_plus_poselimbs),

            # v2_four_direction 和 v3 等价
            v2_four_direction=partial(self.forward_corev2, force_fp32=(not self.disable_force32),
                                      CrossScan=CrossScan, SelectiveScan=SelectiveScanOflex,
                                      CrossMerge=CrossMerge),
            # 只进行 st fb 两个方向的扫描。
            v2_ds_fb_direction=partial(self.forward_corev2, force_fp32=(not self.disable_force32),
                                      CrossScan=CrossScan_ds_fb, SelectiveScan=SelectiveScanOflex,
                                      CrossMerge=CrossMerge_ds_fb),   # CrossScan_ds_fb, CrossMerge_ds_fb
            # 只进行 st f 前向扫描
            v2_st_f_direction=partial(self.forward_corev2, force_fp32=(not self.disable_force32),
                                       CrossScan=CrossScan, SelectiveScan=SelectiveScanOflex,
                                       CrossMerge=CrossMerge),

            v3=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex),
            # v1=partial(self.forward_corev2, force_fp32=True, SelectiveScan=SelectiveScanOflex),
            # v4=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True, CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            # ===============================
            v31d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex,
                         CrossScan=CrossScan_Ab_1direction, CrossMerge=CrossMerge_Ab_1direction,
                         ),
            v32d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex,
                         CrossScan=CrossScan_Ab_2direction, CrossMerge=CrossMerge_Ab_2direction,
                         ),
            v32dc=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, cascade2d=True),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)
        k_group = 2   # k_group = 2

        # in proj =======================================
        d_proj = d_inner if self.disable_z else (d_inner * 2)
        self.in_proj = Linear(d_model, d_proj, bias=bias)
        self.offsetproj = Linear(d_model, d_inner, bias=bias)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,
                groups=d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            # dt proj ============================
            self.dt_projs = [
                self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
                for _ in range(k_group)
            ]
            self.dt_projs_weight = nn.Parameter(
                torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
            self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
            del self.dt_projs

            # A, D =======================================
            self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)
            self.Ds = self.D_init(d_inner, copies=k_group, merge=True)  # (K * D)
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(
                torch.randn((k_group * d_inner, d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((k_group, d_inner)))
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(
                torch.zeros((k_group * d_inner, d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, d_inner)))
        num_group = d_inner // d_state
        self.da_scan = JointStruct_Adaptive_Scan_v2(channels=d_inner, group=num_group)  # 核心  v1 为temporal-spcefic v2: temporal-shared
    def forward_corev2(
            self,
            x: torch.Tensor = None,
            # ==============================
            to_dtype=True,  # True: final out to dtype
            force_fp32=False,  # True: input fp32
            # ==============================
            ssoflex=True,  # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
            # ==============================
            SelectiveScan=SelectiveScanOflex,
            CrossScan=CrossScan,
            CrossMerge=CrossMerge,
            no_einsum=False,  # replace einsum with linear or conv1d to raise throughput
            # ==============================
            cascade2d=False,
            **kwargs,
    ):
        # print(SelectiveScan, CrossScan, CrossMerge, cascade2d)
        # <class 'lib.model.csms6s.SelectiveScanCore'> <class 'lib.model.csms6s.CrossScan'> <class 'lib.model.csms6s.CrossMerge'> False
        x_proj_weight = self.x_proj_weight
        x_proj_bias = getattr(self, "x_proj_bias", None)
        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds
        delta_softplus = True
        out_norm = getattr(self, "out_norm", None)
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, -1, -1, ssoflex)

        if cascade2d:
            def scan_rowcol(
                    x: torch.Tensor,
                    proj_weight: torch.Tensor,
                    proj_bias: torch.Tensor,
                    dt_weight: torch.Tensor,
                    dt_bias: torch.Tensor,  # (2*c)
                    _As: torch.Tensor,  # As = -torch.exp(A_logs.to(torch.float))[:2,] # (2*c, d_state)
                    _Ds: torch.Tensor,
                    width=True,
            ):
                # x: (B, D, H, W)
                # proj_weight: (2 * D, (R+N+N))
                XB, XD, XH, XW = x.shape
                if width:
                    _B, _D, _L = XB * XH, XD, XW
                    xs = x.permute(0, 2, 1, 3).contiguous()
                else:
                    _B, _D, _L = XB * XW, XD, XH
                    xs = x.permute(0, 3, 1, 2).contiguous()
                xs = torch.stack([xs, xs.flip(dims=[-1])], dim=2)  # (B, H, 2, D, W)
                if no_einsum:
                    x_dbl = F.conv1d(xs.view(_B, -1, _L), proj_weight.view(-1, _D, 1),
                                     bias=(proj_bias.view(-1) if proj_bias is not None else None), groups=2)
                    dts, Bs, Cs = torch.split(x_dbl.view(_B, 2, -1, _L), [R, N, N], dim=2)
                    dts = F.conv1d(dts.contiguous().view(_B, -1, _L), dt_weight.view(2 * _D, -1, 1), groups=2)
                else:
                    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, proj_weight)
                    if x_proj_bias is not None:
                        x_dbl = x_dbl + x_proj_bias.view(1, 2, -1, 1)
                    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_weight)

                xs = xs.view(_B, -1, _L)
                dts = dts.contiguous().view(_B, -1, _L)
                As = _As.view(-1, N).to(torch.float)
                Bs = Bs.contiguous().view(_B, 2, N, _L)
                Cs = Cs.contiguous().view(_B, 2, N, _L)
                Ds = _Ds.view(-1)
                delta_bias = dt_bias.view(-1).to(torch.float)

                if force_fp32:
                    xs = xs.to(torch.float)
                dts = dts.to(xs.dtype)
                Bs = Bs.to(xs.dtype)
                Cs = Cs.to(xs.dtype)

                ys: torch.Tensor = selective_scan(
                    xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
                ).view(_B, 2, -1, _L)
                return ys

            As = -torch.exp(A_logs.to(torch.float)).view(4, -1, N)
            y_row = scan_rowcol(
                x,
                proj_weight=x_proj_weight.view(4, -1, D)[:2].contiguous(),
                proj_bias=(x_proj_bias.view(4, -1)[:2].contiguous() if x_proj_bias is not None else None),
                dt_weight=dt_projs_weight.view(4, D, -1)[:2].contiguous(),
                dt_bias=(dt_projs_bias.view(4, -1)[:2].contiguous() if dt_projs_bias is not None else None),
                _As=As[:2].contiguous().view(-1, N),
                _Ds=Ds.view(4, -1)[:2].contiguous().view(-1),
                width=True,
            ).view(B, H, 2, -1, W).sum(dim=2).permute(0, 2, 1, 3)
            y_col = scan_rowcol(
                y_row,
                proj_weight=x_proj_weight.view(4, -1, D)[2:].contiguous().to(y_row.dtype),
                proj_bias=(
                    x_proj_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if x_proj_bias is not None else None),
                dt_weight=dt_projs_weight.view(4, D, -1)[2:].contiguous().to(y_row.dtype),
                dt_bias=(
                    dt_projs_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if dt_projs_bias is not None else None),
                _As=As[2:].contiguous().view(-1, N),
                _Ds=Ds.view(4, -1)[2:].contiguous().view(-1),
                width=False,
            ).view(B, W, 2, -1, H).sum(dim=2).permute(0, 2, 3, 1)
            y = y_col
        else:
            xs = CrossScan.apply(x)
            if no_einsum:
                x_dbl = F.conv1d(xs.view(B, -1, L), x_proj_weight.view(-1, D, 1),
                                 bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
                dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
                dts = F.conv1d(dts.contiguous().view(B, -1, L), dt_projs_weight.view(K * D, -1, 1), groups=K)
            else:
                x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
                if x_proj_bias is not None:
                    x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
                dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

            xs = xs.view(B, -1, L)
            dts = dts.contiguous().view(B, -1, L)
            As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
            Bs = Bs.contiguous().view(B, K, N, L)
            Cs = Cs.contiguous().view(B, K, N, L)
            Ds = Ds.to(torch.float)  # (K * c)
            delta_bias = dt_projs_bias.view(-1).to(torch.float)

            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

            ys: torch.Tensor = selective_scan(
                xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
            ).view(B, K, -1, H, W)

            y: torch.Tensor = CrossMerge.apply(ys)

            if getattr(self, "__DEBUG__", False):
                setattr(self, "__data__", dict(
                    A_logs=A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                    us=xs, dts=dts, delta_bias=delta_bias,
                    ys=ys, y=y,
                ))

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)  # (B, L, C)
        y = out_norm(y)

        return (y.to(x.dtype) if to_dtype else y)

    def forwardv2(self, x: torch.Tensor, **kwargs):
        input = self.offsetproj(x) if self.ssm_ratio != 1.0 else x  # input shape: n t v c   # todo: 目前只能expand=1
        x = self.in_proj(x)
        # print(self.disable_z, self.disable_z_act, self.channel_first, self.with_dconv)
        # False False False True
        # print(self.disable_z)
        if not self.disable_z:
            # print(x.shape)
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
            # print(x.shape, z.shape)
            if not self.disable_z_act:
                z = self.act(z)     # n t v c
        if not self.channel_first:
            # print(x.shape)
            x = x.permute(0, 3, 1, 2).contiguous()   # n t v c -> n c t v
            input = input.permute(0, 3, 1, 2).contiguous()  # n t v c -> n c t v
        if self.with_dconv:
            x = self.conv2d(x)  #   # n c t v
        x = self.act(x)
        # torch.Size([1, 256, 243, 17])   # n c t v
        # global offset
        # input of dascan: # input1 : n c t v;  input2 : n t v c
        # print(input.shape, x.shape)  # torch.Size([1, 64, 243, 17]) torch.Size([1, 128, 243, 17])
        x = self.da_scan(input, x)+x  #input, x shape : n c t v;  output shape: n c t v  # TODO 只进行 v offset
        y = self.forward_core(x)  # x shape n c t v   outputshape : n t v c   # todo 当进行 v offset 后，只进行时间前向， 时间后向的话需要重新 v offset
        # 空间动态扫描后， 1. 进行四方向实验， 2. 进行st 前向和后向， 单向。。  3. 时间动态扫描：ts 前向后向，单向。
        # 时间动态 + 空间动态，并联。  时间使用时间ffn增强，空间使用空间ffn 增强。
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        # print('this is forwardv2')
        return out   # n t v c

class SD_STSSM_vv5(nn.Module, mamba_init, SD_STSSM_v5):
    # 2方向的扫描, dynamic是temporal-shared
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v2_four_direction",
        channel_first=False,
        # ======================
        **kwargs,
    ):
        super().__init__()
        kwargs.update(
            d_model=d_model, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, dt_init_floor=dt_init_floor,
            initialize=initialize, forward_type=forward_type, channel_first=channel_first,
        )
        self.__initv2__(**kwargs)


class SD_SSMBlock_v5(nn.Module):
    # Structure dynamic adaptive SSM block
    # 2方向的扫描, dynamic是temporal-shared
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
            channel_first=False,
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2_four_direction",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            gmlp=False,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SD_STSSM_vv5(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                forward_type=forward_type,
                channel_first=channel_first,
            )

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                            drop=mlp_drop_rate, channels_first=channel_first)

    def _forward(self, input: torch.Tensor):
        x = input
        x = x + self.drop_path(self.op(self.norm(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)

# test
# x = torch.randn(1, 243, 17, 64).cuda()
# net = SD_SSMBlock_v4(hidden_dim=64, forward_type='v2_ds_fb_direction').cuda()   # todo: 单向 需要使用原始的ssm
# output = net(x)
# print(output.shape)   # 1,243,17,64
# pdb.set_trace()

class EDFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(EDFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8

        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.fft = nn.Parameter(torch.ones((dim, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)

        b, c, h, w = x.shape
        h_n = (8 - h % 8) % 8
        w_n = (8 - w % 8) % 8

        x = torch.nn.functional.pad(x, (0, w_n, 0, h_n), mode='reflect')
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)

        x = x[:, :, :h, :w]

        return x


class FMPL_v1(nn.Module):
    # not adaptive
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False, v=17):
        super().__init__()
        self.channel_first = channels_first
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.temporal_t = 9
        self.fc1 = Linear(in_features, 2 * hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.fft = nn.Parameter(torch.ones((1, out_features, 1, self.temporal_t, v // 2 + 1)))
    def forward(self, x: torch.Tensor):
        # 1,243,17,64   N T V C
        x = self.fc1(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
        x = self.fc2(x * self.act(z))
        x = self.drop(x)

        n, t, v, c = x.shape
        t_pad = (self.temporal_t - t % self.temporal_t) % self.temporal_t
        x = rearrange(x, 'n t v c -> n c t v')
        x_t = F.pad(x, (0, 0, 0, t_pad))  # Pad T dimension
        x_t_patch = rearrange(x_t, 'b c (t pt) v -> b c t pt v', pt=self.temporal_t)
        x_patch_fft = torch.fft.rfft2(x_t_patch.float())

        # print(x_patch_fft.shape, self.fft.shape)
        # pdb.set_trace()
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.temporal_t, v))
        x = rearrange(x_patch, 'b c t pt v -> b (t pt) v c', pt=self.temporal_t)

        x = x[:, :t, :, :]  # 保留原始的时间长度 t，去掉 pad

        return x

class FMPL_v2(nn.Module):
    # not adaptive
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False, v=17):
        super().__init__()
        self.channel_first = channels_first
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.temporal_t = 9
        self.fc1 = Linear(in_features, 2 * hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.fft = nn.Parameter(torch.ones((1, out_features, 1, v, self.temporal_t//2 + 1)))
    def forward(self, x: torch.Tensor):
        # 1,243,17,64   N T V C
        x = self.fc1(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
        x = self.fc2(x * self.act(z))
        x = self.drop(x)

        n, t, v, c = x.shape
        t_pad = (self.temporal_t - t % self.temporal_t) % self.temporal_t
        x = rearrange(x, 'n t v c -> n c t v')
        x_t = F.pad(x, (0, 0, 0, t_pad))  # Pad T dimension
        x_t_patch = rearrange(x_t, 'b c (t pt) v -> b c t v pt', pt=self.temporal_t)
        x_patch_fft = torch.fft.rfft2(x_t_patch.float())

        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(v, self.temporal_t))
        x = rearrange(x_patch, 'b c t v pt -> b (t pt) v c', pt=self.temporal_t)

        x = x[:, :t, :, :]  # 保留原始的时间长度 t，去掉 pad

        return x


class FMPL_combined(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False, v=17, temporal_t=9, fft_mode='mixed'):
        super().__init__()
        self.channel_first = channels_first
        self.temporal_t = temporal_t
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.fft_mode = fft_mode

        if fft_mode == 'temporal':
            self.fft = nn.Parameter(torch.ones((1, out_features, 1, temporal_t, v // 2 + 1)))
        elif fft_mode == 'spatial':
            self.fft = nn.Parameter(torch.ones((1, out_features, 1, v, temporal_t // 2 + 1)))
        elif fft_mode == 'mixed':
            self.fft_t = nn.Parameter(torch.ones((1, out_features, 1, temporal_t, v // 2 + 1)))
            self.fft_v = nn.Parameter(torch.ones((1, out_features, 1, v, temporal_t // 2 + 1)))
        else:
            raise ValueError(f"Invalid fft_mode: {fft_mode}")

    def forward(self, x: torch.Tensor):
        # 1,243,17,64   N T V C
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        n, t, v, c = x.shape
        t_pad = (self.temporal_t - t % self.temporal_t) % self.temporal_t
        x = rearrange(x, 'n t v c -> n c t v')
        x_t = F.pad(x, (0, 0, 0, t_pad))

        if self.fft_mode == 'temporal':
            x_t_patch = rearrange(x_t, 'b c (t pt) v -> b c t pt v', pt=self.temporal_t)
            x_patch_fft = torch.fft.rfft2(x_t_patch.float())
            x_patch_fft = x_patch_fft * self.fft
            x_patch = torch.fft.irfft2(x_patch_fft, s=(self.temporal_t, v))
            x = rearrange(x_patch, 'b c t pt v -> b (t pt) v c', pt=self.temporal_t)
        elif self.fft_mode == 'spatial':
            x_t_patch = rearrange(x_t, 'b c (t pt) v -> b c t v pt', pt=self.temporal_t)
            x_patch_fft = torch.fft.rfft2(x_t_patch.float())
            x_patch_fft = x_patch_fft * self.fft
            x_patch = torch.fft.irfft2(x_patch_fft, s=(v, self.temporal_t))
            x = rearrange(x_patch, 'b c t v pt -> b (t pt) v c', pt=self.temporal_t)
        elif self.fft_mode == 'mixed':
            # Temporal FFT
            x_t_patch_t = rearrange(x_t, 'b c (t pt) v -> b c t pt v', pt=self.temporal_t)
            x_patch_fft_t = torch.fft.rfft2(x_t_patch_t.float())
            x_patch_fft_t = x_patch_fft_t * self.fft_t
            x_patch_t = torch.fft.irfft2(x_patch_fft_t, s=(self.temporal_t, v))

            # Spatial FFT
            x_t_patch_v = rearrange(x_t, 'b c (t pt) v -> b c t v pt', pt=self.temporal_t)
            x_patch_fft_v = torch.fft.rfft2(x_t_patch_v.float())
            x_patch_fft_v = x_patch_fft_v * self.fft_v
            x_patch_v = torch.fft.irfft2(x_patch_fft_v, s=(v, self.temporal_t))

            # Combine
            x_patch_v = rearrange(x_patch_v, 'b c t v pt -> b (t pt) v c', pt=self.temporal_t)
            x_patch_t = rearrange(x_patch_t, 'b c t pt v -> b (t pt) v c', pt=self.temporal_t)

            x = (x_patch_t + x_patch_v) / 2

        x = x[:, :t, :, :]  # Remove padding
        return x


class FMPL_Adaptive(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False, v=17, temporal_t=9, fft_mode='mixed'):
        super().__init__()
        self.channel_first = channels_first
        self.temporal_t = temporal_t
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, 2 * hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.fft_mode = fft_mode

        # Adaptive FFT Parameters
        if fft_mode == 'temporal':
            self.fft = nn.Parameter(torch.ones((1, out_features, 1, temporal_t, v // 2 + 1)))
        elif fft_mode == 'spatial':
            self.fft = nn.Parameter(torch.ones((1, out_features, 1, v, temporal_t // 2 + 1)))
        elif fft_mode == 'mixed':
            self.fft_t = nn.Parameter(torch.ones((1, out_features, 1, temporal_t, v // 2 + 1)))
            self.fft_v = nn.Parameter(torch.ones((1, out_features, 1, v, temporal_t // 2 + 1)))
            # Adaptive weights for fft_t and fft_v
            self.adaptive_t = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(out_features, out_features, 1),
                nn.GELU(),
                nn.Conv2d(out_features, out_features, 1),
                nn.Sigmoid()
            )
            self.adaptive_v = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(out_features, out_features, 1),
                nn.GELU(),
                nn.Conv2d(out_features, out_features, 1),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Invalid fft_mode: {fft_mode}")

    def forward(self, x: torch.Tensor):
        # 1,243,17,64   N T V C
        x = self.fc1(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
        x = self.fc2(x * self.act(z))
        x = self.drop(x)

        n, t, v, c = x.shape
        t_pad = (self.temporal_t - t % self.temporal_t) % self.temporal_t
        x = rearrange(x, 'n t v c -> n c t v')
        x_t = F.pad(x, (0, 0, 0, t_pad))

        if self.fft_mode == 'temporal':
            x_t_patch = rearrange(x_t, 'b c (t pt) v -> b c t pt v', pt=self.temporal_t)
            x_patch_fft = torch.fft.rfft2(x_t_patch.float())
            x_patch_fft = x_patch_fft * self.fft
            x_patch = torch.fft.irfft2(x_patch_fft, s=(self.temporal_t, v))
            x = rearrange(x_patch, 'b c t pt v -> b (t pt) v c', pt=self.temporal_t)
        elif self.fft_mode == 'spatial':
            x_t_patch = rearrange(x_t, 'b c (t pt) v -> b c t v pt', pt=self.temporal_t)
            x_patch_fft = torch.fft.rfft2(x_t_patch.float())
            x_patch_fft = x_patch_fft * self.fft
            x_patch = torch.fft.irfft2(x_patch_fft, s=(v, self.temporal_t))
            x = rearrange(x_patch, 'b c t v pt -> b (t pt) v c', pt=self.temporal_t)
        elif self.fft_mode == 'mixed':
            # Temporal FFT
            x_t_patch_t = rearrange(x_t, 'b c (t pt) v -> b c t pt v', pt=self.temporal_t)
            x_patch_fft_t = torch.fft.rfft2(x_t_patch_t.float())
            # Apply adaptive scaling
            scale_t = self.adaptive_t(x.mean(dim=-1, keepdim=True))
            x_patch_fft_t = x_patch_fft_t * self.fft_t * scale_t
            x_patch_t = torch.fft.irfft2(x_patch_fft_t, s=(self.temporal_t, v))

            # Spatial FFT
            x_t_patch_v = rearrange(x_t, 'b c (t pt) v -> b c t v pt', pt=self.temporal_t)
            x_patch_fft_v = torch.fft.rfft2(x_t_patch_v.float())
            # Apply adaptive scaling
            scale_v = self.adaptive_v(x.mean(dim=-2, keepdim=True))
            x_patch_fft_v = x_patch_fft_v * self.fft_v * scale_v
            x_patch_v = torch.fft.irfft2(x_patch_fft_v, s=(v, self.temporal_t))

            # Combine
            x = (x_patch_t + x_patch_v) / 2
            x = rearrange(x, 'b c t pt v -> b (t pt) v c', pt=self.temporal_t)

        x = x[:, :t, :, :]  # Remove padding
        return x

def test_fmpl_combined():
    modes = ['temporal', 'spatial', 'mixed']
    for mode in modes:
        print(f"Testing mode: {mode}")
        model = FMPL_combined(in_features=64, hidden_features=128, out_features=64, fft_mode=mode)
        x = torch.randn(2, 243, 17, 64)  # Batch size 2, 243 frames, 17 joints, 64 channels
        y = model(x)
        assert y.shape == (2, 243, 17, 64), f"Unexpected output shape for {mode}: {y.shape}"
        print(f"{mode} passed. Output shape: {y.shape}")

# test_fmpl_combined()



class FMPL_Adapt2(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False, v=17, temporal_t=9, fft_mode='mixed', reduction=4):
        super().__init__()
        self.channel_first = channels_first
        self.temporal_t = temporal_t
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.fft_mode = fft_mode

        # Adaptive FFT Parameters (Reduced)
        if fft_mode == 'temporal':
            self.fft = nn.Parameter(torch.ones((1, out_features // reduction, 1, temporal_t, v // 2 + 1)))
        elif fft_mode == 'spatial':
            self.fft = nn.Parameter(torch.ones((1, out_features // reduction, 1, v, temporal_t // 2 + 1)))
        elif fft_mode == 'mixed':
            self.fft_t = nn.Parameter(torch.ones((1, out_features // reduction, 1, temporal_t, v // 2 + 1)))
            self.fft_v = nn.Parameter(torch.ones((1, out_features // reduction, 1, v, temporal_t // 2 + 1)))
            # Shared Adaptive Scaling for Temporal and Spatial FFT
            self.adaptive_scale = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(out_features, out_features // reduction, 1),
                nn.GELU(),
                nn.Conv2d(out_features // reduction, out_features // reduction, 1),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Invalid fft_mode: {fft_mode}")

    def forward(self, x: torch.Tensor):
        # 1,243,17,64   N T V C
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)

        n, t, v, c = x.shape
        t_pad = (self.temporal_t - t % self.temporal_t) % self.temporal_t
        x = rearrange(x, 'n t v c -> n c t v')
        x_t = F.pad(x, (0, 0, 0, t_pad))

        if self.fft_mode == 'temporal':
            x_t_patch = rearrange(x_t, 'b c (t pt) v -> b c t pt v', pt=self.temporal_t)
            x_patch_fft = torch.fft.rfft2(x_t_patch.float())
            scale = self.adaptive_scale(x.mean(dim=-1, keepdim=True))
            x_patch_fft = x_patch_fft * self.fft * scale
            x_patch = torch.fft.irfft2(x_patch_fft, s=(self.temporal_t, v))
            x = rearrange(x_patch, 'b c t pt v -> b (t pt) v c', pt=self.temporal_t)
        elif self.fft_mode == 'spatial':
            x_t_patch = rearrange(x_t, 'b c (t pt) v -> b c t v pt', pt=self.temporal_t)
            x_patch_fft = torch.fft.rfft2(x_t_patch.float())
            scale = self.adaptive_scale(x.mean(dim=-2, keepdim=True))
            x_patch_fft = x_patch_fft * self.fft * scale
            x_patch = torch.fft.irfft2(x_patch_fft, s=(v, self.temporal_t))
            x = rearrange(x_patch, 'b c t v pt -> b (t pt) v c', pt=self.temporal_t)
        elif self.fft_mode == 'mixed':
            # Temporal FFT
            x_t_patch_t = rearrange(x_t, 'b c (t pt) v -> b c t pt v', pt=self.temporal_t)
            x_patch_fft_t = torch.fft.rfft2(x_t_patch_t.float())
            scale_t = self.adaptive_scale(x.mean(dim=-1, keepdim=True))
            x_patch_fft_t = x_patch_fft_t * self.fft_t * scale_t
            x_patch_t = torch.fft.irfft2(x_patch_fft_t, s=(self.temporal_t, v))

            # Spatial FFT
            x_t_patch_v = rearrange(x_t, 'b c (t pt) v -> b c t v pt', pt=self.temporal_t)
            x_patch_fft_v = torch.fft.rfft2(x_t_patch_v.float())
            scale_v = self.adaptive_scale(x.mean(dim=-2, keepdim=True))
            x_patch_fft_v = x_patch_fft_v * self.fft_v * scale_v
            x_patch_v = torch.fft.irfft2(x_patch_fft_v, s=(v, self.temporal_t))

            # Combine
            x = (x_patch_t + x_patch_v) / 2
            x = rearrange(x, 'b c t pt v -> b (t pt) v c', pt=self.temporal_t)

        x = x[:, :t, :, :]  # Remove padding
        return x


def test_fmpl_combined():
    modes = ['temporal', 'spatial', 'mixed']
    for mode in modes:
        print(f"Testing mode: {mode}")
        model = FMPL_combined(in_features=64, hidden_features=128, out_features=64, fft_mode=mode)
        x = torch.randn(2, 243, 17, 64)  # Batch size 2, 243 frames, 17 joints, 64 channels
        y = model(x)
        assert y.shape == (2, 243, 17, 64), f"Unexpected output shape for {mode}: {y.shape}"
        print(f"{mode} passed. Output shape: {y.shape}")


# test_fmpl_combined()




class BiSTSSMBlock_FFNV1(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
            channel_first=False,
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",  # v2_plus_poselimbs
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            gmlp=False,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = BiSTSSM(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                forward_type=forward_type,  # v2_plus_poselimbs
                channel_first=channel_first,
            )

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = FMPL_v1(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                            drop=mlp_drop_rate, channels_first=channel_first, v=17)

    def _forward(self, input: torch.Tensor):
        x = input
        x = x + self.drop_path(self.op(self.norm(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)


class BiSTSSMBlock_FFNV2(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
            channel_first=False,
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",  # v2_plus_poselimbs
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            gmlp=False,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = BiSTSSM(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                forward_type=forward_type,  # v2_plus_poselimbs
                channel_first=channel_first,
            )

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            self.norm1 = norm_layer(hidden_dim//2)
            self.norm2 = norm_layer(hidden_dim//2)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp1 = Mlp(in_features=hidden_dim//2, hidden_features=mlp_hidden_dim//2, act_layer=mlp_act_layer,
                            drop=mlp_drop_rate, channels_first=channel_first)
            self.mlp2 = FreqMlp1DTime(in_features=hidden_dim//2, hidden_features=mlp_hidden_dim//2, act_layer=mlp_act_layer,
                                     drop=mlp_drop_rate)

    def _forward(self, input: torch.Tensor):
        x = input
        _,c,_,_ = x.shape
        x = x + self.drop_path(self.op(self.norm(x)))
        x1 = x[:,:c//2] + self.drop_path(self.mlp1(self.norm1(x[:,:c//2])))
        x2 = x[:,c//2:] + self.drop_path(self.mlp2(self.norm2(x[:,c//2:])))
        return torch.cat((x1, x2), dim=1)

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)





class DASSM(nn.Module):
    def __init__(self, in_features, dt_rank: int, dim_inner: int, d_state: int):
        """
        Initializes the SSM module.

        Args:
            in_features (int): The size of the input features.
            dt_rank (int): The rank of the dt projection.
            dim_inner (int): The inner dimension of the dt projection.
            d_state (int): The dimension of the state.

        """
        super().__init__()
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state

        # Linear layer expecting 'in_features' as the input size
        self.deltaBC_layer = nn.Linear(
            in_features, dt_rank + 2 * d_state, bias=False
        )
        self.dt_proj_layer = nn.Linear(dt_rank, dim_inner, bias=True)

        # Defining A_log and D as parameters
        self.A_log = nn.Parameter(
            torch.log(
                torch.arange(1, d_state + 1, dtype=torch.float32).repeat(
                    dim_inner, 1
                )
            )
        )
        self.D = nn.Parameter(torch.ones(dim_inner))

    def forward(self, x, pscan: bool = True):
        """
        Performs forward pass of the SSM module.

        Args:
            x (torch.Tensor): The input tensor.  n t v c
            pscan (bool, optional): Whether to use selective_scan or selective_scan_seq. Defaults to True.

        Returns:
            torch.Tensor: The output tensor.

        """
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        deltaBC = self.deltaBC_layer(x)
        delta, B, C = torch.split(
            deltaBC, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj_layer(delta))

        # Assuming selective_scan and selective_scan_seq are defined functions

        hs = da_selective_scan(x, delta, A, B, C, D)

        y = (
                hs @ C.unsqueeze(-1)
        ).squeeze()  # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y

# x = torch.randn(1, 243, 17, 64).cuda()
# net = DASSM(hidden_dim=64, forward_type='v2_ds_fb_direction').cuda()   # todo: 单向 需要使用原始的ssm
# output = net(x)
# print(output.shape)   # 1,243,17,64
# pdb.set_trace()