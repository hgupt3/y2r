# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Callable
import collections
from torch import Tensor
from itertools import repeat

from y2r.models.model_utils import bilinear_sampler


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


to_2tuple = _ntuple(2)


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn="group", stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            padding=1,
            stride=stride,
            padding_mode="zeros",
        )
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, padding=1, padding_mode="zeros"
        )
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3
            )

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class BasicEncoder(nn.Module):
    def __init__(self, input_dim=3, output_dim=128, stride=4):
        super(BasicEncoder, self).__init__()
        self.stride = stride
        self.norm_fn = "instance"
        self.in_planes = output_dim // 2
        self.norm1 = nn.InstanceNorm2d(self.in_planes)
        self.norm2 = nn.InstanceNorm2d(output_dim * 2)

        self.conv1 = nn.Conv2d(
            input_dim,
            self.in_planes,
            kernel_size=7,
            stride=2,
            padding=3,
            padding_mode="zeros",
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(output_dim // 2, stride=1)
        self.layer2 = self._make_layer(output_dim // 4 * 3, stride=2)
        self.layer3 = self._make_layer(output_dim, stride=2)
        self.layer4 = self._make_layer(output_dim, stride=2)

        self.conv2 = nn.Conv2d(
            output_dim * 3 + output_dim // 4,
            output_dim * 2,
            kernel_size=3,
            padding=1,
            padding_mode="zeros",
        )
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(output_dim * 2, output_dim, kernel_size=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        _, _, H, W = x.shape

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        a = self.layer1(x)
        b = self.layer2(a)
        c = self.layer3(b)
        d = self.layer4(c)

        def _bilinear_intepolate(x):
            return F.interpolate(
                x,
                (H // self.stride, W // self.stride),
                mode="bilinear",
                align_corners=True,
            )

        a = _bilinear_intepolate(a)
        b = _bilinear_intepolate(b)
        c = _bilinear_intepolate(c)
        d = _bilinear_intepolate(d)

        x = self.conv2(torch.cat([a, b, c, d], dim=1))
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        return x


class EfficientCorrBlock:
    def __init__(
        self,
        fmaps,
        num_levels=4,
        radius=4,
        padding_mode="zeros",
    ):
        B, S, C, H, W = fmaps.shape
        self.padding_mode = padding_mode
        self.num_levels = num_levels
        self.radius = radius
        self.fmaps_pyramid = []

        self.fmaps_pyramid.append(fmaps)
        for i in range(self.num_levels - 1):
            fmaps_ = fmaps.reshape(B * S, C, H, W)
            fmaps_ = F.avg_pool2d(fmaps_, 2, stride=2)
            _, _, H, W = fmaps_.shape
            fmaps = fmaps_.reshape(B, S, C, H, W)
            self.fmaps_pyramid.append(fmaps)

    def sample(self, coords, target):
        r = self.radius
        device = coords.device
        B, S, N, D = coords.shape
        assert D == 2

        target = target.permute(0, 1, 3, 2).unsqueeze(-1)

        out_pyramid = []
        for i in range(self.num_levels):
            pyramid = self.fmaps_pyramid[i]
            C, H, W = pyramid.shape[2:]
            centroid_lvl = (
                torch.cat(
                    [torch.zeros_like(coords[..., :1], device=device), coords], dim=-1
                ).reshape(B * S, N, 1, 1, 3)
                / 2**i
            )

            dx = torch.linspace(-r, r, 2 * r + 1, device=device)
            dy = torch.linspace(-r, r, 2 * r + 1, device=device)

            xgrid, ygrid = torch.meshgrid(dy, dx, indexing="ij")
            zgrid = torch.zeros_like(xgrid, device=device)
            delta = torch.stack([zgrid, xgrid, ygrid], axis=-1)
            delta_lvl = delta.view(1, 1, 2 * r + 1, 2 * r + 1, 3)
            coords_lvl = centroid_lvl + delta_lvl
            pyramid_sample = bilinear_sampler(
                pyramid.reshape(B * S, C, 1, H, W), coords_lvl
            )

            corr = torch.sum(target * pyramid_sample.reshape(B, S, C, N, -1), dim=2)
            corr = corr / torch.sqrt(torch.tensor(C).float())
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)  # B, S, N, LRR*2
        out = out.permute(0, 2, 1, 3).contiguous().view(B * N, S, -1).float()
        return out


class CorrBlock:
    def __init__(
        self,
        fmaps,
        num_levels=4,
        radius=4,
        multiple_track_feats=False,
        padding_mode="zeros",
    ):
        B, S, C, H, W = fmaps.shape
        self.S, self.C, self.H, self.W = S, C, H, W
        self.padding_mode = padding_mode
        self.num_levels = num_levels
        self.radius = radius
        self.fmaps_pyramid = []
        self.multiple_track_feats = multiple_track_feats

        self.fmaps_pyramid.append(fmaps)
        for i in range(self.num_levels - 1):
            fmaps_ = fmaps.reshape(B * S, C, H, W)
            fmaps_ = F.avg_pool2d(fmaps_, 2, stride=2)
            _, _, H, W = fmaps_.shape
            fmaps = fmaps_.reshape(B, S, C, H, W)
            self.fmaps_pyramid.append(fmaps)

    def sample(self, coords):
        r = self.radius
        B, S, N, D = coords.shape
        assert D == 2

        H, W = self.H, self.W
        out_pyramid = []
        for i in range(self.num_levels):
            corrs = self.corrs_pyramid[i]  # B, S, N, H, W
            *_, H, W = corrs.shape

            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), axis=-1).to(
                coords.device
            )

            centroid_lvl = coords.reshape(B * S * N, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corrs = bilinear_sampler(
                corrs.reshape(B * S * N, 1, H, W),
                coords_lvl,
                padding_mode=self.padding_mode,
            )
            corrs = corrs.view(B, S, N, -1)
            out_pyramid.append(corrs)

        out = torch.cat(out_pyramid, dim=-1)  # B, S, N, LRR*2
        out = out.permute(0, 2, 1, 3).contiguous().view(B * N, S, -1).float()
        return out

    def corr(self, targets):
        B, S, N, C = targets.shape
        if self.multiple_track_feats:
            targets_split = targets.split(C // self.num_levels, dim=-1)
            B, S, N, C = targets_split[0].shape

        assert C == self.C
        assert S == self.S

        fmap1 = targets

        self.corrs_pyramid = []
        for i, fmaps in enumerate(self.fmaps_pyramid):
            *_, H, W = fmaps.shape
            fmap2s = fmaps.view(B, S, C, H * W)  # B S C H W ->  B S C (H W)
            if self.multiple_track_feats:
                fmap1 = targets_split[i]
            corrs = torch.matmul(fmap1, fmap2s)
            corrs = corrs.view(B, S, N, H, W)  # B S N (H W) -> B S N H W
            corrs = corrs / torch.sqrt(torch.tensor(C).float())
            self.corrs_pyramid.append(corrs)


class Attention(nn.Module):
    def __init__(
        self, query_dim, context_dim=None, num_heads=8, dim_head=48, qkv_bias=False
    ):
        super().__init__()
        inner_dim = dim_head * num_heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head**-0.5
        self.heads = num_heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=qkv_bias)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=qkv_bias)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, attn_bias=None):
        B, N1, C = x.shape
        h = self.heads

        q = self.to_q(x).reshape(B, N1, h, C // h).permute(0, 2, 1, 3)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        N2 = context.shape[1]
        k = k.reshape(B, N2, h, C // h).permute(0, 2, 1, 3)
        v = v.reshape(B, N2, h, C // h).permute(0, 2, 1, 3)

        sim = (q @ k.transpose(-2, -1)) * self.scale

        if attn_bias is not None:
            sim = sim + attn_bias
        attn = sim.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N1, C)
        return self.to_out(x)


class AttnBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        attn_class: Callable[..., nn.Module] = Attention,
        mlp_ratio=4.0,
        drop=0.0,
        **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # Don't pass 'drop' to attention - it's only for MLP
        dim_head = hidden_size // num_heads
        self.attn = attn_class(
            hidden_size, num_heads=num_heads, dim_head=dim_head, qkv_bias=True, **block_kwargs
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=drop,
        )

    def forward(self, x, mask=None):
        attn_bias = None
        if mask is not None:
            # mask is (seq_len, seq_len) boolean: True = can attend
            # Convert to attention bias: 0 where can attend, -inf where can't
            num_heads = self.attn.heads
            B = x.shape[0]
            seq_len = x.shape[1]
            
            # Expand mask to (B, num_heads, seq_len, seq_len)
            mask_expanded = mask.unsqueeze(0).unsqueeze(0).expand(B, num_heads, seq_len, seq_len)
            
            max_neg_value = -torch.finfo(x.dtype).max
            attn_bias = torch.where(mask_expanded, torch.zeros_like(mask_expanded, dtype=x.dtype), 
                                   torch.full_like(mask_expanded, max_neg_value, dtype=x.dtype))
        
        x = x + self.attn(self.norm1(x), attn_bias=attn_bias)
        x = x + self.mlp(self.norm2(x))
        return x

class CrossAttnBlock(nn.Module):
    def __init__(
        self, hidden_size, context_dim, num_heads=1, mlp_ratio=4.0, drop=0.0, **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_context = nn.LayerNorm(hidden_size)
        dim_head = hidden_size // num_heads
        self.cross_attn = Attention(
            hidden_size,
            context_dim=context_dim,
            num_heads=num_heads,
            dim_head=dim_head,
            qkv_bias=True,
            **block_kwargs
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=drop,
        )

    def forward(self, x, context, mask=None):
        attn_bias = None
        if mask is not None:
            if mask.shape[1] == x.shape[1]:
                mask = mask[:, None, :, None].expand(
                    -1, self.cross_attn.heads, -1, context.shape[1]
                )
            else:
                mask = mask[:, None, None].expand(
                    -1, self.cross_attn.heads, x.shape[1], -1
                )

            max_neg_value = -torch.finfo(x.dtype).max
            attn_bias = (~mask) * max_neg_value
        x = x + self.cross_attn(
            self.norm1(x), context=self.norm_context(context), attn_bias=attn_bias
        )
        x = x + self.mlp(self.norm2(x))
        return x
    
    
class EfficientUpdateFormer(nn.Module):
    """
    Transformer model that updates track estimates.
    Modified for future prediction with factorized space-time attention and scene cross-attention.
    
    Uses factorized attention for memory efficiency:
    1. Cross-attention to scene tokens
    2. Spatial self-attention (tokens at same timestep attend to each other)
    3. Temporal self-attention (same position across timesteps attend to each other)
    
    Memory: O(NT(N+T)) instead of O((NT)²) for joint attention
    """

    def __init__(
        self,
        depth=6,
        input_dim=320,
        hidden_size=384,
        num_heads=8,
        output_dim=130,
        mlp_ratio=4.0,
        add_space_attn=True,  # Controls whether to include spatial attention
        p_drop_attn=0.0,
        linear_layer_for_vis_conf=False,
        # Legacy parameters for backward compatibility
        space_depth=None,
        time_depth=None,
    ):
        super().__init__()
        # Handle legacy parameters
        if time_depth is not None:
            depth = time_depth
        
        self.out_channels = 2
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.depth = depth
        self.add_space_attn = add_space_attn  # Store for forward pass
        self.input_transform = torch.nn.Linear(input_dim, hidden_size, bias=True) if input_dim != hidden_size else nn.Identity()
        if linear_layer_for_vis_conf:
            self.flow_head = torch.nn.Linear(hidden_size, output_dim - 2, bias=True)
            self.vis_conf_head = torch.nn.Linear(hidden_size, 2, bias=True)
        else:
            self.flow_head = torch.nn.Linear(hidden_size, output_dim, bias=True)
        self.linear_layer_for_vis_conf = linear_layer_for_vis_conf
        
        # Cross-attention blocks for scene context (one per layer)
        self.scene_cross_attn_blocks = nn.ModuleList([
            CrossAttnBlock(
                hidden_size,
                context_dim=hidden_size,  # Scene tokens will be projected to hidden_size
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=p_drop_attn,
            )
            for _ in range(depth)
        ])
        
        # Spatial attention blocks (tokens at same timestep attend to each other)
        # Memory: O(N²) per timestep, total O(N²T)
        self.space_blocks = nn.ModuleList([
            AttnBlock(
                hidden_size,
                num_heads,
                mlp_ratio=mlp_ratio,
                attn_class=Attention,
                drop=p_drop_attn,
            )
            for _ in range(depth)
        ])
        
        # Temporal attention blocks (same position across timesteps attend to each other)
        # Memory: O(T²) per position, total O(T²N)
        self.time_blocks = nn.ModuleList([
            AttnBlock(
                hidden_size,
                num_heads,
                mlp_ratio=mlp_ratio,
                attn_class=Attention,
                drop=p_drop_attn,
            )
            for _ in range(depth)
        ])
        
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            if hasattr(self, 'flow_head'):
                torch.nn.init.trunc_normal_(self.flow_head.weight, std=0.001)
                if self.linear_layer_for_vis_conf and hasattr(self, 'vis_conf_head'):
                    torch.nn.init.trunc_normal_(self.vis_conf_head.weight, std=0.001)

        self.apply(_basic_init)

    def forward(self, input_tensor, scene_tokens, mask=None, add_space_attn=None, causal_time=False):
        """
        Forward pass with scene cross-attention and factorized space-time attention.
        
        Uses factorized attention for memory efficiency:
        - Spatial: (B, N, T, C) → (B*T, N, C) - N×N attention per timestep
        - Temporal: (B, N, T, C) → (B*N, T, C) - T×T attention per position
        
        Args:
            input_tensor: (B, N, T, input_dim) - point features with temporal encoding
            scene_tokens: (B, num_patches, hidden_size) - ViT scene features
            mask: optional attention mask (ignored in new implementation)
            add_space_attn: if provided, overrides self.add_space_attn
            causal_time: if True, apply causal mask for temporal attention
                        Used for autoregressive training with teacher forcing
        
        Returns:
            flow: (B, N, T, output_dim) - predicted displacements
        """
        tokens = self.input_transform(input_tensor)  # (B, N, T, hidden_size)
        B, N, T, C = tokens.shape
        
        use_space_attn = add_space_attn if add_space_attn is not None else self.add_space_attn
        
        # Create causal mask for temporal attention if needed
        time_causal_mask = None
        if causal_time and T > 1:
            # (T, T) mask where position i can attend to positions <= i
            time_causal_mask = torch.tril(torch.ones(T, T, device=tokens.device, dtype=torch.bool))
        
        # Main transformer loop: Cross-Attn (scene) → Spatial Self-Attn → Temporal Self-Attn
        for i in range(self.depth):
            # 1. CROSS-ATTENTION TO SCENE: all tokens attend to ViT scene tokens (+ text)
            # Reshape: (B, N, T, C) → (B, N*T, C) for cross-attention
            cross_tokens = tokens.contiguous().view(B, N * T, C)
            cross_tokens = self.scene_cross_attn_blocks[i](
                cross_tokens, scene_tokens, mask=None
            )
            tokens = cross_tokens.contiguous().view(B, N, T, C)
            
            # 2. SPATIAL SELF-ATTENTION: tokens at same timestep attend to each other
            # Reshape: (B, N, T, C) → (B*T, N, C) - batch over timesteps
            # Memory: O(N²) per timestep instead of O((NT)²) for joint
            if use_space_attn:
                space_tokens = tokens.permute(0, 2, 1, 3).contiguous().view(B * T, N, C)
                space_tokens = self.space_blocks[i](space_tokens, mask=None)
                tokens = space_tokens.view(B, T, N, C).permute(0, 2, 1, 3).contiguous()
            
            # 3. TEMPORAL SELF-ATTENTION: same position across timesteps attend to each other
            # Reshape: (B, N, T, C) → (B*N, T, C) - batch over spatial positions
            # Memory: O(T²) per position instead of O((NT)²) for joint
            time_tokens = tokens.view(B * N, T, C)
            time_tokens = self.time_blocks[i](time_tokens, mask=time_causal_mask)
            tokens = time_tokens.view(B, N, T, C)
        
        # Output head
        flow = self.flow_head(tokens)
        if self.linear_layer_for_vis_conf:
            vis_conf = self.vis_conf_head(tokens)
            flow = torch.cat([flow, vis_conf], dim=-1)

        return flow


#################################################################################
#                    DiT Components (Diffusion Transformer)                    #
#################################################################################
# Adapted from: https://github.com/facebookresearch/DiT
# Copyright (c) Meta Platforms, Inc. and affiliates.

def modulate(x, shift, scale):
    """Apply affine modulation to layer norm output."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    Copied from DiT (Diffusion Transformer).
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(max_period)) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiTAttnBlock(nn.Module):
    """
    Self-attention block with adaptive layer norm zero (adaLN-Zero) conditioning.
    Adapted from DiT for use with factorized space-time attention.
    """
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        attn_class=Attention,
        drop=0.0,
        **block_kwargs
    ):
        super().__init__()
        # LayerNorms without learnable parameters (adaLN will provide them)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Attention
        dim_head = hidden_size // num_heads
        self.attn = attn_class(
            hidden_size, num_heads=num_heads, dim_head=dim_head, qkv_bias=True, **block_kwargs
        )

        # MLP
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=drop,
        )

        # Adaptive modulation: produces 6 parameters (shift, scale, gate) x 2
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, mask=None):
        """
        Args:
            x: (B, N, C) input tokens
            c: (B, C) conditioning vector (timestep embedding)
            mask: optional attention mask
        """
        # Split modulation parameters
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)

        # Attention block with modulation
        attn_bias = None
        if mask is not None:
            # Handle mask (same logic as AttnBlock)
            num_heads = self.attn.heads
            B, seq_len = x.shape[0], x.shape[1]
            mask_expanded = mask.unsqueeze(0).unsqueeze(0).expand(B, num_heads, seq_len, seq_len)
            max_neg_value = -torch.finfo(x.dtype).max
            attn_bias = torch.where(mask_expanded, torch.zeros_like(mask_expanded, dtype=x.dtype),
                                   torch.full_like(mask_expanded, max_neg_value, dtype=x.dtype))

        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            attn_bias=attn_bias
        )

        # MLP block with modulation
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )

        return x


class DiTCrossAttnBlock(nn.Module):
    """
    Cross-attention block with adaptive layer norm (adaLN-Zero).
    Queries attend to context (scene tokens), conditioned on timestep.
    """
    def __init__(
        self,
        hidden_size,
        context_dim,
        num_heads,
        mlp_ratio=4.0,
        drop=0.0,
        **block_kwargs
    ):
        super().__init__()
        # LayerNorms without learnable parameters (query side)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # Context norm is standard (not conditioned)
        self.norm_context = nn.LayerNorm(hidden_size)

        # Cross-attention
        dim_head = hidden_size // num_heads
        self.cross_attn = Attention(
            hidden_size,
            context_dim=context_dim,
            num_heads=num_heads,
            dim_head=dim_head,
            qkv_bias=True,
            **block_kwargs
        )

        # MLP
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=drop,
        )

        # Adaptive modulation for query and MLP (6 params)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, context, c, mask=None):
        """
        Args:
            x: (B, N, C) query tokens
            context: (B, M, C) key/value tokens (scene features)
            c: (B, C) conditioning vector (timestep embedding)
            mask: optional attention mask
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)

        # Cross-attention with modulation
        attn_bias = None
        if mask is not None:
            # Handle context masking
            if mask.shape[1] == x.shape[1]:
                mask = mask[:, None, :, None].expand(-1, self.cross_attn.heads, -1, context.shape[1])
            else:
                mask = mask[:, None, None].expand(-1, self.cross_attn.heads, x.shape[1], -1)
            max_neg_value = -torch.finfo(x.dtype).max
            attn_bias = (~mask) * max_neg_value

        x = x + gate_msa.unsqueeze(1) * self.cross_attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            context=self.norm_context(context),
            attn_bias=attn_bias
        )

        # MLP with modulation
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )

        return x


class DiTUpdateFormer(nn.Module):
    """
    Transformer with DiT-style conditioning for diffusion trajectory prediction.

    Uses factorized space-time attention with timestep conditioning:
    1. Cross-attention to scene tokens (conditioned)
    2. Spatial self-attention (conditioned)
    3. Temporal self-attention (conditioned)

    Each block is modulated by diffusion timestep via adaLN.
    """
    def __init__(
        self,
        depth=6,
        input_dim=320,
        hidden_size=384,
        num_heads=8,
        output_dim=130,
        mlp_ratio=4.0,
        add_space_attn=True,
        p_drop_attn=0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.add_space_attn = add_space_attn

        # Input projection
        self.input_transform = nn.Linear(input_dim, hidden_size, bias=True) if input_dim != hidden_size else nn.Identity()

        # Output head (simple linear, no conditioning)
        self.flow_head = nn.Linear(hidden_size, output_dim, bias=True)

        # DiT blocks
        self.scene_cross_attn_blocks = nn.ModuleList([
            DiTCrossAttnBlock(
                hidden_size,
                context_dim=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=p_drop_attn
            ) for _ in range(depth)
        ])

        if add_space_attn:
            self.space_attn_blocks = nn.ModuleList([
                DiTAttnBlock(
                    hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_class=Attention,
                    drop=p_drop_attn
                ) for _ in range(depth)
            ])

        self.time_attn_blocks = nn.ModuleList([
            DiTAttnBlock(
                hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attn_class=Attention,
                drop=p_drop_attn
            ) for _ in range(depth)
        ])

        # Initialize weights with DiT-style zero-init
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize with DiT's adaLN-Zero: zero-out gate parameters."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Zero-out adaLN modulation layers (gates start at 0)
        for block in self.scene_cross_attn_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        if self.add_space_attn:
            for block in self.space_attn_blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        for block in self.time_attn_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def forward(self, all_tokens, scene_tokens, timestep_cond, scene_mask=None, causal_mask=None):
        """
        Args:
            all_tokens: (B, N, T, C) - Track (+ hand) tokens
            scene_tokens: (B, M, C) - Scene context tokens (ViT features + text)
            timestep_cond: (B, C) - Diffusion timestep conditioning vector
            scene_mask: optional mask for scene tokens
            causal_mask: optional causal mask for temporal attention

        Returns:
            output: (B, N, T, C) - Updated tokens
        """
        B, N, T, C = all_tokens.shape

        # Input projection
        x = self.input_transform(all_tokens)

        # Process through DiT blocks
        for i in range(self.depth):
            # 1. Cross-attention to scene (all NT tokens attend to scene)
            x_flat = x.view(B, N * T, C)
            x_flat = self.scene_cross_attn_blocks[i](
                x_flat, scene_tokens, timestep_cond, mask=scene_mask
            )
            x = x_flat.view(B, N, T, C)

            # 2. Spatial attention (tokens at same t attend to each other)
            if self.add_space_attn:
                x_space = x.permute(0, 2, 1, 3).reshape(B * T, N, C)  # (B*T, N, C)
                # Repeat timestep_cond for each timestep
                timestep_cond_space = timestep_cond.unsqueeze(1).repeat(1, T, 1).view(B * T, C)
                x_space = self.space_attn_blocks[i](x_space, timestep_cond_space)
                x = x_space.view(B, T, N, C).permute(0, 2, 1, 3)  # (B, N, T, C)

            # 3. Temporal attention (same position across t attend)
            x_time = x.permute(0, 1, 2, 3).reshape(B * N, T, C)  # (B*N, T, C)
            timestep_cond_time = timestep_cond.unsqueeze(1).repeat(1, N, 1).view(B * N, C)
            x_time = self.time_attn_blocks[i](x_time, timestep_cond_time, mask=causal_mask)
            x = x_time.view(B, N, T, C)

        # Output head
        output = self.flow_head(x)

        return output
