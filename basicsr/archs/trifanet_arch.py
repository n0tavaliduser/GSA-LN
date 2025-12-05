import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import DCNv2Pack


class ResidualBlock(nn.Module):
    """Residual block without batch normalization.

    Structure: Conv(3x3) → ReLU → Conv(3x3) with residual scaling (0.1).

    Args:
        num_feat (int): Number of feature channels.

    Returns:
        Tensor: Feature map with residual added and scaled.
    """
    def __init__(self, num_feat=64):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.

        Args:
            x (Tensor): Input of shape (B, C=num_feat, H, W).

        Returns:
            Tensor: Output of shape (B, C=num_feat, H, W).
        """
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        return x + res * 0.1


class ChannelAttention(nn.Module):
    """Frequency Channel Attention (FCA-FFT).

    Builds a per-channel descriptor from frequency domain via rFFT2 magnitude
    (log-compressed), then applies a two-layer MLP (SE-style) to produce
    sigmoid gating weights for channel-wise modulation.

    Args:
        num_feat (int): Number of channels in feature map.
        reduction (int): Reduction ratio for hidden MLP size.
    """
    def __init__(self, num_feat=64, reduction=16):
        super().__init__()
        hidden = max(1, num_feat // reduction)
        self.fc = nn.Sequential(
            nn.Linear(num_feat, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_feat, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Compute channel gating weights from frequency descriptors.

        Args:
            x (Tensor): Feature map (B, C, H, W).

        Returns:
            Tensor: Modulated feature map (B, C, H, W).
        """
        b, c, _, _ = x.size()
        fft_mag = torch.abs(torch.fft.rfft2(x, norm='ortho'))
        desc = torch.log1p(fft_mag).mean(dim=(-2, -1))
        y = self.fc(desc).view(b, c, 1, 1)
        return x * y


class SpatialAttention(nn.Module):
    """Deformed Spatial Attention (DSA).

    Uses DCNv2Pack(2→1) on concatenated average and max maps to generate a
    spatial attention mask with deformable sampling, improving alignment to
    local structures.

    Args:
        kernel_size (int): Kernel size for DCN.
        deformable_groups (int): Number of deformable groups.
    """
    def __init__(self, kernel_size=7, deformable_groups=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.dcn = DCNv2Pack(2, 1, kernel_size, stride=1, padding=padding, deformable_groups=deformable_groups)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Compute spatial attention mask and modulate features.

        Args:
            x (Tensor): Feature map (B, C, H, W).

        Returns:
            Tensor: Modulated feature map (B, C, H, W).
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        att = self.dcn(x_cat, x_cat)
        att = self.sigmoid(att)
        return x * att


class FrequencyAttention(nn.Module):
    """Frequency magnitude modulation.

    Normalizes rFFT2 magnitude across spatial dims, resizes to original size
    and applies a learnable scale to modulate features.
    """
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        """Apply frequency-based modulation.

        Args:
            x (Tensor): Feature map (B, C, H, W).

        Returns:
            Tensor: Modulated feature map (B, C, H, W).
        """
        fft_mag = torch.abs(torch.fft.rfft2(x, norm='ortho'))
        fft_mag = fft_mag / (torch.mean(fft_mag, dim=(-2, -1), keepdim=True) + 1e-6)
        fft_mag = F.interpolate(fft_mag, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return x * (1 + self.scale * fft_mag)


@ARCH_REGISTRY.register()
class TriFANet(nn.Module):
    """Triple-Attention Super-Resolution network.

    Pipeline: head conv → residual backbone → mid conv + skip →
    ChannelAttention → SpatialAttention → FrequencyAttention → upsampler.

    Args:
        num_in_ch (int): Number of input channels.
        num_out_ch (int): Number of output channels.
        num_feat (int): Feature channels.
        num_blocks (int): Number of residual blocks in backbone.
        upscale (int): Upscaling factor for PixelShuffle.
    """
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_blocks=16, upscale=2):
        super().__init__()

        self.conv_in = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.res_blocks = nn.Sequential(*[ResidualBlock(num_feat) for _ in range(num_blocks)])
        self.conv_mid = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.ca = ChannelAttention(num_feat)
        self.sa = SpatialAttention()
        self.fa = FrequencyAttention()

        self.upsampler = nn.Sequential(
            nn.Conv2d(num_feat, num_feat * (upscale ** 2), 3, 1, 1),
            nn.PixelShuffle(upscale),
            nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        )

    def forward(self, x):
        """Forward SR inference.

        Args:
            x (Tensor): Low-resolution input (B, num_in_ch, H, W).

        Returns:
            Tensor: Super-resolved output (B, num_out_ch, H*upscale, W*upscale).
        """
        feat = self.conv_in(x)
        res = self.res_blocks(feat)
        res = self.conv_mid(res)
        feat = feat + res

        feat = self.ca(feat)
        feat = self.sa(feat)
        feat = self.fa(feat)

        out = self.upsampler(feat)
        return out
"""TriFA-Net architectural components.

This module defines the building blocks used by TriFANet:
- ResidualBlock: EDSR-style residual block (no batchnorm)
- ChannelAttention: Frequency Channel Attention (FCA-FFT) for per-channel gating
- SpatialAttention: Deformed Spatial Attention (DSA) using DCNv2Pack
- FrequencyAttention: Simple frequency magnitude modulation
- TriFANet: Super-resolution network composed of the above modules
"""