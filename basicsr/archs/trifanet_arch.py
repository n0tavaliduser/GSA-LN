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
        self.ln = nn.LayerNorm(num_feat)
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
        desc = self.ln(desc)
        y = self.fc(desc).view(b, c, 1, 1)
        return x * y


class SpatialAttention(nn.Module):
    def __init__(self, num_feat=64, kernel_size=7, deformable_groups=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.proj = nn.Conv2d(num_feat, num_feat, 1, 1, 0)
        self.offset_proj = nn.Conv2d(2, num_feat, 1, 1, 0)
        self.dcn = DCNv2Pack(num_feat, 1, kernel_size, stride=1, padding=padding, deformable_groups=deformable_groups)
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
        att = self.dcn(self.proj(x), self.offset_proj(x_cat))
        att = self.sigmoid(att)
        return x * att


class FrequencyAttention(nn.Module):
    """Frequency magnitude modulation.

    Applies per-frequency magnitude weighting in rFFT2 domain and reconstructs
    a spatial modulation map via irFFT2, then scales features.
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
        spec = torch.fft.rfft2(x, norm='ortho')
        mag = torch.abs(spec)
        w = mag / (torch.mean(mag, dim=(-2, -1), keepdim=True) + 1e-6)
        spec_w = spec * w
        m = torch.fft.irfft2(spec_w, s=x.shape[-2:], norm='ortho')
        m = F.layer_norm(m, m.shape[-2:])
        m = torch.clamp(m, -1.0, 1.0)
        return x * (1 + self.scale * m)

class FrequencyAttentionMB(nn.Module):
    """Multi-band frequency attention with low/mid/high bands."""
    def __init__(self, num_feat, low_thr=0.33, mid_thr=0.66, mask_order='natural', shift_mag=False):
        super().__init__()
        self.low_thr = low_thr
        self.mid_thr = mid_thr
        self.mask_order = mask_order
        self.shift_mag = shift_mag
        self.scale = nn.Parameter(torch.ones(1))
        self.mlp = nn.Sequential(
            nn.Linear(3, 3, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(3, 3, bias=False),
            nn.Sigmoid()
        )
        self.ln3 = nn.LayerNorm(3)
        self.register_buffer('r_norm', torch.tensor([]), persistent=False)
        self.register_buffer('low_mask', torch.tensor([]), persistent=False)
        self.register_buffer('mid_mask', torch.tensor([]), persistent=False)
        self.register_buffer('high_mask', torch.tensor([]), persistent=False)

    def _ensure_masks(self, h, w, device, dtype):
        if self.r_norm.numel() == 0 or self.r_norm.shape != (h, w):
            if self.mask_order == 'natural':
                fy = torch.fft.fftfreq(h, d=1.0).to(device=device, dtype=dtype).view(h, 1)
                fx = torch.fft.fftfreq(w, d=1.0).to(device=device, dtype=dtype).view(1, w)
                r = torch.sqrt(fy ** 2 + fx ** 2)
            else:
                yy = torch.arange(h, device=device, dtype=dtype).view(h, 1)
                xx = torch.arange(w, device=device, dtype=dtype).view(1, w)
                cy = (h - 1) / 2.0
                cx = (w - 1) / 2.0
                r = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            r = r / (r.max() + 1e-6)
            low = (r <= self.low_thr).to(dtype).view(1, 1, h, w)
            mid = ((r > self.low_thr) & (r <= self.mid_thr)).to(dtype).view(1, 1, h, w)
            high = (r > self.mid_thr).to(dtype).view(1, 1, h, w)
            self.r_norm = r
            self.low_mask = low
            self.mid_mask = mid
            self.high_mask = high

    def forward(self, x):
        b, c, h, w = x.shape
        mag = torch.abs(torch.fft.fft2(x, norm='ortho'))
        if self.mask_order != 'natural' and self.shift_mag:
            mag = torch.fft.fftshift(mag, dim=(-2, -1))
        self._ensure_masks(h, w, x.device, mag.dtype)
        low_mean = (mag * self.low_mask).mean(dim=(-2, -1))
        mid_mean = (mag * self.mid_mask).mean(dim=(-2, -1))
        high_mean = (mag * self.high_mask).mean(dim=(-2, -1))
        desc = torch.stack([low_mean, mid_mean, high_mean], dim=-1)
        desc = torch.log1p(desc)
        desc = self.ln3(desc)
        g = self.mlp(desc.view(-1, 3)).view(b, c, 3)
        wsum = (g * desc).sum(dim=-1)
        y = wsum.view(b, c, 1, 1)
        return x * (1 + self.scale * y)


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
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_blocks=16, upscale=2, freq_mb=True, low_thr=0.33, mid_thr=0.66, mask_order='natural', shift_mag=False):
        super().__init__()

        self.conv_in = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.res_blocks = nn.Sequential(*[ResidualBlock(num_feat) for _ in range(num_blocks)])
        self.conv_mid = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.ca = ChannelAttention(num_feat)
        self.sa = SpatialAttention(num_feat)
        self.fa = FrequencyAttentionMB(num_feat, low_thr=low_thr, mid_thr=mid_thr, mask_order=mask_order, shift_mag=shift_mag) if freq_mb else FrequencyAttention()

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