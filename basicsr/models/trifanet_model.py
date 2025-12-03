import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY

class ResidualBlock(nn.Module):
    """EDSR-style residual block tanpa batchnorm"""
    def __init__(self, num_feat=64):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        return x + res * 0.1  # residual scaling

class ChannelAttention(nn.Module):
    """Channel Attention sederhana (SE-Block style)"""
    def __init__(self, num_feat=64, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_feat, num_feat // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_feat // reduction, num_feat, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SpatialAttention(nn.Module):
    """Spatial Attention sederhana (CBAM style)"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(x_cat))
        return x * attention


class FrequencyAttention(nn.Module):
    """Frequency Attention sederhana pakai FFT magnitude"""
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # FFT2 real-valued magnitude sebagai weighting
        fft_mag = torch.abs(torch.fft.rfft2(x, norm='ortho'))
        fft_mag = fft_mag / (torch.mean(fft_mag, dim=(-2, -1), keepdim=True) + 1e-6)
        fft_mag = F.interpolate(fft_mag, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return x * (1 + self.scale * fft_mag)

@ARCH_REGISTRY.register()
class TriFANet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_blocks=16, upscale=2):
        super().__init__()

        # Head
        self.conv_in = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        # Body (EDSR residual backbone)
        self.res_blocks = nn.Sequential(*[ResidualBlock(num_feat) for _ in range(num_blocks)])
        self.conv_mid = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # Triple Attention
        self.ca = ChannelAttention(num_feat)
        self.sa = SpatialAttention()
        self.fa = FrequencyAttention()

        # Reconstruction
        self.upsampler = nn.Sequential(
            nn.Conv2d(num_feat, num_feat * (upscale ** 2), 3, 1, 1),
            nn.PixelShuffle(upscale),
            nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        )

    def forward(self, x):
        # Head
        feat = self.conv_in(x)
        res = self.res_blocks(feat)
        res = self.conv_mid(res)
        feat = feat + res

        # Apply triple attention
        feat = self.ca(feat)
        feat = self.sa(feat)
        feat = self.fa(feat)

        # Reconstruction / Upsampling
        out = self.upsampler(feat)
        return out
