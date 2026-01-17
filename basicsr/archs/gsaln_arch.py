import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import DCNv2Pack


class ResidualBlock(nn.Module):
    """Residual block without batch normalization.

    Structure: Conv(3x3) -> ReLU -> Conv(3x3) -> SE Block -> Residual Scaling (0.1).

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

        # Simple Channel Attention (SE Layer)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // 16, num_feat, 1),
            nn.Sigmoid()
        )

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

        # Apply SE Attention
        res = res * self.se(res)

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

    def forward(self, x, fft_mag=None):
        """Compute channel gating weights from frequency descriptors.

        Args:
            x (Tensor): Feature map (B, C, H, W).
            fft_mag (Tensor, optional): Precomputed rFFT magnitude.

        Returns:
            Tensor: Modulated feature map (B, C, H, W).
        """
        b, c, _, _ = x.size()
        b, c, _, _ = x.size()
        if fft_mag is None:
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

    def forward(self, x, fft_mag=None):
        """Apply frequency-based modulation.

        Args:
            x (Tensor): Feature map (B, C, H, W).
            fft_mag (Tensor, optional): Precomputed rFFT magnitude (ignored here as we need complex spec).

        Returns:
            Tensor: Modulated feature map (B, C, H, W).
        """
        # We need the complex spectrum (spec) for irfft2 reconstruction.
        # Even if fft_mag is passed, we cannot recover phase, so we must recompute rfft2.
        spec = torch.fft.rfft2(x, norm='ortho')
        mag = torch.abs(spec)
        w = mag / (torch.mean(mag, dim=(-2, -1), keepdim=True) + 1e-6)
        spec_w = spec * w
        m = torch.fft.irfft2(spec_w, s=x.shape[-2:], norm='ortho')
        m = F.layer_norm(m, m.shape[-2:])
        m = torch.clamp(m, -1.0, 1.0)
        return x * (1 + self.scale * m)

class FrequencyAttentionLearnable(nn.Module):
    """
    Frequency Attention with Adaptive Mask (Learnable).
    
    Instead of using rigid thresholds and circular shapes,
    this module uses a small MLP to predict frequency masks
    based on coordinates (u, v).
    """
    def __init__(self, num_feat, hidden_dim=32):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        
        # Mask Predictor MLP: Input (x, y) -> Output (Low, Mid, High probabilities)
        self.mask_predictor = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3) # 3 bands: Low, Mid, High
        )
        
        # Feature MLP
        self.mlp_feat = nn.Sequential(
            nn.Linear(3, 3, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(3, 3, bias=False),
            nn.Sigmoid()
        )
        self.ln3 = nn.LayerNorm(3)

    def _get_coord_grid(self, h, w, device, dtype):
        # Create normalized coordinate grid [-1, 1]
        # For rfft2, X axis is only [0, 1] because half spectrum
        y = torch.linspace(-1, 1, h, device=device, dtype=dtype)
        x = torch.linspace(0, 1, w, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        return torch.stack([grid_x, grid_y], dim=-1) # Shape: (H, W, 2)

    def forward(self, x, fft_mag=None):
        b, c, h, w = x.shape
        
        # 1. Compute FFT Magnitude (if not provided)
        if fft_mag is None:
             spec = torch.fft.rfft2(x, norm='ortho')
             fft_mag = torch.abs(spec)
        
        mag = fft_mag
        h_spec, w_spec = mag.shape[-2], mag.shape[-1] # w_spec is usually w//2 + 1

        # 2. Generate Masks Dynamically
        # Coordinate grid
        coords = self._get_coord_grid(h_spec, w_spec, x.device, mag.dtype)
        
        # Predict mask weights for each frequency pixel
        # Output: (H_spec, W_spec, 3) -> Softmax so total probability = 1
        mask_weights = self.mask_predictor(coords) 
        mask_weights = F.softmax(mask_weights, dim=-1) # Last dim: [Low, Mid, High]

        # Separate into 3 physical masks
        # Permute so channel is first for broadcasting: (3, 1, 1, H, W)
        masks = mask_weights.permute(2, 0, 1).unsqueeze(1).unsqueeze(1) 
        low_mask = masks[0]
        mid_mask = masks[1]
        high_mask = masks[2]

        # 3. Weighted Pooling (Soft Masking)
        # Using sum(mag * soft_mask) / sum(soft_mask) for weighted average
        low_mean = (mag * low_mask).sum(dim=(-2, -1)) / (low_mask.sum(dim=(-2, -1)) + 1e-6)
        mid_mean = (mag * mid_mask).sum(dim=(-2, -1)) / (mid_mask.sum(dim=(-2, -1)) + 1e-6)
        high_mean = (mag * high_mask).sum(dim=(-2, -1)) / (high_mask.sum(dim=(-2, -1)) + 1e-6)

        # 4. Attention Generation
        desc = torch.stack([low_mean, mid_mean, high_mean], dim=-1)
        desc = torch.log1p(desc)
        desc = self.ln3(desc)
        
        g = self.mlp_feat(desc.view(-1, 3)).view(b, c, 3)
        wsum = (g * desc).sum(dim=-1)
        y = wsum.view(b, c, 1, 1)
        
        return x * (1 + self.scale * y)


@ARCH_REGISTRY.register()
class GSALN(nn.Module):
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
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_blocks=16, upscale=2, freq_mb=True):
        super().__init__()

        self.conv_in = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.res_blocks = nn.Sequential(*[ResidualBlock(num_feat) for _ in range(num_blocks)])
        self.conv_mid = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.ca = ChannelAttention(num_feat)
        self.sa = SpatialAttention(num_feat)
        self.fa = FrequencyAttentionLearnable(num_feat) if freq_mb else FrequencyAttention()

        self.fusion = nn.Conv2d(num_feat * 3, num_feat, 1, 1, 0)

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
        feat_backbone = feat + res

        # Precompute FFT (rfft2) for efficiency
        # Using rfft2 reduces computation by ~2x compared to fft2 and is sufficient for magnitude stats
        fft_spec = torch.fft.rfft2(feat_backbone, norm='ortho')
        fft_mag = torch.abs(fft_spec)

        feat_ca = self.ca(feat_backbone, fft_mag=fft_mag)
        feat_sa = self.sa(feat_backbone)
        feat_fa = self.fa(feat_backbone, fft_mag=fft_mag)

        feat_cat = torch.cat([feat_ca, feat_sa, feat_fa], dim=1)
        feat_fused = self.fusion(feat_cat)
        feat_final = feat_fused + feat_backbone

        out = self.upsampler(feat_final)
        return out
"""GSA-LN architectural components.

This module defines the building blocks used by GSALN:
- ResidualBlock: EDSR-style residual block (no batchnorm)
- ChannelAttention: Frequency Channel Attention (FCA-FFT) for per-channel gating
- SpatialAttention: Deformed Spatial Attention (DSA) using DCNv2Pack
- FrequencyAttention: Simple frequency magnitude modulation
- GSALN: Super-resolution network composed of the above modules
"""
