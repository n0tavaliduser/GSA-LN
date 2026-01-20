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
        
        # Input feature projection
        self.proj = nn.Conv2d(num_feat, num_feat, 1, 1, 0)
        
        # DCN Offset Projection
        # Output channel offset typically: deformable_groups * 2 * kernel_size * kernel_size
        # Note: DCNv2Pack often handles channel matching internally.
        # Zero Initialization is MANDATORY for stability.
        self.offset_proj = nn.Conv2d(2, num_feat, 1, 1, 0)
        
        self.dcn = DCNv2Pack(num_feat, num_feat, kernel_size, stride=1, padding=padding, deformable_groups=deformable_groups)
        
        # --- FIX: ZERO INITIALIZATION ---
        self._init_offset()

    def _init_offset(self):
        # Initialize offset and bias to 0 to ensure DCN stability at initialization
        self.offset_proj.weight.data.zero_()
        self.offset_proj.bias.data.zero_()

    def forward(self, x):
        """
        DCN Spatial Attention.
        Output: Aligned Features (Raw), not Mask.
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        
        # Predict pixel displacement offsets
        offset = self.offset_proj(x_cat)
        
        # Perform Deformable Convolution
        # Output: Aligned features
        feat_aligned = self.dcn(self.proj(x), offset)
        
        return feat_aligned


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

class FourierFeatureMapping(nn.Module):
    """Fourier Feature Mapping for high-frequency coordinate encoding."""
    def __init__(self, input_dim=2, mapping_size=64, scale=10):
        super().__init__()
        self.B = nn.Parameter(torch.randn(input_dim, mapping_size) * scale, requires_grad=False)

    def forward(self, x):
        # x: (..., 2)
        x_proj = (2 * torch.pi * x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class FrequencyAttentionLearnable(nn.Module):
    """
    Frequency Attention with Top-Tier Upgrades (Hybrid Option 1 + Option 3):
    1. Fourier Feature Mapping (Positional Encoding) - for sharp mask boundaries.
    2. Content-Aware Gating (Dynamic Context) - for image-adaptive masking.
    """
    def __init__(self, num_feat, hidden_dim=64):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        
        # --- 1. Fourier Feature Mapping ---
        self.pos_enc_dim = hidden_dim 
        self.pos_encoder = FourierFeatureMapping(input_dim=2, mapping_size=self.pos_enc_dim // 2, scale=10)
        
        # --- 2. Context Encoder (Content-Aware) ---
        self.context_pool = nn.AdaptiveAvgPool2d(1)
        self.context_mlp = nn.Sequential(
            nn.Linear(num_feat, hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # --- 3. Mask Predictor ---
        # Input: EncodedCoords + Context
        total_input_dim = self.pos_enc_dim + hidden_dim
        
        self.mask_predictor = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, 3) 
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
        y = torch.linspace(-1, 1, h, device=device, dtype=dtype)
        x = torch.linspace(0, 1, w, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        return torch.stack([grid_x, grid_y], dim=-1)

    def forward(self, x, fft_mag=None):
        b, c, h, w = x.shape
        
        if fft_mag is None:
             spec = torch.fft.rfft2(x, norm='ortho')
             fft_mag = torch.abs(spec)
        
        mag = fft_mag
        h_spec, w_spec = mag.shape[-2], mag.shape[-1]

        # 1. Get Coordinates & Apply Fourier Mapping
        coords = self._get_coord_grid(h_spec, w_spec, x.device, mag.dtype)
        coords_enc = self.pos_encoder(coords) # (H, W, pos_enc_dim)
        
        # 2. Get Image Context
        ctx = self.context_pool(x).view(b, -1)      
        ctx = self.context_mlp(ctx)                 # (B, hidden_dim)
        
        # 3. Combine (Dynamic Masking)
        coords_expanded = coords_enc.unsqueeze(0).expand(b, -1, -1, -1)
        ctx_expanded = ctx.view(b, 1, 1, -1).expand(-1, h_spec, w_spec, -1)
        mlp_input = torch.cat([coords_expanded, ctx_expanded], dim=-1)
        
        # 4. Predict Masks
        mask_weights = self.mask_predictor(mlp_input) # (B, H, W, 3)
        mask_weights = F.softmax(mask_weights, dim=-1)

        # 5. Extract specific masks
        # (B, H, W, 3) -> (3, B, 1, H, W)
        masks = mask_weights.permute(3, 0, 1, 2).unsqueeze(2)
        low_mask = masks[0]
        mid_mask = masks[1]
        high_mask = masks[2]

        # 6. Weighted Pooling
        low_mean = (mag * low_mask).sum(dim=(-2, -1)) / (low_mask.sum(dim=(-2, -1)) + 1e-6)
        mid_mean = (mag * mid_mask).sum(dim=(-2, -1)) / (mid_mask.sum(dim=(-2, -1)) + 1e-6)
        high_mean = (mag * high_mask).sum(dim=(-2, -1)) / (high_mask.sum(dim=(-2, -1)) + 1e-6)

        # 7. Attention Generation
        desc = torch.stack([low_mean, mid_mean, high_mean], dim=-1)
        desc = torch.log1p(desc)
        desc = self.ln3(desc)
        
        g = self.mlp_feat(desc.view(-1, 3)).view(b, c, 3)
        wsum = (g * desc).sum(dim=-1)
        y = wsum.view(b, c, 1, 1)
        
        return x * (1 + self.scale * y)


class ContrastAwarePixelAttention(nn.Module):
    """
    PSNR BOOSTER: Contrast-Aware Pixel Attention (CAPA).
    Forces the model to focus on pixels with the highest MSE error (edges/textures).
    """
    def __init__(self, num_feat):
        super().__init__()
        self.conv_c = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Calculate local contrast as an attention guide
        # High contrast = area with high potential MSE/PSNR error
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Contrast map (Max - Avg)
        contrast = max_out - avg_out
        
        # Predict pixel-wise weights based on original features and contrast
        mask = self.conv_c(x * contrast) 
        
        return x * mask


class LearnableResidualScaling(nn.Module):
    """
    PSNR BOOSTER: Replaces static 0.1 multiplication with a learnable parameter.
    Significantly improves long-term stability and convergence.
    """
    def __init__(self, num_feat):
        super().__init__()
        self.res_scale = nn.Parameter(torch.ones(1, num_feat, 1, 1) * 0.1)

    def forward(self, x, res):
        return x + (res * self.res_scale)


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

        # Added: Pixel-Wise Booster (CAPA) just before upsampler
        self.pa_booster = ContrastAwarePixelAttention(num_feat)
        
        # Changed: Use Learnable Scaling (LRS) for Global Residual
        self.lrs = LearnableResidualScaling(num_feat)

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

        # Step 1 (Channel)
        feat_ca = self.ca(feat_backbone)

        # Step 2 (Spatial/DCN)
        feat_sa = self.sa(feat_ca)

        # Step 3 (Frequency/Fourier)
        # Recompute FFT on the output of Spatial Step
        fft_spec = torch.fft.rfft2(feat_sa, norm='ortho')
        fft_mag = torch.abs(fft_spec)
        
        feat_fa = self.fa(feat_sa, fft_mag=fft_mag)

        # BEFORE UPSAMPLER: Apply final CAPA refinement
        feat_refined = self.pa_booster(feat_fa)

        # USE Learnable Scaling for final fusion
        feat_final = self.lrs(feat_backbone, feat_refined)

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
