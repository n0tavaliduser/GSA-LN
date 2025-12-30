import torch
from torch import nn as nn

from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class VDSR(nn.Module):
    """VDSR network structure.

    Paper: Accurate Image Super-Resolution Using Very Deep Convolutional Networks.
    Ref git repo: https://github.com/twtygqyy/pytorch-vdsr

    VDSR performs SR on pre-upscaled (bicubic) images, learning the residual.
    
    Args:
        num_in_ch (int): Channel number of inputs. Default: 1.
        num_out_ch (int): Channel number of outputs. Default: 1.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_block (int): Number of conv layers (default 18 conv layers total = 20 layers including first and last).
            Default: 18.
    """

    def __init__(self,
                 num_in_ch=1,
                 num_out_ch=1,
                 num_feat=64,
                 num_block=18):
        super(VDSR, self).__init__()

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=True)
        
        # Build body with num_block conv layers
        body = []
        for _ in range(num_block):
            body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True))
            body.append(nn.ReLU(inplace=True))
        self.body = nn.Sequential(*body)
        
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1, bias=True)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # VDSR learns residual, so output = input + residual
        residual = self.relu(self.conv_first(x))
        residual = self.body(residual)
        residual = self.conv_last(residual)
        
        out = x + residual
        return out
