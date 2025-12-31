import torch
from torch import nn as nn

from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class SRCNN(nn.Module):
    """SRCNN network structure.

    Paper: Image Super-Resolution Using Deep Convolutional Networks (ECCV 2014).
    Ref git repo: https://github.com/yjn870/SRCNN-pytorch

    SRCNN performs SR on pre-upscaled (bicubic) images.
    Original architecture: 9-1-5 (kernel sizes) with 64-32-channels configuration.
    
    Args:
        num_in_ch (int): Channel number of inputs. Default: 1.
        num_out_ch (int): Channel number of outputs. Default: 1.
        num_feat (int): Channel number of first layer features. Default: 64.
        num_feat2 (int): Channel number of second layer features. Default: 32.
    """

    def __init__(self,
                 num_in_ch=1,
                 num_out_ch=1,
                 num_feat=64,
                 num_feat2=32):
        super(SRCNN, self).__init__()

        # Patch extraction and representation
        self.conv1 = nn.Conv2d(num_in_ch, num_feat, kernel_size=9, padding=4)
        # Non-linear mapping
        self.conv2 = nn.Conv2d(num_feat, num_feat2, kernel_size=1, padding=0)
        # Reconstruction (some implementations use 5x5)
        self.conv3 = nn.Conv2d(num_feat2, num_out_ch, kernel_size=5, padding=2)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out
