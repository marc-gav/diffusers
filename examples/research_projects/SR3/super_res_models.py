from diffusers import UNet2DModel
import torch
from torch.nn import functional as F

"""This model has been greatly inspired by https://github.com/openai/guided-diffusion"""

class SuperResUNet(UNet2DModel):
    """
    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """
    def __init__(self, sample_size, in_channels, out_channels, layers_per_block, block_out_channels, down_block_types, up_block_types, *args, **kwargs):
        super().__init__(
            sample_size=sample_size,
            in_channels=in_channels * 2,
            out_channels=out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            *args,
            **kwargs
        )

    def forward(self, noised_high_res, timesteps, low_res, **kwargs):
        _batch_size, _channels, new_height, new_width = noised_high_res.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        noised_high_res = torch.cat([noised_high_res, upsampled], dim=1)
        return super().forward(noised_high_res, timesteps, **kwargs)
