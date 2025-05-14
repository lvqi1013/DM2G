import torch
from diffusers import AutoencoderKL

latent_channels = 4  # 举例
vae = AutoencoderKL(
    in_channels=1,
    out_channels=1,
    latent_channels=latent_channels,
    sample_size=64,
    block_out_channels=(16, 32, 64),
    norm_num_groups=4,
    down_block_types=("DownEncoderBlock2D",) * 3,
    up_block_types=("UpDecoderBlock2D",) * 3
)

x = torch.randn(1, 1, 64, 64)
latent = vae.encode(x).latent_dist.mean
print(latent.shape)  # → torch.Size([1, 4, 8, 8])
