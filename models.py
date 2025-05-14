from diffusers import UNet2DModel,UNet2DConditionModel,AutoencoderKL
import torch
from torch import nn
from config import TrainingConfig

config = TrainingConfig()

vae = AutoencoderKL(
    in_channels = config.in_channels,
    out_channels = config.out_channels,
    latent_channels= config.latent_channels,
    sample_size=config.target_size,
    block_out_channels=(16, 32, 64),
    norm_num_groups=4,
    down_block_types=("DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D",),
    up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D")
)


unet = UNet2DConditionModel(
    in_channels=config.latent_channels,
    out_channels=config.latent_channels,
    sample_size=16,
    
    layers_per_block=2,
    down_block_types=('AttnDownBlock2D', 'AttnDownBlock2D'),
    up_block_types=('AttnUpBlock2D', 'UpBlock2D'),
    block_out_channels=(128, 256),
    norm_num_groups=1,

    num_class_embeds=2,
    time_embedding_act_fn='silu',
    cross_attention_dim=256,
    class_embeddings_concat=True,
)

# unet = UNet2DConditionModel(
#     sample_size=config.target_size,  # 图片的尺寸
#     in_channels=1,  # 输入图像的通道数
#     out_channels=1,  # 输出的通道数
#     layers_per_block=2,  # 残差层的块数
#     block_out_channels=(64, 128, 128, 256, 256, 512),  # 调整通道数匹配块数量
#     down_block_types=(
#         "DownBlock2D",
#         "DownBlock2D",  # 下采样
#         "DownBlock2D",
#         "DownBlock2D",
#         "DownBlock2D",
#         "DownBlock2D",  # 添加自注意力的下采样
#     ),
#     up_block_types=(
#         "UpBlock2D",  # 对应 CrossAttnDownBlock2D
#         "UpBlock2D",
#         "UpBlock2D",
#         "UpBlock2D",
#         "UpBlock2D",
#         "UpBlock2D"
#     ),
#     class_embed_type="timestep",
#     num_class_embeds=config.num_classes,
#     time_embedding_act_fn='silu',
#     class_embeddings_concat=True,
#     cross_attention_dim=256
# )
