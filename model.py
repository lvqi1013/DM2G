from diffusers import UNet2DModel,UNet2DConditionModel
import torch
from torch import nn
from config import TrainingConfig

config = TrainingConfig()


supervised_model = UNet2DConditionModel(
    sample_size=config.target_size,  # 图片的尺寸
    in_channels=1,  # 输入图像的通道数
    out_channels=1,  # 输出的通道数
    layers_per_block=2,  # 残差层的块数
    block_out_channels=(64, 128, 128, 256, 256, 512),  # 调整通道数匹配块数量
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",  # 下采样
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "CrossAttnDownBlock2D",  # 添加自注意力的下采样
    ),
    up_block_types=(
        "CrossAttnUpBlock2D",  # 对应 CrossAttnDownBlock2D
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D"
    ),
    class_embed_type="timestep",
    num_class_embeds=config.num_classes,
    time_embedding_act_fn='silu',
    class_embeddings_concat=True,
    cross_attention_dim=256
)
