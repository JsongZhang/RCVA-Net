import torch
import torch.nn as nn


class VideoTransformerModel(nn.Module):
    def __init__(self, input_dim=768, nhead=8, num_encoder_layers=12, dim_feedforward=2048, dropout=0.1):
        super(VideoTransformerModel, self).__init__()

        # Transformer的编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,  # 输入特征的维度
            nhead=nhead,  # 多头注意力机制中的头数
            dim_feedforward=dim_feedforward,  # 前馈网络的维度
            dropout=dropout  # dropout比率
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 分类头，这里假设最终目标是分类任务，根据你的需要修改
        self.classifier = nn.Linear(input_dim, 5)  # 假设有10个类别

    def forward(self, x):
        # x的形状应为 (batch_size, seq_length, input_dim)
        # 对序列特征进行编码
        transformed = self.transformer_encoder(x)

        # 取序列的最后一个输出用于分类
        output = self.classifier(transformed[:, -1, :])

        return output

# Example usage:
# model = VideoTransformerModel()
# input_tensor = torch.randn(32, 196, 768)  # 假设batch_size=32, 序列长度=196, 特征维度=768
# output = model(input_tensor)
