"""
2024.2.1 15:00
TRANSFORMER

Transformer模型，因子数据通过一个特征提取的全连接层后，所有数据（不使用掩码）通过位置编码层（即增强输入数据，使其带有时间序列信息），
再送入Transformer编码器层，最后通过一个全连接层（解码器）输出。

位置编码层的作用是为输入数据增加时间序列信息，以便Transformer编码器层能够利用。

Transformer编码器由多个Transformer编码器层组成，包括自多头注意力层、前馈神经网络层和残差连接。
不同于传统的Transformer多层解码器，本模型不需要预测序列信息，因此采用全连接层直接将编码器的输出进行解码，得到预测值。
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size=64, max_len=120):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2, dtype=torch.float32) * (-np.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数列
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数列
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, vec_x):
        vec_x = vec_x + self.pe[:vec_x.size(0), :]  # [seq_len, batch_size, d_model]
        return vec_x


class Transformer(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, transformer_layers=2, num_heads=4, dropout=0.0, **kwargs):
        super(Transformer, self).__init__()
        self.input_size = d_feat
        self.feature_layer = nn.Linear(d_feat, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size=hidden_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=transformer_layers)
        self.decoder_layer = nn.Linear(hidden_size, 1)

    def forward(self, input_x):
        input_x = self.feature_layer(input_x)
        input_x = input_x.transpose(1, 0)

        mask = None  # 所有历史特征都参与计算
        input_x_pos = self.pos_encoder(input_x)
        enc_x = self.transformer_encoder(input_x_pos, mask)
        dec_x = self.decoder_layer(enc_x.transpose(1, 0)[:, -1, :])

        return dec_x[...]


# test
if __name__ == '__main__':
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    x = torch.randn(1000, 60, 100).to(device)
    model = Transformer(d_feat=100, hidden_size=64, transformer_layers=2, num_heads=4, dropout=0.3, 双向=True).to(
        device)
    out = model(x)
    print(out.shape)
