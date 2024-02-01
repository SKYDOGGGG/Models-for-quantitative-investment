import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from TRANSFORMER import PositionalEncoding


class FusionLayer(nn.Module):
    def __init__(self, input_dim):
        super(FusionLayer, self).__init__()
        self.weight = nn.Parameter(torch.tensor([0.5, 0.5]))  # 初始化权重参数

    def forward(self, rnn_out, transformer_out):
        weights = F.softmax(self.weight, dim=0)  # 转换为权重分布
        fused_out = weights[0] * rnn_out + weights[1] * transformer_out
        return fused_out


class TransformerRNN(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, rnn_layers=2, rnn_type='GRU', transformer_layers=2, num_heads=2,
                 dropout=0.3, 双向=False, **kwargs):
        super(TransformerRNN, self).__init__()
        self.input_size = d_feat
        self.hid_size = hidden_size
        self.rnn_type = rnn_type.lower()
        self.rnn_layers = rnn_layers
        self.dropout = dropout
        self.bidirectional = 双向
        self.bi_ind = 1 + int(双向)

        self.feature_layer = nn.Linear(d_feat, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size=hidden_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=transformer_layers)
        self.decoder_layer = nn.Linear(hidden_size, 1)

        try:
            klass = getattr(nn, self.rnn_type.upper())
        except Exception as e:
            raise ValueError("unknown rnn_type `%s`" % self.rnn_type) from e

        self.rnn = klass(
            input_size=self.hid_size,
            hidden_size=self.hid_size,
            num_layers=self.rnn_layers,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        self.attention = nn.Sequential(
            nn.Linear(in_features=self.hid_size * self.bi_ind, out_features=(self.hid_size * self.bi_ind) // 2),
            nn.Dropout(p=dropout),
            nn.Tanh(),
            nn.Linear(in_features=(self.hid_size * self.bi_ind) // 2, out_features=1),
            nn.Softmax(dim=1),
        )

        self.fc_rnn_out = nn.Linear(in_features=self.hid_size * self.bi_ind, out_features=1)
        self.FusionLayer = FusionLayer(input_dim=self.hid_size * self.bi_ind)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)

        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if "weight" in name:
                    init.xavier_uniform_(param)
                elif "bias" in name:
                    init.zeros_(param)

    def forward(self, input_x):
        # feature
        input_x = self.feature_layer(input_x)

        # rnn
        rnn_out, _ = self.rnn(input_x)
        attention_score = self.attention(rnn_out)
        attention_out = torch.sum(torch.mul(rnn_out, attention_score), dim=1)
        rnn_out = self.fc_rnn_out(attention_out)

        # transformer
        input_x = input_x.transpose(1, 0)

        mask = None  # 所有历史特征都参与计算
        input_x_pos = self.pos_encoder(input_x)
        enc_x = self.transformer_encoder(input_x_pos, mask)
        dec_x = self.decoder_layer(enc_x.transpose(1, 0)[:, -1, :])

        final_out = self.FusionLayer(rnn_out, dec_x)

        return final_out


# test
if __name__ == '__main__':
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    x = torch.randn(1000, 60, 120).to(device)
    model = TransformerRNN(d_feat=120, hidden_size=64, rnn_layers=2, rnn_type='gru', transformer_layers=2, num_heads=2,
                           dropout=0.3, 双向=False).to(device)
    out = model(x)
    print(out.shape)
