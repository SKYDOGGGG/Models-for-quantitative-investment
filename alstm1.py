"""
2024.1.23 17:00
Alstm1
网络结构分为注意力网络和普通网络，普通网络中加入了一层在时间维度上进行卷积的卷积层，卷积后通道数不变（仍然是六十天），然后传入线性层和rnn层；注意力网络不变
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ALSTMModel(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, rnn_type="GRU"):
        super().__init__()
        self.hid_size = hidden_size
        self.input_size = d_feat
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.rnn_layer = num_layers
        self._build_model()

    def _build_model(self):
        try:
            klass = getattr(nn, self.rnn_type.upper())
        except Exception as e:
            raise ValueError("unknown rnn_type `%s`" % self.rnn_type) from e
        self.net = nn.Sequential()
        self.conv1 = nn.Conv1d(in_channels=60, out_channels=60, kernel_size=3, padding=1)
        self.fc_in = nn.Linear(in_features=self.input_size, out_features=self.hid_size)

        self.rnn = klass(
            input_size=self.hid_size,
            hidden_size=self.hid_size,
            num_layers=self.rnn_layer,
            batch_first=True,
            dropout=self.dropout,
        )
        self.fc_out = nn.Linear(in_features=self.hid_size * 2, out_features=1)
        self.att_fc = nn.Linear(in_features=self.hid_size, out_features=self.hid_size)

    def attention_net(self, rnn_out):
        attention_weights = F.softmax(self.att_fc(rnn_out), dim=1)
        weighted_hidden = torch.bmm(attention_weights.transpose(1, 2), rnn_out)
        return weighted_hidden.squeeze(1)

    def forward(self, inputs):
        # inputs: [batch_size, input_size,input_day]
        inputs = inputs.view(len(inputs), -1, self.input_size)
        conv_out = F.relu(self.conv1(inputs))
        out = F.tanh(self.fc_in(conv_out))

        rnn_out, _ = self.rnn(out)  # [batch, seq_len, num_directions * hidden_size]
        print('rnn_out', rnn_out.shape)
        attention_score = self.attention_net(rnn_out)  # [batch, seq_len, 1]
        print('attention_score', attention_score.shape)
        out_att = torch.sum(attention_score, dim=1)
        out = self.fc_out(
            torch.cat((rnn_out[:, -1, :], out_att), dim=1)
        )  # [batch, seq_len, num_directions * hidden_size] -> [batch, 1]
        return out[..., 0]


if __name__ == "__main__":
    x = torch.randn(5000, 60, 200)
    model = ALSTMModel(d_feat=200, hidden_size=64, num_layers=2, dropout=0.0, rnn_type="GRU")
    y = model(x)
    y = y.detach().numpy()
    print(y.shape)
