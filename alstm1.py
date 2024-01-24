"""
2024.1.23 17:00
Alstm1
网络结构分为注意力网络和普通网络，普通网络中加入了一层在时间维度上进行卷积的卷积层，一维卷积层在因子维度上沿着时间步（交易日）移动；
注意力网络不变
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
        self.conv1 = nn.Conv1d(in_channels=self.input_size, out_channels=self.input_size, kernel_size=3, padding=1)
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
        inputs = inputs.view(len(inputs), self.input_size, -1)
        print('conv_out', inputs.shape)
        conv_out = F.relu(self.conv1(inputs))
        print('conv_out',conv_out.shape)
        conv_out = conv_out.permute(0, 2, 1)

        out = F.tanh(self.fc_in(conv_out))
        rnn_out, _ = self.rnn(out)
        print('rnn_out', rnn_out.shape)
        attention_score = self.attention_net(rnn_out)
        print('attention_score', attention_score.shape)
        out_att = torch.sum(attention_score, dim=1)
        out = self.fc_out(
            torch.cat((rnn_out[:, -1, :], out_att), dim=1)
        )
        return out[..., 0]


if __name__ == "__main__":
    x = torch.randn(12000, 30, 500)
    model = ALSTMModel(d_feat=500, hidden_size=64, num_layers=2, dropout=0.8, rnn_type="GRU")
    y = model(x)
    y = y.detach().numpy()
    print(y.shape)
