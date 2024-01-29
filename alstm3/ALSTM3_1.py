"""
2024.1.29 16:00
ALSTM3_1

在ALSTM_3的基础上进行修改，如减少卷积层，以减少过拟合问题；同时引入双向rnn，以提高模型性能。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, bi_indicator=2, num_heads=4):
        super(MultiHeadAttention, self).__init__()
        self.hid_size = input_size * bi_indicator
        self.num_heads = num_heads
        self.attention_size = self.hid_size // self.num_heads
        # query, key, value的线性层
        self.w_q = nn.Linear(in_features=self.hid_size, out_features=self.hid_size)
        self.w_kv = nn.Linear(in_features=self.hid_size, out_features=self.hid_size)

        self.w_o = nn.Linear(in_features=self.hid_size, out_features=self.hid_size)

    def forward(self, query, keys, value):
        batch_size = query.size(0)

        query = self.w_q(query).view(batch_size, -1, self.num_heads, self.attention_size)
        keys = self.w_kv(keys).view(batch_size, -1, self.num_heads, self.attention_size)
        value = self.w_kv(value).view(batch_size, -1, self.num_heads, self.attention_size)

        attention_score = torch.mul(query, keys)
        attention_score = attention_score / np.sqrt(self.attention_size)
        attention_weights = F.softmax(attention_score, dim=1)

        context_vector = torch.mul(attention_weights, value)
        context_vector = context_vector.contiguous().view(batch_size, -1, self.num_heads * self.attention_size)
        context_vector = self.w_o(context_vector)
        return context_vector


class ALSTM3_1(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.5,
                 rnn_type="GRU", 双向=False):
        super().__init__()
        self.hid_size = hidden_size
        self.input_size = d_feat
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.rnn_layer = num_layers
        self.bidirectional = 双向
        self.bi_ind = 1 + int(双向)
        self.attention = MultiHeadAttention(input_size=self.hid_size, bi_indicator=self.bi_ind, num_heads=4)
        self._build_model()

    def _build_model(self):
        try:
            klass = getattr(nn, self.rnn_type.upper())
        except Exception as e:
            raise ValueError("unknown rnn_type `%s`" % self.rnn_type) from e

        self.net = nn.Sequential(
            nn.Conv1d(in_channels=self.input_size, out_channels=self.hid_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.hid_size, momentum=0.99, eps=1e-3),
            nn.ReLU(),
        )

        self.fc_out = nn.Linear(in_features=self.hid_size * (1 + self.bi_ind), out_features=1)

        self.rnn = klass(
            input_size=self.input_size,
            hidden_size=self.hid_size,
            num_layers=self.rnn_layer,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        )

        self.lstm_drop = nn.Dropout(p=self.dropout)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def init_weights(self, m):
        import torch.nn.init as init
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

    def forward(self, inputs):
        # inputs: [batch_size, input_size, input_day]
        inputs = inputs.view(len(inputs), self.input_size, -1)
        conv_out = self.net(inputs)  # [batch_size, hid_size, input_day]
        conv_out = self.global_pool(conv_out)

        inputs_lstm = inputs.clone().permute(0, 2, 1)  # [input_size, batch_size, input_day]

        if self.rnn_type.lower() == "lstm":
            rnn_out, (h, c) = self.rnn(inputs_lstm)
            h = torch.cat((h[-2, :, :], h[-1, :, :]), dim=1)
        elif self.rnn_type.lower() == "gru":
            rnn_out, h = self.rnn(inputs_lstm)
            h = torch.cat((h[-2, :, :], h[-1, :, :]), dim=1)
        else:
            raise ValueError("unknown rnn_type `%s`" % self.rnn_type)

        context = self.attention(h, rnn_out, rnn_out)

        out_lstm = self.lstm_drop(context)
        out_lstm = out_lstm.transpose(1, 2)

        out_lstm = self.global_pool(out_lstm)

        out = torch.cat((conv_out, out_lstm), dim=1)
        out = out.view(len(out), -1)
        out = self.fc_out(out)

        return out[...]


# test
if __name__ == "__main__":
    x = torch.randn(1000, 60, 100)
    model = ALSTM3_1(d_feat=100, hidden_size=64, num_layers=2, dropout=0.5,
                       rnn_type="gru", 双向=True)
    y = model(x)
    y = y.detach().numpy()
    print(y.shape)
    print(y)
