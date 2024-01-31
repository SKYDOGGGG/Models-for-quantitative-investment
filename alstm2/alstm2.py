"""
2024.1.24 15:00
ALSTM2

引入Bahdanau Attention机制，模型由卷积网络和引入注意力机制的RNN（GRU/LSTM）组成

Attention用于计算注意力分数和上下文向量。
它使用三个线性层（w_q, w_k, w_v）来计算查询（query）和键（keys）之间的关系，然后产生注意力分数和上下文向量。
上下文向量是每个股票的每个隐藏层在时间维度上的加权值，权重由注意力分数决定，格式为[batch_size, hid_size]。

整体上，输入数据同时通过：
1.卷积神经网络（包含卷积层），然后通过一个全连接层；
2. GRU 网络，其输出与最后一个隐藏状态（在双向rnn中，是最后两个隐藏状态）一起被送入注意力模块，再经过 Dropout 层。

卷积网络和 GRU 网络的输出被合并并通过最后一个全连接层，产生最终的输出。


"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np


class Attention(nn.Module):
    def __init__(self, input_size, bi_ind=1):
        super(Attention, self).__init__()
        self.hid_size = input_size * bi_ind
        self.w_q = nn.Linear(in_features=self.hid_size, out_features=self.hid_size, bias=False)
        self.w_k = nn.Linear(in_features=self.hid_size, out_features=self.hid_size, bias=False)
        self.w_v = nn.Linear(in_features=self.hid_size, out_features=1, bias=False)

    def forward(self, query, keys):
        query = query.unsqueeze(0)  # [1, batch_size, hid_size]
        query = query.permute(1, 0, 2)  # [batch_size, 1, hid_size]
        energy = self.w_v(torch.tanh(self.w_q(query) + self.w_k(keys)))  # [seq_len, batch_size, hid_size]
        attention = energy.squeeze(2)
        attention_weights = F.softmax(attention, dim=1)
        attention_score = attention_weights.unsqueeze(2)  # [seq_len, batch_size, 1]
        context_vector = torch.sum(keys * attention_score, dim=1)  # [batch_size, hid_size]
        return context_vector, attention_weights


class ALSTM2(nn.Module):
    def __init__(self, d_feat=6, hidden_size1=128, hidden_size=64, num_layers=2, dropout=0.0,
                 rnn_type="GRU", 双向=False, **kwargs):
        super().__init__()
        self.hid_size1 = hidden_size1
        self.hid_size2 = hidden_size
        self.input_size = d_feat
        self.dropout = dropout
        self.bidirectional = 双向
        self.bi_ind = 1 + int(双向)
        self.attention = Attention(input_size=self.hid_size2, bi_ind=self.bi_ind)
        self.rnn_type = rnn_type.lower()
        self.rnn_layer = num_layers
        self._build_model()

    def _build_model(self):
        try:
            klass = getattr(nn, self.rnn_type.upper())
        except Exception as e:
            raise ValueError("unknown rnn_type `%s`" % self.rnn_type) from e

        self.net = nn.Sequential(
            nn.Conv1d(in_channels=self.input_size, out_channels=self.hid_size1, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.hid_size1, momentum=0.99, eps=1e-3),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.hid_size1, out_channels=self.hid_size2, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.hid_size2, momentum=0.99, eps=1e-3),
            nn.ReLU(),
        )

        self.fc1 = nn.AdaptiveAvgPool1d(1)
        self.fc_out = nn.Linear(in_features=self.hid_size2 * (1 + self.bi_ind), out_features=1)

        self.rnn = klass(
            input_size=self.input_size,
            hidden_size=self.hid_size2,
            num_layers=self.rnn_layer,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        self.lstm_drop = nn.Dropout(p=self.dropout)

    def forward(self, inputs):
        # inputs: [batch_size, input_size, input_day]
        inputs = inputs.view(len(inputs), self.input_size, -1)
        conv_out = self.net(inputs)  # [batch_size, hid_size2, input_day]
        conv_out = self.fc1(conv_out)

        inputs_lstm = inputs.clone().permute(0, 2, 1)  # [input_size, batch_size, input_day]

        if self.rnn_type.lower() == "lstm":
            rnn_out, (h, c) = self.rnn(inputs_lstm)
            if self.bidirectional:
                h = torch.cat((h[-2, :, :], h[-1, :, :]), dim=1)
            else:
                h = h[-1, :, :]
        elif self.rnn_type.lower() == "gru":
            rnn_out, h = self.rnn(inputs_lstm)
            if self.bidirectional:
                h = torch.cat((h[-2, :, :], h[-1, :, :]), dim=1)
            else:
                h = h[-1, :, :]
        else:
            raise ValueError("unknown rnn_type `%s`" % self.rnn_type)

        context, attention_score = self.attention(h, rnn_out)

        out_lstm = self.lstm_drop(context)

        out = torch.cat((conv_out.squeeze(-1), out_lstm), dim=-1)
        out = self.fc_out(out)
        return out[...]


# test
if __name__ == "__main__":
    x = torch.randn(1000, 20, 100)
    model = ALSTM2(d_feat=100, time_period=20, hidden_size1=128, hidden_size2=64, num_layers=2, dropout=0.6,
                   rnn_type="gru", 双向=True)
    y = model(x)
    print(y.shape)
