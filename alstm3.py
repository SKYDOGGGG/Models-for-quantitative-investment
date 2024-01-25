"""
2024.1.25 14:30
Alstm3

引入多头注意力机制(替换alstm2中的Bahdanau)，模型由卷积网络和引入注意力机制的RNN（GRU或LSTM）组成

多头注意力：
将注意力机制的计算过程分为多个头部，每个头部都有自己的注意力分数和上下文向量，最后将多个头部的上下文向量拼接起来，再通过一个线性层进行变换，
得到最终的上下文向量。

优势：可以多个头独立计算，提高并行度，同时可以让每个头部关注不同的特征。

网络结构变动：改动了注意力类的网络结构，将query/key/value分别过不同的线性层，再以此计算注意力分数和上下文向量，最后拼接多个头部后再经过一个线性层；
              rnn的类型可以采用gru或lstm

更新：取消time_period参数以统一代码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# 加入多头注意力的改进的alstm2
class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.hid_size = input_size
        self.num_heads = num_heads
        self.attention_size = input_size // num_heads
        # query, key, value的线性层
        self.w_q = nn.Linear(in_features=input_size, out_features=input_size)
        self.w_k = nn.Linear(in_features=input_size, out_features=input_size)
        self.w_v = nn.Linear(in_features=input_size, out_features=input_size)

        self.w_o = nn.Linear(in_features=input_size, out_features=input_size)

    def forward(self, query, keys, value):
        batch_size = query.size(0)

        query = self.w_q(query).view(batch_size, -1, self.num_heads, self.attention_size)
        keys = self.w_k(keys).view(batch_size, -1, self.num_heads, self.attention_size)
        value = self.w_v(value).view(batch_size, -1, self.num_heads, self.attention_size)

        attention_score = torch.mul(query, keys)
        attention_score = attention_score / np.sqrt(self.attention_size)
        attention_weights = F.softmax(attention_score, dim=1)

        context_vector = torch.mul(attention_weights, value)
        context_vector = context_vector.contiguous().view(batch_size, -1, self.num_heads * self.attention_size)
        context_vector = self.w_o(context_vector)
        return context_vector


class ALSTMModel(nn.Module):
    def __init__(self, d_feat=6, hidden_size1=128, hidden_size2=64, num_layers=2, dropout=0.0,
                 rnn_type="GRU"):
        super().__init__()
        self.hid_size1 = hidden_size1
        self.hid_size2 = hidden_size2
        self.input_size = d_feat
        self.dropout = dropout
        self.attention = MultiHeadAttention(input_size=self.hid_size2)
        self.rnn_type = rnn_type
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

        self.fc_out = nn.Linear(in_features=self.hid_size2 * 2, out_features=1)

        self.rnn = klass(
            input_size=self.input_size,
            hidden_size=self.hid_size2,
            num_layers=self.rnn_layer,
            batch_first=True,
            dropout=self.dropout,
        )

        self.lstm_drop = nn.Dropout(p=self.dropout)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, inputs):
        # inputs: [batch_size, input_size, input_day]
        inputs = inputs.view(len(inputs), self.input_size, -1)
        conv_out = self.net(inputs)  # [batch_size, hid_size2, input_day]
        conv_out = self.global_pool(conv_out)

        inputs_lstm = inputs.clone().permute(0, 2, 1)  # [input_size, batch_size, input_day]

        if self.rnn_type.lower() == "lstm":
            rnn_out, (h, c) = self.rnn(inputs_lstm)
        elif self.rnn_type.lower() == "gru":
            rnn_out, h = self.rnn(inputs_lstm)
        else:
            raise ValueError("unknown rnn_type `%s`" % self.rnn_type)

        context = self.attention(h[-1], rnn_out, rnn_out)

        out_lstm = self.lstm_drop(context)
        out_lstm = out_lstm.transpose(1, 2)

        out_lstm = self.global_pool(out_lstm)

        out = torch.cat((conv_out, out_lstm), dim=2)
        out = out.view(len(out), -1)
        out = self.fc_out(out)

        return out[...]


# # test
# if __name__ == "__main__":
#     x = torch.randn(1000, 40, 400)
#     model = ALSTMModel(d_feat=400, hidden_size1=128, hidden_size2=64, num_layers=3, dropout=0.6,
#                        rnn_type="gru")
#     y = model(x)
#     y = y.detach().numpy()
#     print(y.shape)
#     print(y)
