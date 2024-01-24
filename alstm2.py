"""
2024.1.24 14:00
Alstm2
引入Bahdanau Attention机制，模型由卷积网络、RNN（GRU）以及注意力机制
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Attention(nn.Module):
    def __init__(self, input_size):
        super(Attention, self).__init__()
        self.hid_size = input_size
        self.w_q = nn.Linear(in_features=input_size, out_features=input_size, bias=False)
        self.w_k = nn.Linear(in_features=input_size, out_features=input_size, bias=False)
        self.w_v = nn.Linear(in_features=input_size, out_features=1, bias=False)

    def forward(self, query, keys):
        query = query.unsqueeze(0) 
        query = query.permute(1, 0, 2)
        energy = self.w_v(torch.tanh(self.w_q(query) + self.w_k(keys)))
        attention = energy.squeeze(2)
        attention_weights = F.softmax(attention, dim=1)
        attention_score = attention_weights.unsqueeze(2)
        context_vector = torch.sum(keys * attention_score, dim=1) 
        return context_vector, attention_weights


class ALSTMModel(nn.Module):
    def __init__(self, d_feat=6, time_period=60, hidden_size1=128, hidden_size2=64, num_layers=2, dropout=0.0,
                 rnn_type="GRU"):
        super().__init__()
        self.hid_size1 = hidden_size1
        self.hid_size2 = hidden_size2
        self.time_period = time_period
        self.input_size = d_feat
        self.dropout = dropout
        self.attention = Attention(input_size=self.hid_size2)
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

        self.fc1 = nn.Linear(in_features=self.time_period, out_features=1)
        self.fc_out = nn.Linear(in_features=self.hid_size2 * 2, out_features=1)

        self.rnn = klass(
            input_size=self.input_size,
            hidden_size=self.hid_size2,
            num_layers=self.rnn_layer,
            batch_first=True,
            dropout=self.dropout,
        )

        self.lstm_drop = nn.Dropout(p=self.dropout)

    def forward(self, inputs):
        # inputs: [batch_size, input_size, input_day]
        inputs = inputs.view(len(inputs), self.input_size, -1)
        conv_out = self.net(inputs)  # [batch_size, hid_size2, input_day]
        conv_out = self.fc1(conv_out)

        inputs_lstm = inputs.clone().permute(0, 2, 1)  # [input_size, batch_size, input_day]

        rnn_out, (h, c) = self.rnn(inputs_lstm)

        context, attention_score = self.attention(h, rnn_out)

        out_lstm = self.lstm_drop(context)

        out = torch.cat((conv_out, out_lstm.unsqueeze(2)), dim=2)
        out = out.view(len(out), -1)
        out = self.fc_out(out)
        return out.squeeze(1)


if __name__ == "__main__":
    x = torch.randn(20000, 20, 300)
    model = ALSTMModel(d_feat=300, time_period=20, hidden_size1=128, hidden_size2=64, num_layers=2, dropout=0.6,
                       rnn_type="GRU")
    y = model(x)
    y = y.detach().numpy()
    print(y.shape)
    print(y)
